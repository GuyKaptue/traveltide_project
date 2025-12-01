# core/features/user_advanced_metrics.py
"""Module to compute advanced user metrics for perk-based segmentation."""

import os
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import requests  # type: ignore
from requests.structures import CaseInsensitiveDict  # type: ignore
from src.utils import get_path
from .features_helpers import build_geoapify_url, haversine


class UserAdvancedMetrics:
    def __init__(self, df_sessions, df_nc_sessions, df_users, sustainable_threshold: float = 1500.0):
        self.df_sessions = df_sessions.copy()
        self.df_nc_sessions = df_nc_sessions.copy()
        self.df_users = df_users.copy()
        self.output_dir = os.path.join(get_path("processed"), "feature")
        os.makedirs(self.output_dir, exist_ok=True)

        # Hotel-only bookings (hotel booked, flight not booked)
        self.df_bookins_hotel_only = self.df_nc_sessions[
            (self.df_nc_sessions['hotel_booked'] == True) &  # noqa: E712
            (self.df_nc_sessions['flight_booked'] == False)  # noqa: E712
        ].copy()

        self.sustainable_threshold = sustainable_threshold

    # -------------------------
    # Discount metrics
    # -------------------------
    def compute_discount_metrics(self):
        df = self.df_nc_sessions.copy()
        df['dollars_saved_per_km'] = (
            df['flight_discount_amount'] * df['base_fare_usd']
        ) / df['distance_km'].replace(0, np.nan)

        df['bargain_hunter_index'] = (
            df['flight_discount'].astype(float) *
            df['flight_discount_amount'] *
            df['dollars_saved_per_km']
        )

        return df.groupby('user_id').agg(
            avg_dollars_saved_per_km=('dollars_saved_per_km', 'mean'),
            bargain_hunter_index=('bargain_hunter_index', 'mean')
        ).reset_index()

    # -------------------------
    # Browsing behavior
    # -------------------------
    def compute_browsing_behavior(self):
        df = self.df_sessions.copy()
        browsing = df[df['window_shopping'] == 1].groupby('user_id').agg(
            num_browsing_sessions=('session_id', 'count'),
            avg_browsing_duration=('session_duration', 'mean'),
            total_browsing_clicks=('page_clicks', 'sum'),
            browsing_intensity=('page_clicks', lambda x: x.sum() / x.count() if x.count() > 0 else np.nan)
        ).reset_index()

        total_sessions = df.groupby('user_id').size().rename("total_sessions")
        booking_sessions = df[df['trip_id'].notna()].groupby('user_id').size().rename("booking_sessions")

        conversion = (booking_sessions / total_sessions).rename("conversion_rate")
        browsing = browsing.merge(conversion, on="user_id", how="left")

        return browsing

    # -------------------------
    # RFM metrics
    # -------------------------
    def compute_rfm_metrics(self):
        df = self.df_nc_sessions.copy()
        reference_date = df['session_end'].max()

        rfm = df.groupby('user_id').agg(
            recency=('session_end', lambda x: (reference_date - x.max()).days if len(x) > 0 else np.nan),
            frequency=('trip_id', 'nunique'),
            monetary=('money_spent_per_flight', lambda x: x.sum() + df.loc[x.index, 'money_spent_hotel'].sum())
        ).reset_index()

        for col, labels in [
            ('recency', [5, 4, 3, 2, 1]),
            ('frequency', [1, 2, 3, 4, 5]),
            ('monetary', [1, 2, 3, 4, 5])
        ]:
            unique_vals = rfm[col].nunique(dropna=True)
            if unique_vals < 2:
                rfm[f'{col[0].upper()}_score'] = np.nan
            else:
                n_bins = min(len(labels), unique_vals)
                try:
                    temp = pd.qcut(rfm[col], n_bins, labels=labels[:n_bins], duplicates='drop')
                    rfm[f'{col[0].upper()}_score'] = temp.astype(float)
                except ValueError:
                    rfm[f'{col[0].upper()}_score'] = pd.Series(
                        np.floor(pd.Series(rfm[col]).rank(method='min') / (len(rfm[col]) / n_bins))
                    )

        rfm['RFM_score'] = rfm[['R_score', 'F_score', 'M_score']].sum(axis=1, min_count=1)
        return rfm[['user_id', 'RFM_score']]

    # -------------------------
    # Hotel coordinates + distance
    # -------------------------
    def fetch_hotel_coordinates(self):
        coord_path = os.path.join(self.output_dir, "hotel_to_coordinates.csv")
        if os.path.exists(coord_path):
            print(f"📂 Loading existing hotel coordinates from {coord_path}")
            df_hotel_to_coordinates = pd.read_csv(coord_path)
        else:
            headers = CaseInsensitiveDict()
            headers["Accept"] = "application/json"

            df_hotel_to_coordinates = pd.DataFrame(columns=['hotel_name', 'hotel_lat', 'hotel_lon'])
            hotels = self.df_bookins_hotel_only['hotel_name'].unique()

            for i, hotel in enumerate(hotels):
                try:
                    parts = hotel.split(' - ')
                    city = parts[1].strip() if len(parts) > 1 else parts[0].strip()
                    url = build_geoapify_url(city)
                    resp = requests.get(url, headers=headers)
                    data = resp.json()

                    if "features" in data and len(data["features"]) > 0:
                        lon = data['features'][0]['properties']['lon']
                        lat = data['features'][0]['properties']['lat']
                        df_hotel_to_coordinates.loc[i] = [hotel, lat, lon]
                    else:
                        df_hotel_to_coordinates.loc[i] = [hotel, np.nan, np.nan]
                except Exception:
                    df_hotel_to_coordinates.loc[i] = [hotel, np.nan, np.nan]

            df_hotel_to_coordinates.to_csv(coord_path, index=False)

        self.df_bookins_hotel_only = pd.merge(
            self.df_bookins_hotel_only, df_hotel_to_coordinates, on="hotel_name", how="left"
        )

        self.df_bookins_hotel_only['distance_km'] = self.df_bookins_hotel_only.apply(
            lambda row: haversine(
                row['home_airport_lat'], row['home_airport_lon'],
                row['hotel_lat'], row['hotel_lon']
            ) if pd.notna(row['hotel_lat']) and pd.notna(row['hotel_lon']) else np.nan,
            axis=1
        )

        self.df_bookins_hotel_only.to_csv(
            os.path.join(self.output_dir, "hotel_only_bookings_with_distance.csv"), index=False
        )

        return self.df_bookins_hotel_only

    # -------------------------
    # Sustainable metrics
    # -------------------------
    def compute_sustainable_metrics(self):
        df = self.df_bookins_hotel_only.copy()
        df['sustainable'] = np.where(df['distance_km'] <= self.sustainable_threshold, 1,
                                     np.where(pd.isna(df['distance_km']), np.nan, 0))

        user_base1 = df.groupby('user_id').agg(
            sustainable_index=('sustainable', 'mean')
        ).reset_index()

        df.to_csv(os.path.join(self.output_dir, "hotel_only_sustainable_trips.csv"), index=False)
        user_base1.to_csv(os.path.join(self.output_dir, "user_sustainable_index.csv"), index=False)

        return user_base1

    # -------------------------
    # Pipeline runner
    # -------------------------
    def run(self):
        print("--- Start: Calculating Advanced User Metrics ---")

        print("1/6: Fetching hotel coordinates and computing distances...")
        self.fetch_hotel_coordinates()

        print("2/6: Computing discount metrics...")
        discount = self.compute_discount_metrics()
        print(f"  Discount metrics shape: {discount.shape}")

        print("3/6: Computing browsing behavior metrics...")
        browsing = self.compute_browsing_behavior()
        print(f"  Browsing metrics shape: {browsing.shape}")

        print("4/6: Computing RFM metrics...")
        rfm = self.compute_rfm_metrics()
        print(f"   RFM metrics shape: {rfm.shape}")

        print("6/6: Computing sustainable travel metrics...")
        sustainable = self.compute_sustainable_metrics()
        print(f"  Sustainable metrics shape: {sustainable.shape}")

        # Merge all metrics together
        print(" Merging all advanced metrics...")
        df_final = discount
        for df in [browsing, rfm, sustainable]:
            df_final = pd.merge(df_final, df, on='user_id', how='outer')
        print(f"  Final merged shape: {df_final.shape}")
        
        # Save outputs
        raw_path = os.path.join(self.output_dir, "user_advanced_metrics_raw.csv")

        df_final.to_csv(raw_path, index=False)

        print("--- End: Advanced User Metrics Pipeline Completed ---")
        print(f"✅ Advanced metrics complete: {len(df_final.columns)} features for {len(df_final)} users")

        return df_final
