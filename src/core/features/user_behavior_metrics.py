# core/features/user_behavior_metrics.py
"""Module to compute user behavior metrics from session and trip data."""

# Standard library imports
import os # For interacting with the operating system (e.g., file paths)

# Third-party imports
import pandas as pd # type: ignore # Data manipulation and analysis
import numpy as np  # type: ignore # Numerical operations, especially for conditional logic
from sklearn.preprocessing import OneHotEncoder # type: ignore # For converting categorical features to numerical

# Project-specific utility imports
from src.utils import get_path  # Helper to get standard project paths

# Imported helper functions (Used directly in the class methods)
from .features_helpers import (
    haversine,            # For calculating distance
    get_age,              # (Kept for import, though direct age calculation is used)  # noqa: F401
    is_group_trip,        # For identifying group trips
    is_pair_trip,         # For identifying pair trips
    is_business_week_trip,# For identifying business weekday trips
    is_weekend_trip_new,  # For identifying weekend trips
    is_discount_trip_new, # For identifying discount trips
    get_season            # For determining trip season
)

class UserBehaviorMetrics:
    """
    Computes various user behavior metrics based on session, trip, and user data.
    The metrics cover user activity (sessions) and trip characteristics (bookings).
    """

    def __init__(self, df_sessions, df_nc_sessions, df_users):
        """
        Initializes the UserBehaviorMetrics class with raw dataframes.

        Args:
            df_sessions (pd.DataFrame): Raw session data (including sessions with no trip).
            df_nc_sessions (pd.DataFrame): Raw non-canceled trip data.
            df_users (pd.DataFrame): Raw user profile data.
        """
        # Create copies to ensure original dataframes are not modified
        self.df_sessions = df_sessions.copy()
        self.df_nc_sessions = df_nc_sessions.copy()
        self.df_users = df_users.copy()
        # Define the output directory path for saving the final user base file
        self.output_dir = os.path.join(get_path("processed"), "features")
        
        os.makedirs(self.output_dir, exist_ok=True)

    def preprocess_dates(self):
        """Converts relevant date columns in both session and non-canceled trip dataframes to datetime objects."""
        print("Preprocessing dates...")

        date_cols_sessions = ['session_start', 'session_end', 'departure_time']
        date_cols_nc_sessions = ['session_end', 'departure_time', 'return_time', 'check_in_time']

        # Process session dataframe
        for col in date_cols_sessions:
            if col in self.df_sessions.columns:
                # Convert to datetime, coercing errors to NaT (Not a Time)
                self.df_sessions[col] = pd.to_datetime(
                    self.df_sessions[col], errors='coerce', format='mixed'
                )

        # Process non-canceled session (trip) dataframe
        for col in date_cols_nc_sessions:
            if col in self.df_nc_sessions.columns:
                self.df_nc_sessions[col] = pd.to_datetime(
                    self.df_nc_sessions[col], errors='coerce', format='mixed'
                )

    def enrich_sessions(self):
        """
        Enriches the session dataframe with derived features:
        - window_shopping (session without a trip_id).
        - canceled_trip (session with a trip_id not found in non-canceled trips).
        - session_duration (in seconds).
        """
        print("Enriching session data with window shopping and cancellation flags...")
        # Get unique trip IDs from non-canceled trips for reference
        nc_trip_ids = self.df_nc_sessions['trip_id'].unique()

        # 'window_shopping': 1 if 'trip_id' is missing (no trip booked in session)
        self.df_sessions['window_shopping'] = self.df_sessions['trip_id'].isna().astype(int)

        # 'canceled_trip': 1 if a trip_id exists but is not in the non-canceled trip list
        self.df_sessions['canceled_trip'] = self.df_sessions['trip_id'].apply(
            lambda x: 0 if pd.isna(x) else int(x not in nc_trip_ids)
        )

        # Calculate session duration in seconds
        self.df_sessions['session_duration'] = (
            self.df_sessions['session_end'] - self.df_sessions['session_start']
        ).dt.total_seconds()

    def aggregate_user_sessions(self):
        """
        Aggregates session-level metrics by user_id to compute user activity behavior.

        Returns:
            pd.DataFrame: A dataframe with user_id and aggregated session metrics.
        """
        print("Aggregating session metrics by user...")
        return self.df_sessions.groupby('user_id').agg(
            num_clicks=('page_clicks', 'sum'),
            avg_session_clicks=('page_clicks', 'mean'),
            max_session_clicks=('page_clicks', 'max'),
            num_empty_sessions=('window_shopping', 'sum'), # Total window shopping sessions
            num_canceled_trips=('canceled_trip', 'sum'), # Total canceled trips initiated in sessions
            num_sessions=('session_id', 'nunique'),
            avg_session_duration=('session_duration', 'mean')
        ).reset_index()

    def enrich_trips(self):
        """
        Enriches the non-canceled trip dataframe with various trip characteristics:
        - Flight and hotel counts/costs, discounts applied.
        - Booking time lag (time_after_booking), trip length, international status.
        - Trip type flags (group, pair, business, weekend, discount) using direct imports.
        - Season (using one-hot encoding) using direct import.
        - Distance (using Haversine formula) using direct import.
        """
        print("Enriching trip data with derived features and costs...")
        df = self.df_nc_sessions.copy()

        # --- Cost and Booking Calculation ---
        # ... (num_flights, num_hotels, money_spent_per_flight, money_spent_per_seat, money_spent_hotel calculations remain here)

        # Number of flights booked (return flight counts as a second flight)
        df['num_flights'] = np.where(
            (df['flight_booked']) & (df['return_flight_booked']), 2,
            np.where(df['flight_booked'], 1, 0)
        )
        # Number of hotels booked (1 or 0)
        df['num_hotels'] = df['hotel_booked'].astype(int)

        # Calculate money spent per flight, considering flight discount
        df['money_spent_per_flight'] = np.where(
            df['flight_discount'],
            df['base_fare_usd'] * (1 - df['flight_discount_amount']),
            df['base_fare_usd']
        )
        # Calculate money spent per seat
        # Handle division by zero: use 1 for seats if 0 is present
        df['money_spent_per_seat'] = df['money_spent_per_flight'] / df['seats'].replace(0, 1)

        # Calculate base hotel cost (Price * Nights * Rooms)
        base_hotel_cost = df['hotel_price_per_room_night_usd'] * df['nights'] * df['rooms']
        # Calculate total money spent on hotel, considering hotel discount
        df['money_spent_hotel'] = np.where(
            df['hotel_discount'],
            base_hotel_cost * (1 - df['hotel_discount_amount']),
            base_hotel_cost
        )


        # --- Time and Location Metrics ---
        # Time difference in days between booking and departure
        df['time_after_booking'] = (df['departure_time'] - df['session_end']).dt.days
        # Trip length in days
        df['trip_length_days'] = (df['return_time'] - df['departure_time']).dt.days
        
        # *** FIX for KeyError: 'destination_country' ***
        # Check if the columns exist before attempting the string comparison
        if 'home_country' in df.columns and 'destination_country' in df.columns:
            df['is_international'] = df['home_country'].str.lower() != df['destination_country'].str.lower()
        else:
            # Set to NaN or False if the data is unavailable to prevent crashing
            print("⚠️ Warning: 'destination_country' or 'home_country' column missing. 'is_international' set to NaN.")
            df['is_international'] = np.nan
        # *** END FIX ***
        
        # Extract booking year
        df['booking_year'] = df['session_end'].dt.year

        # --- Trip Type Flags (using imported helpers directly) ---
        df['group_trip'] = df.apply(is_group_trip, axis=1) # Using imported function directly
        df['pair_trip'] = df.apply(is_pair_trip, axis=1) # Using imported function directly
        df['business_week_trip'] = df.apply(is_business_week_trip, axis=1) # Using imported function directly
        df['weekend_trip'] = df.apply(is_weekend_trip_new, axis=1) # Using imported function directly
        df['discount_trip'] = df.apply(is_discount_trip_new, axis=1) # Using imported function directly
        df['season'] = df['departure_time'].apply(get_season) # Using imported function directly

        # --- Season One-Hot Encoding ---
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        season_encoded = encoder.fit_transform(df[['season']].fillna('unknown'))
        season_cols = encoder.get_feature_names_out(['season'])
        df_season_encoded = pd.DataFrame(season_encoded, columns=season_cols, index=df.index)
        df = df.join(df_season_encoded)
        # Drop the original categorical 'season' column
        df.drop(columns='season', inplace=True, errors='ignore')

        # --- Distance Calculation (using imported Haversine directly) ---
        df['distance_km'] = df.apply(
            lambda row: haversine( # Using imported function directly
                row['home_airport_lat'], row['home_airport_lon'],
                row['destination_airport_lat'], row['destination_airport_lon']
            ) if not pd.isnull(row['home_airport_lat']) and not pd.isnull(row['destination_airport_lat']) else np.nan,
            axis=1
        )

        self.df_nc_sessions = df # Update the instance variable

    def aggregate_user_trips(self):
        """
        Aggregates trip-level metrics by user_id to compute user trip preferences and spending.

        Returns:
            pd.DataFrame: A dataframe with user_id and aggregated trip metrics (user_base_2).
        """
        print("Aggregating trip metrics by user...")
        agg_params = {
            # 1. VOLUME & DIVERSITY
            'num_trips': ('trip_id', 'nunique'),
            'num_destinations': ('destination', 'nunique'),
            'num_flights': ('num_flights', 'sum'),
            'num_hotels': ('num_hotels', 'sum'),
            
            # 2. TRIP TYPE & PATTERNS
            'num_group_trips': ('group_trip', 'sum'),
            'num_pair_trips': ('pair_trip', 'sum'),
            'num_business_trips': ('business_week_trip', 'sum'),
            'num_weekend_trips_agg': ('weekend_trip', 'sum'),
            'num_discount_trips_agg': ('discount_trip', 'sum'),
            
            # 3. FINANCIAL & SPEND
            'money_spent_hotel_total': ('money_spent_hotel', 'sum'),
            'avg_money_spent_flight': ('money_spent_per_flight', 'mean'),
            'avg_money_spent_hotel_trip': ('money_spent_hotel', 'mean'),
            'avg_money_spent_per_seat': ('money_spent_per_seat', 'mean'),
            
            # 4. LOGISTICS & GEOGRAPHY
            'avg_km_flown': ('distance_km', 'mean'),
            'avg_bags': ('checked_bags', 'mean'),
            'international_ratio': ('is_international', 'mean'),
            
            # 5. TIMING & DURATION
            'avg_time_after_booking': ('time_after_booking', 'mean'),
            'avg_trip_length': ('trip_length_days', 'mean')
        }

        # Include aggregation for season-specific trip counts (from one-hot encoding)
        season_cols = [col for col in self.df_nc_sessions.columns if col.startswith('season_')]
        season_agg_params = {
            f'num_{col}': (col, 'sum')
            for col in season_cols
        }

        # Combine all aggregation parameters
        agg_params.update(season_agg_params)
        user_base = self.df_nc_sessions.groupby('user_id').agg(**agg_params).reset_index()

        # --- Booking Growth Calculation ---
        def compute_growth(group):
            """Calculates the booking growth rate between the last two years with bookings."""
            counts = group['booking_year'].value_counts().sort_index()
            if len(counts) >= 2:
                # Growth = (Count_LatestYear - Count_PreviousYear) / Count_PreviousYear
                growth = (counts.iloc[-1] - counts.iloc[-2]) / counts.iloc[-2]
                return growth
            # Return 0.0 if not enough data points (years) for growth calculation
            return 0.0

        # Apply the growth calculation per user
        growth_df = self.df_nc_sessions.groupby('user_id').apply(compute_growth).reset_index(name='booking_growth')
        # Merge growth back into the user base
        user_base = pd.merge(user_base, growth_df, on='user_id', how='left')

        return user_base

    def finalize_user_table(self, user_base_sessions, user_base_trips):
        """
        Merges session and trip metrics with user profile data, calculates age,
        derives a user persona, and saves the final table.

        Args:
            user_base_sessions (pd.DataFrame): Aggregated session metrics.
            user_base_trips (pd.DataFrame): Aggregated trip metrics.

        Returns:
            pd.DataFrame: The final, merged, and enriched user base table.
        """
        print("Merging user data and deriving features (Age, Persona)...")

        # --- Age Calculation ---
        self.df_users['birthdate'] = pd.to_datetime(self.df_users['birthdate'], format='mixed', errors='coerce')
        today = pd.Timestamp.now().normalize()
        # Calculate age in years (using direct date difference calculation)
        self.df_users['age'] = (today - self.df_users['birthdate']).dt.days / 365

        # --- Merge Dataframes ---
        df_user_base = pd.merge(user_base_sessions, user_base_trips, on='user_id', how='left')
        df_user_base = pd.merge(df_user_base, self.df_users, on='user_id', how='left')

        # Fill NaNs created by left joins with 0 for trip-related metrics where appropriate
        numeric_cols_to_fill = [col for col in df_user_base.columns if col not in self.df_users.columns and df_user_base[col].dtype in [np.float64, np.int64]]
        df_user_base[numeric_cols_to_fill] = df_user_base[numeric_cols_to_fill].fillna(0)

        # --- Global Share Calculation ---
        # Calculate the user's share of total trips booked across all users
        df_user_base['global_booking_share'] = df_user_base['num_trips'] / df_user_base['num_trips'].sum()

        # --- Persona Derivation ---
        def infer_persona_type(row):
            """Infers a simple user persona based on trip and profile characteristics."""
            # Priority 1: Group Traveler
            if row.get('num_group_trips', 0) >= 1:
                return 'Group'
            # Priority 2: Couple Traveler
            elif row.get('num_pair_trips', 0) >= 1:
                return 'Couple'
            # Priority 3: Family Traveler (if has children)
            elif row.get('has_children', False):
                return 'Family'
            # Priority 4: Business Traveler (high frequency and long distance)
            elif row.get('num_trips', 0) >= 5 and row.get('avg_km_flown', 0) > 1000:
                return 'Business'
            # Priority 5: Weekender (high weekend trip count)
            elif row.get('num_weekend_trips_agg', 0) >= 2:
                return 'Weekender'
            # Default
            else:
                return 'Solo'

        df_user_base['persona_type'] = df_user_base.apply(infer_persona_type, axis=1)

        # --- Cleanup and Save ---
        # Drop original/intermediate columns
        drop_cols = ['birthdate', 'home_airport', 'home_airport_lat', 'home_airport_lon', 'sign_up_date']
        df_user_base.drop(columns=drop_cols, inplace=True, errors='ignore')

        # Save final enriched user base to CSV
        save_path = os.path.join(self.output_dir, "user_behavior_metrics.csv")
        df_user_base.to_csv(save_path, index=False)

        print(f"✅ User base saved to {save_path}")
        print(f"Rows: {len(df_user_base)}, Columns: {len(df_user_base.columns)}")

        return df_user_base

    def run(self):
        """
        Executes the full pipeline for computing and finalizing user behavior metrics.

        Returns:
            pd.DataFrame: The final user base dataframe.
        """
        print("--- Start: Calculating User Behavior Metrics ---")
        self.preprocess_dates()

        print("1/5: Enriching session data...")
        self.enrich_sessions()

        print("2/5: Aggregating session metrics...")
        user_base_sessions = self.aggregate_user_sessions()

        print("3/5: Enriching trip data (Types, Costs, Distance)...")
        self.enrich_trips()

        print("4/5: Aggregating trip metrics...")
        user_base_trips = self.aggregate_user_trips()

        print("5/5: Finalizing user table and saving...")
        final_df = self.finalize_user_table(user_base_sessions, user_base_trips)

        print("--- End: User Behavior Metrics Pipeline Completed ---")
        return final_df
