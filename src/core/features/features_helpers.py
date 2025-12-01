 
# core/features/features_helpers.py
"""Helper methods for feature engineering."""
import os
import math
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import urllib.parse
from src.utils import get_path  # type: ignore  # noqa: F401
from dotenv import load_dotenv # type: ignore


# Load .env once
load_dotenv()
 
 # --- HELPER METHODS ---
 
def build_geoapify_url(city: str) -> str:
    """Build Geoapify autocomplete URL for a given city."""
    api_key = os.getenv("GEOAPIFY_API_KEY")
    if not api_key:
        raise ValueError("Missing GEOAPIFY_API_KEY in environment")

   
    encoded_city = urllib.parse.quote(city)
    base_url = "https://api.geoapify.com/v1/geocode/autocomplete"
    return f"{base_url}?text={encoded_city}&apiKey={api_key}"

def haversine(lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates using Haversine formula."""
        R = 6371  # Earth radius in kilometers
        try:
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            return R * c
        except (ValueError, TypeError):
            return np.nan

def get_age(row):
        """Calculate age from birthdate."""
        try:
            birth_date = pd.to_datetime(row['birthdate'], format='mixed', errors='coerce')
            if pd.isnull(birth_date):
                return np.nan
            return (pd.Timestamp.now().normalize() - birth_date).days / 365
        except:  # noqa: E722
            return np.nan

def is_group_trip(row):
        """Group trip: multiple seats and rooms."""
        return int(
            row.get('flight_booked', False) and
            row.get('return_flight_booked', False) and
            row.get('hotel_booked', False) and
            row.get('seats', 0) > 2 and
            row.get('rooms', 0) > 1
        )

def is_pair_trip(row):
        """Couple trip: exactly 2 seats and 1 room."""
        return int(
            row.get('flight_booked', False) and
            row.get('return_flight_booked', False) and
            row.get('hotel_booked', False) and
            row.get('seats', 0) == 2 and
            row.get('rooms', 0) == 1
        )

def is_business_week_trip(row):
        """Business trip: weekday travel, short stays, age 25–60."""
        age = get_age(row)
        departure = row.get('departure_time')
        return_ = row.get('return_time')

        if pd.isnull(departure) or pd.isnull(return_):
            return 0

        return int(
            row.get('flight_booked', False) and
            row.get('return_flight_booked', False) and
            row.get('hotel_booked', False) and
            row.get('seats', 0) == 1 and
            row.get('nights', 0) >= 1 and
            row.get('nights', 0) < 6 and
            25 <= age <= 60 and
            departure.weekday() <= 4 and
            return_.weekday() <= 4
        )

def is_weekend_trip_new(row):
        """Weekend trip: ≤2 nights, Fri–Sun travel."""
        departure = row.get('departure_time')
        return_ = row.get('return_time')

        if pd.isnull(departure) or pd.isnull(return_):
            return 0

        return int(
            row.get('flight_booked', False) and
            row.get('return_flight_booked', False) and
            row.get('hotel_booked', False) and
            row.get('nights', 0) <= 2 and
            departure.weekday() >= 4 and
            return_.weekday() <= 6
        )

def is_discount_trip_new(row):
        """Trip with any discount applied."""
        return int(row.get('hotel_discount', False) or row.get('flight_discount', False))

def get_season(dt):
        """Determine season based on departure month."""
        if pd.isnull(dt):
            return "unknown"
        if dt.month in [12, 1, 2]:
            return "winter"
        elif dt.month in [6, 7, 8]:
            return "summer"
        elif dt.month in [9, 10, 11]:
            return "fall"
        else:
            return "spring"
