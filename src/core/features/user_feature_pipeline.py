# core/features/user_feature_pipeline.py
"""
Pipeline to combine behavior and advanced user metrics into a unified feature set.

This pipeline integrates two main components:
1. UserBehaviorMetrics - Basic user behavior metrics
2. UserAdvancedMetrics - Advanced user metrics and scaled features

The pipeline produces two outputs:
- A combined feature set with raw and derived features
- A scaled version of advanced metrics for machine learning models
"""

from typing import Tuple  # noqa: F401
import os  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import logging  # type: ignore
from IPython.display import display  # type: ignore
from src.utils import get_path  # type: ignore
from .user_behavior_metrics import UserBehaviorMetrics
from .user_advanced_metrics import UserAdvancedMetrics

# Configure logging to display informational messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger(__name__)


class UserFeaturePipeline:
    def __init__(self, df_sessions: pd.DataFrame, df_nc_sessions: pd.DataFrame, df_users: pd.DataFrame):
        self.df_sessions = df_sessions.copy()
        self.df_nc_sessions = df_nc_sessions.copy()
        self.df_users = df_users.copy()

        self.output_dir = os.path.join(get_path("processed"), "features")

        self.behavior = UserBehaviorMetrics(df_sessions, df_nc_sessions, df_users)
        self.advanced = UserAdvancedMetrics(df_sessions, df_nc_sessions, df_users)

        os.makedirs(self.output_dir, exist_ok=True)

    def run(self) -> pd.DataFrame:
        """
        Execute the full feature engineering pipeline.

        Returns:
            pd.DataFrame: Combined feature dataframe with all metrics and derived features.
        """
        print("🔧 Starting full user feature pipeline...")

        try:
            # Step 1: Behavior metrics
            print(" Running behavior metrics...")
            user_base = self.behavior.run()
            print(f"Behavior metrics complete: shape={user_base.shape}")
            display(user_base.head())

            # Update advanced metrics inputs
            self.advanced.df_sessions = self.behavior.df_sessions.copy()
            self.advanced.df_nc_sessions = self.behavior.df_nc_sessions.copy()

            # Step 2: Advanced metrics
            print(" Running advanced metrics...")
            df_advanced_raw = self.advanced.run()
            print(f"Advanced metrics complete: raw_shape={df_advanced_raw.shape}")
            display(df_advanced_raw.head())

            # Step 3: Merge behavior + advanced metrics
            print(" Merging behavior and advanced metrics...")
            df_combined = pd.merge(user_base, df_advanced_raw, on='user_id', how='left')
            # ❌ Do not fill NaNs → preserve missing values
            print(f"Merged dataset shape={df_combined.shape}")
            display(df_combined.head())

            # Step 4: Derived features
            self._calculate_total_spend(df_combined)
            self._calculate_cancellation_rate(df_combined)
            self._calculate_browsing_rate(df_combined)
            self._calculate_business_rate(df_combined)
            self._calculate_group_rate(df_combined)

            # Step 5: Save outputs
            raw_path = os.path.join(self.output_dir, "user_base.csv")
            df_combined.to_csv(raw_path, index=False)
            print(f" Combined feature set saved: {len(df_combined.columns)} features for {len(df_combined)} users")

            return df_combined

        except Exception as e:
            print(f"❌ Error in UserFeaturePipeline: {e}")
            raise

    # -------------------------
    # Derived feature calculators
    # -------------------------
    def _calculate_total_spend(self, df: pd.DataFrame) -> None:
        df['total_spend'] = df['money_spent_hotel_total'] + df['avg_money_spent_flight'] * df['num_flights']

    def _calculate_cancellation_rate(self, df: pd.DataFrame) -> None:
        df['cancellation_rate'] = np.where(
            (df['num_trips'] > 0) & df['num_canceled_trips'].notna(),
            df['num_canceled_trips'] / df['num_trips'],
            np.nan
        )

    def _calculate_browsing_rate(self, df: pd.DataFrame) -> None:
        df['browsing_rate'] = np.where(
            (df['num_sessions'] > 0) & df['num_empty_sessions'].notna(),
            df['num_empty_sessions'] / df['num_sessions'],
            np.nan
        )

    def _calculate_business_rate(self, df: pd.DataFrame) -> None:
        df['business_rate'] = np.where(
            (df['num_trips'] > 0) & df['num_business_trips'].notna(),
            df['num_business_trips'] / df['num_trips'],
            np.nan
        )

    def _calculate_group_rate(self, df: pd.DataFrame) -> None:
        df['group_rate'] = np.where(
            (df['num_trips'] > 0) & df['num_group_trips'].notna(),
            df['num_group_trips'] / df['num_trips'],
            np.nan
        )