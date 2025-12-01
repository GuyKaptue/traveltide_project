# core/segment/non_ml/metrics.py

import numpy as np # type: ignore
import pandas as pd # type: ignore
from typing import List


class MetricsComputer:
    """
    Handles computation of derived behavioral metrics for customer segmentation.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize metrics computer with user dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            User data with raw features
        """
        self.df = df.copy()
    
    def compute_all_metrics(self) -> pd.DataFrame:
        """
        Compute all intermediate metrics required for segmentation.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with added metric columns
        """
        print("[STEP 1] Computing intermediate metrics...")
        
        self._compute_cancellation_rate()
        self._compute_browsing_rate()
        self._compute_group_rate()
        self._compute_business_rate()
        self._compute_total_spend()
        
        print("✅ Intermediate metrics complete")
        return self.df
    
    def _compute_cancellation_rate(self) -> None:
        """Compute cancellation rate: canceled trips / total trips"""
        if self._has_columns(["num_canceled_trips", "num_trips"]):
            self.df["cancellation_rate"] = np.divide(
                self.df["num_canceled_trips"].fillna(0),
                self.df["num_trips"].replace(0, np.nan)
            ).fillna(0)
            print("   ➡️ Added cancellation_rate")
        else:
            self.df["cancellation_rate"] = 0
            print("   ⚠️ Missing columns for cancellation_rate, using default 0")
    
    def _compute_browsing_rate(self) -> None:
        """Compute browsing rate: empty sessions / total sessions"""
        if self._has_columns(["num_empty_sessions", "num_sessions"]):
            self.df["browsing_rate"] = np.divide(
                self.df["num_empty_sessions"].fillna(0),
                self.df["num_sessions"].replace(0, np.nan)
            ).fillna(0)
            print("   ➡️ Computed browsing_rate")
        else:
            self.df["browsing_rate"] = 0
            print("   ⚠️ Missing columns for browsing_rate, using default 0")
    
    def _compute_group_rate(self) -> None:
        """Compute group rate: group trips / total trips"""
        if self._has_columns(["num_group_trips", "num_trips"]):
            self.df["group_rate"] = np.divide(
                self.df["num_group_trips"].fillna(0),
                self.df["num_trips"].replace(0, np.nan)
            ).fillna(0)
            print("   ➡️ Computed group_rate")
        else:
            self.df["group_rate"] = 0
            print("   ⚠️ Missing columns for group_rate, using default 0")
    
    def _compute_business_rate(self) -> None:
        """Compute business rate: business trips / total trips"""
        if self._has_columns(["num_business_trips", "num_trips"]):
            self.df["business_rate"] = np.divide(
                self.df["num_business_trips"].fillna(0),
                self.df["num_trips"].replace(0, np.nan)
            ).fillna(0)
            print("   ➡️ Computed business_rate")
        else:
            self.df["business_rate"] = 0
            print("   ⚠️ Missing columns for business_rate, using default 0")
    
    def _compute_total_spend(self) -> None:
        """Compute total spend: hotel spend + (flight spend × flights)"""
        required_cols = ["money_spent_hotel_total", "avg_money_spent_flight", "num_flights"]
        if self._has_columns(required_cols):
            self.df["total_spend"] = (
                self.df["money_spent_hotel_total"].fillna(0)
                + self.df["avg_money_spent_flight"].fillna(0) * self.df["num_flights"].fillna(0)
            )
            print("   ➡️ Computed total_spend")
        else:
            self.df["total_spend"] = 0
            print("   ⚠️ Missing columns for total_spend, using default 0")
    
    def _has_columns(self, columns: List[str]) -> bool:
        """Check if all required columns exist in dataframe"""
        return all(col in self.df.columns for col in columns)
    
    def get_dataframe(self) -> pd.DataFrame:
        """Return the dataframe with computed metrics"""
        return self.df