# core/segment/non_ml/perk_assigner.py

import numpy as np # type: ignore
import pandas as pd # type: ignore
from typing import Dict, List, Any


class PerkAssigner:
    """
    Handles perk assignment logic based on tier-based business rules.
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        thresholds: Dict[str, float],
        config: Dict[str, Any]
    ):
        """
        Initialize perk assigner.
        
        Parameters
        ----------
        df : pd.DataFrame
            User data with computed metrics
        thresholds : Dict[str, float]
            Computed threshold values
        config : Dict[str, Any]
            Configuration with perk definitions
        """
        self.df = df.copy()
        self.thresholds = thresholds
        self.config = config
        self.perk_groups = config['perk_groups']
        self.perks = [pg["perk"] for pg in self.perk_groups]
        self.groups = [pg["group"] for pg in self.perk_groups]
    
    def assign_perks(self) -> pd.DataFrame:
        """
        Assign perks using mutually exclusive tier-based rules.
        
        Returns
        -------
        pd.DataFrame
            Distribution summary of perk assignments
        """
        print("[STEP 3] Assigning perks...")
        
        conditions = self._build_tier_conditions()
        self._apply_assignments(conditions)
        distribution = self._calculate_distribution()
        
        self._print_distribution(distribution)
        
        print("✅ Perks assigned")
        return distribution
    
    def _build_tier_conditions(self) -> List[pd.Series]:
        """
        Build mutually exclusive tier conditions.
        
        Returns
        -------
        List[pd.Series]
            List of boolean Series for each tier
        """
        df = self.df
        T = self.thresholds
        
        # Tier 1: VIP High-Frequency Spenders
        tier1 = (
            (df["total_spend"] >= T["TOTAL_SPEND"]) &
            (df["num_trips"] >= T["TRIP_COUNT"])
        )
        
        # Tier 2: High-Intent Browsers (exclude Tier 1)
        tier2 = (
            (df["browsing_rate"] >= T["BROWSING_RATE"])
        ) & ~tier1
        
        # Tier 3: Group & Family / Heavy Baggage (exclude higher tiers)
        tier3 = (
            (df["avg_bags"] >= T["AVG_BAGS"]) |
            (df["group_rate"] >= T["GROUP_RATE"])
        ) & ~tier1 & ~tier2
        
        # Tier 4: Hotel & Business Focused (exclude higher tiers)
        tier4 = (
            (df["money_spent_hotel_total"] >= T["HOTEL_SPEND"]) |
            (df["business_rate"] >= T["BUSINESS_RATE"])
        ) & ~tier1 & ~tier2 & ~tier3
        
        # Baseline: Everyone else
        baseline = ~tier1 & ~tier2 & ~tier3 & ~tier4
        
        return [tier1, tier2, tier3, tier4, baseline]
    
    def _apply_assignments(self, conditions: List[pd.Series]) -> None:
        """
        Apply perk and group assignments to dataframe.
        
        Parameters
        ----------
        conditions : List[pd.Series]
            Boolean conditions for each tier
        """
        self.df["assigned_perk"] = np.select(
            conditions, 
            self.perks, 
            default=self.perks[-1]
        )
        self.df["assigned_group"] = np.select(
            conditions, 
            self.groups, 
            default=self.groups[-1]
        )
    
    def _calculate_distribution(self) -> pd.DataFrame:
        """
        Calculate perk distribution summary.
        
        Returns
        -------
        pd.DataFrame
            Distribution with counts and percentages
        """
        total_users = len(self.df)
        
        distribution = (
            self.df.groupby(["assigned_perk"])
            .size()
            .reset_index(name="Count")
        )
        
        distribution["Percentage"] = (distribution["Count"] / total_users) * 100
        
        # Add group names (1:1 mapping)
        distribution = distribution.merge(
            self.df[["assigned_perk", "assigned_group"]].drop_duplicates(),
            on="assigned_perk",
            how="left"
        )
        
        # Reorder columns
        distribution = distribution[
            ["assigned_group", "assigned_perk", "Count", "Percentage"]
        ]
        
        return distribution
    
    def _print_distribution(self, distribution: pd.DataFrame) -> None:
        """Print distribution summary to console"""
        print("   Distribution (with mutual exclusivity):")
        for _, row in distribution.iterrows():
            print(
                f"   - {row['assigned_group']}: "
                f"{row['Count']:,} users ({row['Percentage']:.1f}%)"
            )
    
    def get_dataframe(self) -> pd.DataFrame:
        """Return dataframe with assignments"""
        return self.df
    
    def validate_assignments(self) -> None:
        """Validate that all users have been assigned"""
        unassigned = self.df["assigned_perk"].isna().sum()
        if unassigned > 0:
            print(f"   ⚠️ WARNING: {unassigned} users without assignment!")
        
        # Check for segment sizes
        total = len(self.df)
        for group in self.groups:
            count = len(self.df[self.df["assigned_group"] == group])
            pct = (count / total) * 100
            if pct < 1:
                print(f"   ⚠️ Very small segment: {group} ({pct:.2f}%)")
            elif pct > 60:
                print(f"   ⚠️ Dominant segment: {group} ({pct:.1f}%)")