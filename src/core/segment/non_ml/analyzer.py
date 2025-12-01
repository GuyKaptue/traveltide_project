# core/segment/non_ml/analyzer.py

import pandas as pd # type: ignore
from typing import Dict, Any


class SegmentationAnalyzer:
    """
    Provides analysis and debugging tools for segmentation results.
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        thresholds: Dict[str, float],
        config: Dict[str, Any]
    ):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        df : pd.DataFrame
            Segmented user data
        thresholds : Dict[str, float]
            Threshold values used
        config : Dict[str, Any]
            Configuration
        """
        self.df = df
        self.thresholds = thresholds
        self.config = config
    
    def analyze_perk_assignments(self) -> pd.DataFrame:
        """
        Analyze and explain why users received each perk.
        
        Returns
        -------
        pd.DataFrame
            Analysis results with assignment reasoning
        """
        print("\n[ANALYSIS] Analyzing perk assignment reasons...")
        
        T = self.thresholds
        analysis_results = []
        
        for perk in self.df['assigned_perk'].unique():
            perk_users = self.df[self.df['assigned_perk'] == perk]
            count = len(perk_users)
            pct = (count / len(self.df)) * 100
            
            reason = self._get_assignment_reason(perk, perk_users, T)
            
            analysis_results.append({
                'Perk': perk,
                'Count': count,
                'Percentage': pct,
                'Assignment Reason': reason
            })
        
        analysis_df = pd.DataFrame(analysis_results)
        
        print("✅ Perk assignment analysis complete")
        return analysis_df
    
    def _get_assignment_reason(
        self, 
        perk: str, 
        perk_users: pd.DataFrame, 
        T: Dict[str, float]
    ) -> str:
        """Generate explanation for why users got this perk"""
        
        if perk == "1 night free hotel plus flight":
            return (
                f"High spend (≥${T['TOTAL_SPEND']:.0f}) + "
                f"Frequent travel (≥{T['TRIP_COUNT']:.0f} trips)"
            )
        
        elif perk == "exclusive discounts":
            high_browsers = len(
                perk_users[perk_users['browsing_rate'] >= T['BROWSING_RATE']]
            )
            return f"High browsing rate ({high_browsers} users ≥{T['BROWSING_RATE']:.2f})"
        
        elif perk == "free checked bags":
            heavy_bags = len(
                perk_users[perk_users['avg_bags'] >= T['AVG_BAGS']]
            )
            group_travel = len(
                perk_users[perk_users['group_rate'] >= T['GROUP_RATE']]
            )
            return (
                f"Heavy baggage ({heavy_bags} users ≥{T['AVG_BAGS']:.1f} bags) OR "
                f"Group travel ({group_travel} users ≥{T['GROUP_RATE']:.2f} rate)"
            )
        
        elif perk == "free hotel meal":
            hotel_spend = len(
                perk_users[
                    perk_users['money_spent_hotel_total'] >= T['HOTEL_SPEND']
                ]
            )
            business_travel = len(
                perk_users[perk_users['business_rate'] >= T['BUSINESS_RATE']]
            )
            return (
                f"Hotel spend ({hotel_spend} users ≥${T['HOTEL_SPEND']:.0f}) OR "
                f"Business travel ({business_travel} users ≥{T['BUSINESS_RATE']:.2f} rate)"
            )
        
        else:  # no cancellation fees (baseline)
            return "Did not meet any higher-tier criteria (baseline)"
    
    def validate_segmentation(self) -> None:
        """Validate segmentation quality and print warnings"""
        print("\n[VALIDATION] Checking segmentation quality...")
        
        total = len(self.df)
        perk_groups = self.config['perk_groups']
        
        for pg in perk_groups:
            group = pg["group"]
            count = len(self.df[self.df["assigned_group"] == group])
            pct = (count / total) * 100
            
            print(f"   {group}: {count:,} ({pct:.1f}%)")
            
            # Issue warnings
            if pct < 1:
                print(f"  ⚠️ Very small segment (<1%)")  # noqa: F541
            elif pct > 60:
                print(f" ⚠️ Dominant segment (>60%)")  # noqa: F541
        
        # Check for unassigned users
        unassigned = self.df["assigned_perk"].isna().sum()
        if unassigned > 0:
            print(f"   ❌ {unassigned} users without perk assignment!")
        else:
            print("   ✅ All users successfully assigned")
    
    def create_segment_summary(self) -> pd.DataFrame:
        """
        Create comprehensive segment summary table.
        
        Returns
        -------
        pd.DataFrame
            Summary with key statistics per segment
        """
        perk_groups = self.config['perk_groups']
        total_users = len(self.df)
        
        summary_data = []
        for pg in perk_groups:
            perk = pg["perk"]
            segment = pg["group"]
            
            cluster_df = self.df[self.df["assigned_perk"] == perk]
            count = len(cluster_df)
            pct = (count / total_users * 100) if total_users else 0
            
            avg_spend = (
                f"${cluster_df['total_spend'].mean():.0f}"
                if "total_spend" in self.df.columns and count > 0 
                else "N/A"
            )
            avg_trips = (
                f"{cluster_df['num_trips'].mean():.1f}"
                if "num_trips" in self.df.columns and count > 0 
                else "N/A"
            )
            
            summary_data.append({
                "Segment": segment,
                "Assigned Perk": perk,
                "Count": count,
                "Percentage": f"{pct:.1f}%",
                "Avg Spend": avg_spend,
                "Avg Trips": avg_trips
            })
        
        return pd.DataFrame(summary_data)
    
    def get_threshold_statistics(self) -> pd.DataFrame:
        """
        Get statistics about how many users meet each threshold.
        
        Returns
        -------
        pd.DataFrame
            Threshold coverage statistics
        """
        stats = []
        
        threshold_defs = self.config['threshold_definitions']
        for name, config in threshold_defs.items():
            column = config['column']
            threshold = self.thresholds[name]
            
            if column in self.df.columns:
                above_threshold = (self.df[column] >= threshold).sum()
                pct = (above_threshold / len(self.df)) * 100
                
                stats.append({
                    'Threshold': name,
                    'Column': column,
                    'Value': threshold,
                    'Users Above': above_threshold,
                    'Percentage': pct
                })
        
        return pd.DataFrame(stats)