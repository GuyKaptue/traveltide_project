# core/segment/non_ml/data_manager.py

import os
import pandas as pd # type: ignore
from typing import Dict, Any


class DataManager:
    """
    Handles all data saving and export operations for segmentation results.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize data manager.
        
        Parameters
        ----------
        output_dir : str
            Directory path for saving output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_all_results(
        self,
        df: pd.DataFrame,
        thresholds: Dict[str, float],
        config: Dict[str, Any]
    ) -> str:
        """
        Save all segmentation results to CSV files.
        
        Parameters
        ----------
        df : pd.DataFrame
            Complete dataframe with assignments and metrics
        thresholds : Dict[str, float]
            Computed thresholds
        config : Dict[str, Any]
            Configuration used
            
        Returns
        -------
        str
            Path to main segmentation file
        """
        print("[STEP 4] Saving customer segmentation results...")
        
        # 1. Main segmentation file
        main_path = self._save_main_segmentation(df)
        
        # 2. Perk distribution summary
        self._save_perk_distribution(df)
        
        # 3. Thresholds used
        self._save_thresholds(thresholds)
        
        # 4. Configuration details
        self._save_config_details(config, thresholds)
        
        # 5. Complete detailed data
        self._save_detailed_data(df)
        
        print(f"\n💾 ALL FILES SAVED TO: {self.output_dir}")
        return main_path
    
    def _save_main_segmentation(self, df: pd.DataFrame) -> str:
        """Save main customer segmentation file"""
        main_path = os.path.join(self.output_dir, "customer_segment.csv")
        
        # Select relevant columns
        output_columns = ['user_id'] if 'user_id' in df.columns else []
        output_columns.extend([
            'assigned_perk', 'assigned_group', 'total_spend', 'num_trips', 
            'num_flights', 'money_spent_hotel_total', 'browsing_rate', 
            'business_rate', 'group_rate', 'avg_bags', 'cancellation_rate'
        ])
        
        # Filter to available columns
        available_columns = [col for col in output_columns if col in df.columns]
        segmentation_df = df[available_columns]
        
        segmentation_df.to_csv(main_path, index=False)
        print(f"   ✅ Saved main segmentation: {main_path}")
        print(f"   - Users: {len(segmentation_df):,}")
        print(f"   - Columns: {len(available_columns)}")
        
        return main_path
    
    def _save_perk_distribution(self, df: pd.DataFrame) -> None:
        """Save perk distribution summary"""
        distribution_path = os.path.join(self.output_dir, "perk_distribution.csv")
        
        perk_dist = df.groupby(['assigned_perk', 'assigned_group']).size()
        perk_dist = perk_dist.reset_index(name='count')
        perk_dist['percentage'] = (perk_dist['count'] / len(df)) * 100
        
        perk_dist.to_csv(distribution_path, index=False)
        print(f"   ✅ Saved perk distribution: {distribution_path}")
    
    def _save_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Save threshold values"""
        thresholds_path = os.path.join(self.output_dir, "segmentation_thresholds.csv")
        
        thresholds_df = pd.DataFrame([
            {'threshold': k, 'value': v} 
            for k, v in thresholds.items()
        ])
        
        thresholds_df.to_csv(thresholds_path, index=False)
        print(f"   ✅ Saved thresholds: {thresholds_path}")
    
    def _save_config_details(
        self, 
        config: Dict[str, Any], 
        thresholds: Dict[str, float]
    ) -> None:
        """Save configuration details with actual computed values"""
        config_path = os.path.join(self.output_dir, "segmentation_config.csv")
        
        config_data = []
        for threshold_name, threshold_config in config['threshold_definitions'].items():
            config_data.append({
                'threshold': threshold_name,
                'column': threshold_config['column'],
                'quantile': threshold_config['quantile'],
                'fallback': threshold_config['fallback'],
                'actual_value': thresholds.get(threshold_name, 'N/A')
            })
        
        pd.DataFrame(config_data).to_csv(config_path, index=False)
        print(f"   ✅ Saved configuration: {config_path}")
    
    def _save_detailed_data(self, df: pd.DataFrame) -> None:
        """Save complete detailed analysis with all metrics"""
        detailed_path = os.path.join(
            self.output_dir, 
            "customer_segmentation_detailed.csv"
        )
        
        df.to_csv(detailed_path, index=False)
        print(f"   ✅ Saved detailed analysis: {detailed_path}")
    
    def save_analysis_results(self, analysis_df: pd.DataFrame) -> None:
        """
        Save perk assignment analysis results.
        
        Parameters
        ----------
        analysis_df : pd.DataFrame
            Analysis results dataframe
        """
        analysis_path = os.path.join(
            self.output_dir, 
            "perk_assignment_analysis.csv"
        )
        analysis_df.to_csv(analysis_path, index=False)
        print(f"   ✅ Saved perk assignment analysis: {analysis_path}")