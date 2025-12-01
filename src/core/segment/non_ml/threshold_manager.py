# core/segment/non_ml/threshold_manager.py

import pandas as pd # type: ignore
from typing import Dict, Any


class ThresholdManager:
    """
    Manages computation and storage of quantile-based thresholds for segmentation.
    """
    
    def __init__(self, df: pd.DataFrame, config: Dict[str, Any]):
        """
        Initialize threshold manager.
        
        Parameters
        ----------
        df : pd.DataFrame
            User data with computed metrics
        config : Dict[str, Any]
            Configuration dictionary with threshold definitions
        """
        self.df = df
        self.config = config
        self.thresholds: Dict[str, float] = {}
    
    def compute_thresholds(self) -> Dict[str, float]:
        """
        Compute quantile-based thresholds using configuration.
        
        Returns
        -------
        Dict[str, float]
            Dictionary mapping threshold names to values
        """
        print("[STEP 2] Calculating quantile thresholds from configuration...")
        
        threshold_definitions = self.config['threshold_definitions']
        
        for threshold_name, threshold_config in threshold_definitions.items():
            self._compute_single_threshold(threshold_name, threshold_config)
        
        print("✅ Thresholds computed from configuration")
        return self.thresholds
    
    def _compute_single_threshold(
        self, 
        threshold_name: str, 
        threshold_config: Dict[str, Any]
    ) -> None:
        """
        Compute a single threshold value.
        
        Parameters
        ----------
        threshold_name : str
            Name of the threshold (e.g., 'TOTAL_SPEND')
        threshold_config : Dict[str, Any]
            Configuration for this threshold
        """
        column = threshold_config['column']
        quantile = threshold_config['quantile']
        fallback = threshold_config['fallback']
        description = threshold_config.get('description', '')
        
        if column in self.df.columns:
            q_value = self.df[column].quantile(quantile)
            
            # Special handling for sparse rates
            if q_value == 0 and threshold_name in ["GROUP_RATE", "BUSINESS_RATE"]:
                self.thresholds[threshold_name] = fallback
                print(f"   ➡️ {threshold_name}: {fallback:.3f} (fallback - sparse data)")
            else:
                self.thresholds[threshold_name] = q_value
                print(f"   ➡️ {threshold_name}: {q_value:.3f} (quantile {quantile})")
            
            # Print description if available
            if description:
                self._print_description(description)
        else:
            self.thresholds[threshold_name] = fallback
            print(f"   ⚠️ {threshold_name}: {fallback:.3f} (fallback - column missing)")
    
    def _print_description(self, description: str, max_length: int = 80) -> None:
        """Print threshold description, truncated if too long"""
        clean_desc = ' '.join(description.split())  # Remove extra whitespace
        if len(clean_desc) > max_length:
            clean_desc = clean_desc[:max_length] + '...'
        print(f"      📝 {clean_desc}")
    
    def get_thresholds(self) -> Dict[str, float]:
        """Return computed thresholds"""
        return self.thresholds
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert thresholds to DataFrame for easy export"""
        return pd.DataFrame([
            {'threshold': k, 'value': v} 
            for k, v in self.thresholds.items()
        ])