# core/segment/ml_model/feature_engineer.py

import numpy as np # type: ignore
import pandas as pd # type: ignore
from typing import List, Dict, Any
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler # type: ignore


class FeatureEngineer:
    """
    Handles all feature engineering for clustering algorithms.
    Creates derived features and selects optimal feature sets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature engineer.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
        """
        self.config = config
        self.scaler = None
        self.feature_columns = []
        
    def engineer_features(
        self, 
        df: pd.DataFrame, 
        algorithm: str = 'kmeans'
    ) -> pd.DataFrame:
        """
        Create derived features optimized for clustering.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        algorithm : str
            'kmeans' or 'dbscan' for algorithm-specific features
            
        Returns
        -------
        pd.DataFrame
            DataFrame with engineered features
        """
        print(f"[FEATURE ENGINEERING] Creating features for {algorithm}...")
        df_eng = df.copy()
        
        # Core derived features (common to both algorithms)
        df_eng = self._create_core_features(df_eng)
        
        # Algorithm-specific features
        if algorithm == 'dbscan':
            df_eng = self._create_dbscan_features(df_eng)
        
        print(f"   ✅ Feature engineering complete")  # noqa: F541
        return df_eng
    
    def _create_core_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create core derived features used by all algorithms."""
        
        # 1. Spending efficiency
        if 'total_spend' in df.columns and 'num_trips' in df.columns:
            df['spend_per_trip'] = np.where(
                df['num_trips'] > 0,
                df['total_spend'] / df['num_trips'],
                0
            )
        
        # 2. Hotel preference
        if 'money_spent_hotel_total' in df.columns and 'total_spend' in df.columns:
            df['hotel_preference_score'] = np.where(
                df['total_spend'] > 0,
                df['money_spent_hotel_total'] / df['total_spend'],
                0
            )
        
        # 3. Session efficiency
        if 'num_trips' in df.columns and 'num_sessions' in df.columns:
            df['session_efficiency'] = np.where(
                df['num_sessions'] > 0,
                df['num_trips'] / df['num_sessions'],
                0
            )
        
        # 4. Click efficiency
        if 'num_trips' in df.columns and 'num_clicks' in df.columns:
            df['click_efficiency'] = np.where(
                df['num_clicks'] > 0,
                df['num_trips'] / df['num_clicks'],
                0
            )
        
        # 5. Weekend preference
        if 'num_weekend_trips_agg' in df.columns and 'num_trips' in df.columns:
            df['weekend_ratio'] = np.where(
                df['num_trips'] > 0,
                df['num_weekend_trips_agg'] / df['num_trips'],
                0
            )
        
        # 6. Discount usage
        if 'num_discount_trips_agg' in df.columns and 'num_trips' in df.columns:
            df['discount_ratio'] = np.where(
                df['num_trips'] > 0,
                df['num_discount_trips_agg'] / df['num_trips'],
                0
            )
        
        # 7. Price sensitivity composite
        price_features = []
        if 'bargain_hunter_index' in df.columns:
            price_features.append('bargain_hunter_index')
        if 'avg_dollars_saved_per_km' in df.columns:
            price_features.append('avg_dollars_saved_per_km')
        if 'discount_ratio' in df.columns:
            price_features.append('discount_ratio')
        
        if price_features:
            df['price_sensitivity_index'] = df[price_features].fillna(0).mean(axis=1)
        
        print(f"   ➡️ Created {len([c for c in df.columns if c not in df.columns])} core features")
        return df
    
    def _create_dbscan_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create DBSCAN-specific composite features."""
        
        # Engagement score
        engagement_cols = ['num_clicks', 'num_sessions', 'avg_session_duration']
        available_eng = [c for c in engagement_cols if c in df.columns]
        
        if len(available_eng) >= 2:
            for col in available_eng:
                col_min, col_max = df[col].min(), df[col].max()
                if col_max > col_min:
                    df[f'{col}_norm'] = (df[col] - col_min) / (col_max - col_min)
                else:
                    df[f'{col}_norm'] = 0
            
            norm_cols = [f'{col}_norm' for col in available_eng]
            df['engagement_score'] = df[norm_cols].mean(axis=1)
            df.drop(columns=norm_cols, inplace=True)
        
        # Travel activity score
        travel_cols = ['num_trips', 'num_destinations', 'avg_km_flown']
        available_travel = [c for c in travel_cols if c in df.columns]
        
        if len(available_travel) >= 2:
            for col in available_travel:
                col_min, col_max = df[col].min(), df[col].max()
                if col_max > col_min:
                    df[f'{col}_norm'] = (df[col] - col_min) / (col_max - col_min)
                else:
                    df[f'{col}_norm'] = 0
            
            norm_cols = [f'{col}_norm' for col in available_travel]
            df['travel_activity_score'] = df[norm_cols].mean(axis=1)
            df.drop(columns=norm_cols, inplace=True)
        
        # Spending power score
        spend_cols = ['total_spend', 'avg_money_spent_flight', 'money_spent_hotel_total']
        available_spend = [c for c in spend_cols if c in df.columns]
        
        if len(available_spend) >= 2:
            for col in available_spend:
                col_min, col_max = df[col].min(), df[col].max()
                if col_max > col_min:
                    df[f'{col}_norm'] = (df[col] - col_min) / (col_max - col_min)
                else:
                    df[f'{col}_norm'] = 0
            
            norm_cols = [f'{col}_norm' for col in available_spend]
            df['spending_power_score'] = df[norm_cols].mean(axis=1)
            df.drop(columns=norm_cols, inplace=True)
        
        print(f"   ➡️ Created DBSCAN composite features")  # noqa: F541
        return df
    
    def select_features(
        self, 
        df: pd.DataFrame, 
        algorithm: str = 'kmeans'
    ) -> List[str]:
        """
        Select optimal feature set based on algorithm and config.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with engineered features
        algorithm : str
            'kmeans' or 'dbscan'
            
        Returns
        -------
        List[str]
            List of selected feature names
        """
        print(f"[FEATURE SELECTION] Selecting features for {algorithm}...")
        
        seg_config = self.config.get('segmentation', {})
        
        if algorithm == 'kmeans':
            feature_config = seg_config.get('features', {}).get('optimal_set', {})
            selected_features = self._select_from_config(df, feature_config)
        else:  # dbscan
            feature_config = seg_config.get('dbscan_features', {})
            selected_features = self._select_from_config(df, feature_config)
        
        # Fallback if not enough features
        if len(selected_features) < 5:
            print("   ⚠️ Not enough features from config, using all numeric columns")
            selected_features = self._select_all_numeric(df)
        
        self.feature_columns = selected_features
        print(f"   ✅ Selected {len(selected_features)} features")
        return selected_features
    
    def _select_from_config(
        self, 
        df: pd.DataFrame, 
        feature_config: Dict
    ) -> List[str]:
        """Select features from config that exist in dataframe."""
        selected = []
        
        for group_name, features in feature_config.items():
            if isinstance(features, list):
                for feat in features:
                    if feat in df.columns and feat not in selected:
                        selected.append(feat)
        
        return selected
    
    def _select_all_numeric(self, df: pd.DataFrame) -> List[str]:
        """Fallback: select all numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude ID and target columns
        exclude = [
            'user_id', 'cluster', 'kmeans_cluster', 'dbscan_cluster',
            'is_noise', 'assigned_perk', 'segment_name', 'index',
            'Unnamed: 0'
        ]
        
        return [c for c in numeric_cols if c not in exclude][:20]
    
    def scale_features(
        self, 
        X: pd.DataFrame, 
        method: str = 'robust'
    ) -> np.ndarray:
        """
        Scale features using specified method.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        method : str
            'robust', 'standard', or 'minmax'
            
        Returns
        -------
        np.ndarray
            Scaled feature matrix
        """
        print(f"[SCALING] Using {method} scaler...")
        
        # Handle missing values and infinities
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Select scaler
        if method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        X_scaled = self.scaler.fit_transform(X.values)
        print(f"   ✅ Scaled {X_scaled.shape[1]} features")
        
        return X_scaled
    
    def winsorize_features(
        self, 
        df: pd.DataFrame, 
        features: List[str]
    ) -> pd.DataFrame:
        """
        Apply winsorization to reduce outlier impact.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        features : List[str]
            Features to winsorize
            
        Returns
        -------
        pd.DataFrame
            Winsorized dataframe
        """
        print("[WINSORIZATION] Clipping extreme values...")
        df = df.copy()
        
        # Get winsorization config
        winsorize_config = self.config.get('segmentation', {}).get(
            'features', {}
        ).get('winsorize', {})
        
        winsorized_count = 0
        for col, (lower, upper) in winsorize_config.items():
            if col in features:
                lo = df[col].quantile(lower)
                hi = df[col].quantile(upper)
                df[col] = df[col].clip(lo, hi)
                winsorized_count += 1
        
        print(f"   ✅ Winsorized {winsorized_count} features")
        return df