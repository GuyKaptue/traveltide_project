# core/segment/ml_model/kmeans_cluster.py
"""
Balanced Clustering Pipeline with Configurable Constraints
===========================================================
Ensures each cluster represents 12-23% of users for balanced perk assignment.

Key features:
1. Quantile-based thresholds from config
2. Post-clustering rebalancing to achieve target distributions
3. Hierarchical assignment for extreme cases (VIPs, baseline)
4. Business-friendly cluster naming
"""
 # noqa: F541
 # type: ignore
import os
import yaml  # type: ignore  # noqa: F401
import numpy as np # type: ignore
import pandas as pd # type: ignore
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import RobustScaler # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.cluster import KMeans # type: ignore
#import matplotlib.pyplot as plt # type: ignore
#import seaborn as sns # type: ignore  # noqa: F401
#from IPython.display import display # type: ignore

#import plotly.express as px # type: ignore
#import plotly.graph_objects as go # type: ignore
import plotly.io as pio # type: ignore


from src.utils import project_root, get_path, load_yaml  # noqa: E402

# Import the refactored components
from .data_exporter import DataExporter
from .visualizer import ClusterVisualizer
from .metrics_calculator import MetricsCalculator



# --- STEP 1: CHANGE THE RENDERER FOR VS CODE ---
# "notebook_connected" works best in VS Code. 
# It uses an internet connection to load the JS library (keeping file sizes small).
pio.renderers.default = "notebook_connected"

class KmeansClustering:
    """
    Clustering pipeline with balanced segment sizes (12-23% each).
    """
    
    def __init__(
        self, 
        config_path: Optional[str] = None,
        random_state: int = 42, 
        run_name: str = "kmeans"
    ):
        self.random_state = random_state
        self.run_name = run_name
        self.scaler = None
        self.pca_model = None
        self.kmeans_model = None
        
        # Load configuration
        self.config_path = config_path or os.path.join(
            project_root, 'config', 'ml_config.yaml'
        )
        
        # Initialize self.config by calling the loader
        self.config = self._load_config(self.config_path)
        
        self.thresholds = {}
        
        # Output directories
        base_processed = get_path("processed")
        base_reports = get_path("reports")
        
        self.data_output_dir = os.path.join(
            base_processed, "segment", "ml_model"
        )
        self.fig_output_dir = os.path.join(
            base_reports, "segment", "ml_model", self.run_name
        )
        
        os.makedirs(self.data_output_dir, exist_ok=True)
        os.makedirs(self.fig_output_dir, exist_ok=True)
        
        # 4. Initialize the refactored components
        # DataExporter handles all file saving
        self.data_exporter = DataExporter(self.data_output_dir)
        # Visualizer and MetricsCalculator require the config dictionary
        self.visualizer = ClusterVisualizer(self.fig_output_dir, self.config)
        self.metrics_calculator = MetricsCalculator(self.config)
        
        print(f"✅ BalancedClusteringPipeline initialized")  # noqa: F541
        print(f"   - Run name: {self.run_name}")
        print(f"   - Data output: {self.data_output_dir}")
        print(f"   - Figure output: {self.fig_output_dir}")
    
    # --- Configuration loading and default config methods remain here ---
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        if config_path is None:
            return self._get_default_config()
        
        try:
            config = load_yaml(self.config_path)
            print(f"✅ Configuration loaded successfully")  # noqa: F541
            
            # Extract the segmentation section first
            seg_config = config.get('segmentation', {})
            
            # Validate required sections using the LOCAL variable, not self.config
            required_sections = ['threshold_definitions', 'all_perks', 'group_names']
            for section in required_sections:
                # --- FIX: Check seg_config, not self.config ---
                if section not in seg_config:
                    # Fallback to default if a specific section is missing inside the file
                    print(f"⚠️  Missing section '{section}' in config, using defaults")
                    return self._get_default_config()
                    
            print(f"   - Threshold definitions: {len(seg_config['threshold_definitions'])}")
            print(f"   - Perks: {len(seg_config['all_perks'])}")
            print(f"   - Group names: {len(seg_config['group_names'])}")
            print(f"✅ Loaded config from: {config_path}")
            
            return seg_config
            
        except Exception as e:
            print(f"⚠️  Failed to load config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default configuration matching your requirements."""
        return {
            'threshold_definitions': {
                'TOTAL_SPEND': {
                    'column': 'total_spend',
                    'quantile': 0.80,
                    'fallback': 3000
                },
                'TRIP_COUNT': {
                    'column': 'num_trips',
                    'quantile': 0.80,
                    'fallback': 1.5
                },
                'BROWSING_RATE': {
                    'column': 'browsing_rate',
                    'quantile': 0.80,
                    'fallback': 0.6
                },
                'HOTEL_SPEND': {
                    'column': 'money_spent_hotel_total',
                    'quantile': 0.80,
                    'fallback': 1200
                },
                'BUSINESS_RATE': {
                    'column': 'business_rate',
                    'quantile': 0.80,
                    'fallback': 0.2
                },
                'GROUP_RATE': {
                    'column': 'group_rate',
                    'quantile': 0.80,
                    'fallback': 0.1
                },
                'AVG_BAGS': {
                    'column': 'avg_bags',
                    'quantile': 0.80,
                    'fallback': 1.0
                }
            },
            'all_perks': [
                "1 night free hotel plus flight",
                "free hotel meal",
                "free checked bags",
                "no cancellation fees"
                "exclusive discounts",
            ],
            'group_names': [
                "VIP High-Frequency Spenders",
                "Hotel & Business Focused Travelers",
                "Group & Family Travelers / Heavy Baggage",
                "High-Intent Browsers & Spenders",
                "Baseline Travelers"
            ],
            'target_distribution': {
                'min_pct': 12.0,  # Minimum 12% per cluster
                'max_pct': 23.0,  # Maximum 23% per cluster
                'preferred': [20.0, 22.0, 21.0, 19.0, 18.0]  # Preferred distribution
            }
        }
    
    def calculate_thresholds(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate dynamic thresholds from data using quantiles.
        """
        print("[THRESHOLDS] Calculating dynamic thresholds...")
        thresholds = {}
        
        threshold_defs = self.config.get('threshold_definitions', {})
        
        for key, definition in threshold_defs.items():
            col = definition['column']
            quantile = definition['quantile']
            fallback = definition['fallback']
            
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    threshold = float(values.quantile(quantile))
                    # Use fallback if quantile is too extreme or zero
                    if threshold == 0 or np.isnan(threshold):
                        threshold = fallback
                    thresholds[key] = threshold
                    print(f"   - {key}: {threshold:.2f} (quantile={quantile})")
                else:
                    thresholds[key] = fallback
                    print(f"   - {key}: {fallback:.2f} (fallback)")
            else:
                print(f"   ⚠️  Column '{col}' not found, using fallback: {fallback}")
                thresholds[key] = fallback
        
        self.thresholds = thresholds
        return thresholds
    
    # =========================================================================
    # FEATURE ENGINEERING (Logic remains the same)
    # =========================================================================
    
    def engineer_optimal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create optimal derived features for clustering."""
        df = df.copy()
        print("[FEATURE ENGINEERING] Creating optimal derived metrics...")
        
        # Value metrics
        df['spend_per_trip'] = np.where(
            df['num_trips'] > 0,
            df['total_spend'] / df['num_trips'],
            0
        )
        
        df['hotel_preference_score'] = np.where(
            df['total_spend'] > 0,
            df['money_spent_hotel_total'] / df['total_spend'],
            0
        )
        
        # Travel ratios
        df['weekend_ratio'] = np.where(
            df['num_trips'] > 0,
            df['num_weekend_trips_agg'] / df['num_trips'],
            0
        )
        
        df['discount_ratio'] = np.where(
            df['num_trips'] > 0,
            df['num_discount_trips_agg'] / df['num_trips'],
            0
        )
        
        # Efficiency metrics
        df['session_efficiency'] = np.where(
            df['num_sessions'] > 0,
            df['num_trips'] / df['num_sessions'],
            0
        )
        
        df['click_efficiency'] = np.where(
            df['num_clicks'] > 0,
            df['num_trips'] / df['num_clicks'],
            0
        )
        
        # Price sensitivity composite
        price_cols = []
        if 'bargain_hunter_index' in df.columns:
            price_cols.append('bargain_hunter_index')
        if 'avg_dollars_saved_per_km' in df.columns:
            price_cols.append('avg_dollars_saved_per_km')
        price_cols.append('discount_ratio')
        
        if len(price_cols) > 0:
            df['price_sensitivity_index'] = df[price_cols].fillna(0).mean(axis=1)
        else:
            df['price_sensitivity_index'] = 0
        
        print(f"   ✅ Created 8 new derived features")  # noqa: F541
        return df
    
    def select_optimal_features(self, df: pd.DataFrame) -> List[str]:
        """Select the optimal feature set for clustering."""
        optimal_features = {
            'value_loyalty': [
                'total_spend', 'spend_per_trip', 'booking_growth',
                'RFM_score', 'global_booking_share'
            ],
            'trip_behavior': [
                'num_trips', 'num_destinations', 'international_ratio',
                'avg_km_flown', 'avg_trip_length'
            ],
            'travel_preferences': [
                'business_rate', 'group_rate', 'weekend_ratio',
                'hotel_preference_score', 'avg_bags'
            ],
            'price_sensitivity': [
                'price_sensitivity_index', 'discount_ratio',
                'avg_money_spent_per_seat'
            ],
            'behavioral': [
                'cancellation_rate', 'conversion_rate',
                'browsing_intensity', 'avg_time_after_booking'
            ],
            'engagement': [
                'session_efficiency', 'click_efficiency'
            ]
        }
        
        # Flatten and keep only existing columns
        all_features = []
        for group, features in optimal_features.items():
            for feat in features:
                if feat in df.columns:
                    all_features.append(feat)
        
        print(f"[FEATURE SELECTION] Selected {len(all_features)}/25 optimal features")
        return all_features
    
    def winsorize_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Apply winsorization to reduce impact of extreme outliers."""
        df = df.copy()
        
        winsorize_cols = {
            'total_spend': (0.01, 0.99),
            'spend_per_trip': (0.01, 0.99),
            'avg_km_flown': (0.01, 0.99),
            'num_clicks': (0.01, 0.99)
        }
        
        winsorized_count = 0
        for col, (lower, upper) in winsorize_cols.items():
            if col in features:
                lo = df[col].quantile(lower)
                hi = df[col].quantile(upper)
                df[col] = df[col].clip(lo, hi)
                winsorized_count += 1
        
        print(f"[WINSORIZATION] Applied to {winsorized_count} features")
        return df
    
    # =========================================================================
    # SCALING & DIMENSIONALITY REDUCTION (Logic remains the same)
    # =========================================================================
    
    def scale_and_reduce(
        self, 
        X: pd.DataFrame,
        n_components: float = 0.95, #int=10
        enable_pca: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """Scale features and optionally apply PCA."""
        X = X.fillna(0).replace([np.inf, -np.inf], np.nan).fillna(0)
        feature_names = X.columns.tolist()
        
        print("[SCALING] Using RobustScaler...")
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X.values)
        print(f"   ✅ Scaled {X_scaled.shape[1]} features")
        
        X_pca = None
        if enable_pca:
            print(f"[PCA] Reducing to {n_components} components with whitening...")
            self.pca_model = PCA(
                n_components=n_components,
                whiten=True,
                random_state=self.random_state
            )
            X_pca = self.pca_model.fit_transform(X_scaled)
            
            cumvar = self.pca_model.explained_variance_ratio_.cumsum()
            print(f"   ✅ Explained variance: {cumvar[-1]:.1%}")
        
        return X_scaled, X_pca, feature_names
    
    # =========================================================================
    # BALANCED CLUSTERING WITH CONSTRAINTS (Logic remains the same)
    # =========================================================================
    
    def fit_balanced_kmeans(
        self,
        X: np.ndarray,
        df: pd.DataFrame,
        n_clusters: int = 5
    ) -> np.ndarray:
        """
        Fit K-means and rebalance to achieve target distribution.
        """
        print(f"[BALANCED KMEANS] Fitting with k={n_clusters}...")
        
        # Step 1: Initial K-means clustering
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=20,
            max_iter=500
        )
        initial_labels = self.kmeans_model.fit_predict(X)
        
        # Step 2: Analyze initial distribution
        unique, counts = np.unique(initial_labels, return_counts=True)
        total = len(initial_labels)
        
        print(f"   📊 Initial distribution:")  # noqa: F541
        for cluster_id, count in zip(unique, counts):
            pct = (count / total) * 100
            print(f"     - Cluster {cluster_id}: {count} ({pct:.1f}%)")
        
        # Step 3: Check if rebalancing is needed
        target_dist = self.config.get('target_distribution', {})
        min_pct = target_dist.get('min_pct', 12.0)
        max_pct = target_dist.get('max_pct', 23.0)
        
        needs_rebalancing = any(
            (count / total) * 100 < min_pct or (count / total) * 100 > max_pct
            for count in counts
        )
        
        if needs_rebalancing:
            print(f"   ⚠️  Rebalancing needed (target: {min_pct}-{max_pct}%)")
            balanced_labels = self._rebalance_clusters(
                X, df, initial_labels, n_clusters
            )
        else:
            print(f"   ✅ Distribution already balanced")  # noqa: F541
            balanced_labels = initial_labels
        
        # Step 4: Report final metrics
        
        unique, counts = np.unique(balanced_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            pct = (count / total) * 100
            print(f"     - Cluster {cluster_id}: {count} ({pct:.1f}%)")
        
        return balanced_labels
    
    def _rebalance_clusters(
        self,
        X: np.ndarray,
        df: pd.DataFrame,
        initial_labels: np.ndarray,
        n_clusters: int
    ) -> np.ndarray:
        """
        Rebalance clusters using hierarchical rule-based assignment.
        """
        print("   [REBALANCING] Applying hierarchical assignment...")
        
        # Create working copy
        df_work = df.copy()
        df_work['initial_cluster'] = initial_labels
        df_work['final_cluster'] = -1  # Unassigned
        
        # Get thresholds
        th = self.thresholds
        
        # Cluster 0: VIP High-Frequency Spenders (target: 12-15%)
        vip_mask = (
            (df_work['total_spend'] >= th.get('TOTAL_SPEND', 3000)) &
            (df_work['num_trips'] >= th.get('TRIP_COUNT', 1.5))
        )
        df_work.loc[vip_mask, 'final_cluster'] = 0
        
        # Cluster 1: High-Intent Browsers & Spenders (target: 20-23%)
        browser_mask = (
            (df_work['final_cluster'] == -1) &
            (
                (df_work['browsing_rate'] >= th.get('BROWSING_RATE', 0.6)) |
                (df_work['total_spend'] >= th.get('TOTAL_SPEND', 4000) * 0.7)
            )
        )
        df_work.loc[browser_mask, 'final_cluster'] = 1
        
        # Cluster 2: Group & Family / Heavy Baggage (target: 18-21%)
        group_mask = (
            (df_work['final_cluster'] == -1) &
            (
                (df_work['group_rate'] >= th.get('GROUP_RATE', 0.1)) |
                (df_work['avg_bags'] >= th.get('AVG_BAGS', 1.0))
            )
        )
        df_work.loc[group_mask, 'final_cluster'] = 2
        
        # Cluster 3: Hotel & Business Focused (target: 17-20%)
        business_mask = (
            (df_work['final_cluster'] == -1) &
            (
                (df_work['money_spent_hotel_total'] >= th.get('HOTEL_SPEND', 1200)) |
                (df_work['business_rate'] >= th.get('BUSINESS_RATE', 0.2))
            )
        )
        df_work.loc[business_mask, 'final_cluster'] = 3
        
        # Cluster 4: Baseline Travelers (remaining, target: ~23%)
        baseline_mask = (df_work['final_cluster'] == -1)
        df_work.loc[baseline_mask, 'final_cluster'] = 4
        
        # Check distribution and adjust if needed
        final_labels = df_work['final_cluster'].values
        unique, counts = np.unique(final_labels, return_counts=True)
        total = len(final_labels)
        
        target_dist = self.config.get('target_distribution', {})
        min_pct = target_dist.get('min_pct', 12.0)
        max_pct = target_dist.get('max_pct', 23.0)
        
        # If any cluster is still too small or large, use distance-based reassignment
        for cluster_id, count in zip(unique, counts):
            pct = (count / total) * 100
            if pct < min_pct or pct > max_pct:
                print(f"     ⚠️  Cluster {cluster_id}: {pct:.1f}% (adjusting...)")
                final_labels = self._adjust_cluster_size(
                    X, final_labels, cluster_id, count, total, min_pct, max_pct
                )
        
        return final_labels
    
    
    # =========================================================================
    # NAMING & PERKS (Logic remains the same)
    # =========================================================================
    
    def assign_names_and_perks(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 
        Assign human-readable names and perks to clusters using perk_groups. 
        (Logic remains identical to previous successful update)
        """
        df = df.copy()
        
        group_names = self.config.get('group_names', [f"Cluster {i}" for i in range(5)])
        perk_groups = self.config.get('perk_groups', [])
        
        cluster_to_name = {i: name for i, name in enumerate(group_names)}
        name_to_perk = {item['group']: item['perk'] for item in perk_groups}
        
        cluster_to_perk = {
            i: name_to_perk.get(name, "No perk assigned")
            for i, name in cluster_to_name.items()
        }
        
        df['segment_name'] = df['cluster'].map(cluster_to_name)
        df['assigned_perk'] = df['cluster'].map(cluster_to_perk)
        
        print("[NAMING] Assigned segment names and perks:")
        
        for i in range(len(group_names)):
            count = (df['cluster'] == i).sum()
            pct = count / len(df) * 100
            name = cluster_to_name.get(i, f"Cluster {i}")
            perk = cluster_to_perk.get(i, "No perk assigned")
            print(f"   - {i}. {name} ({pct:.1f}%) -> Perk: {perk}")
        
        return df
    
    def _adjust_cluster_size(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        cluster_id: int,
        current_count: int,
        total: int,
        min_pct: float,
        max_pct: float
    ) -> np.ndarray:
        """
        Adjust cluster size by reassigning points based on distance to centroids.
        """
        current_pct = (current_count / total) * 100
        
        if current_pct < min_pct:
            # Cluster too small: steal points from largest clusters
            target_count = int(total * min_pct / 100)
            needed = target_count - current_count
            
            # Find points in other clusters closest to this cluster's centroid
            cluster_mask = (labels == cluster_id)
            centroid = X[cluster_mask].mean(axis=0)
            
            # Calculate distances for all points NOT in this cluster
            other_mask = ~cluster_mask
            distances = np.linalg.norm(X[other_mask] - centroid, axis=1)
            other_indices = np.where(other_mask)[0]
            
            # Reassign closest points
            closest_indices = other_indices[np.argsort(distances)[:needed]]
            labels[closest_indices] = cluster_id
            
        elif current_pct > max_pct:
            # Cluster too large: give away furthest points
            target_count = int(total * max_pct / 100)
            excess = current_count - target_count
            
            # Find points furthest from this cluster's centroid
            cluster_mask = (labels == cluster_id)
            centroid = X[cluster_mask].mean(axis=0)
            cluster_indices = np.where(cluster_mask)[0]
            
            distances = np.linalg.norm(X[cluster_mask] - centroid, axis=1)
            furthest_indices = cluster_indices[np.argsort(distances)[-excess:]]
            
            # Reassign to nearest OTHER cluster
            for idx in furthest_indices:
                point = X[idx]
                other_clusters = [c for c in np.unique(labels) if c != cluster_id]
                
                min_dist = np.inf
                best_cluster = other_clusters[0]
                
                for other_c in other_clusters:
                    other_mask = (labels == other_c)
                    other_centroid = X[other_mask].mean(axis=0)
                    dist = np.linalg.norm(point - other_centroid)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = other_c
                
                labels[idx] = best_cluster
        
        return labels
 
    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================
    
    def run_pipeline(
        self, 
        df: pd.DataFrame, 
        n_clusters: int = 5,
        enable_pca: bool = True,
        n_components: float = 0.95
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Execute the full clustering and assignment pipeline.
        """
        print("\n" + "="*80)
        print(f"🚀 STARTING K-MEANS CLUSTERING PIPELINE: {self.run_name}")
        print("="*80)

        # Step 1-5: Data Prep (Unchanged)
        print("\nStep 1/9: Calculating Thresholds...")
        self.calculate_thresholds(df)

        print("\nStep 2/9: Feature Engineering...")
        df_engineered = self.engineer_optimal_features(df)

        print("\nStep 3/9: Feature Selection...")
        optimal_features = self.select_optimal_features(df_engineered)

        print("\nStep 4/9: Winsorization...")
        df_winsorized = self.winsorize_features(df_engineered, optimal_features)

        print("\nStep 5/9: Scaling and Dimensionality Reduction...")
        X_scaled, X_pca, feature_names = self.scale_and_reduce(
            df_winsorized[optimal_features],
            n_components=n_components,
            enable_pca=enable_pca
        )
        X_cluster = X_pca if X_pca is not None else X_scaled

        # Step 6: Balanced Clustering & Hierarchical Assignment (RESTORED LOGIC)
        # This single step now handles K-Means fitting and the hierarchical re-assignment.
        print("\nStep 6/9: Balanced K-means Clustering & Hierarchical Assignment...")
        labels = self.fit_balanced_kmeans(X_cluster, df_engineered, n_clusters=n_clusters)

        # Create the resulting DataFrame
        df_result = df_engineered.copy()
        df_result['cluster'] = labels

        # Step 7: Assign Names & Perks
        print("\nStep 7/9: Assigning Names & Perks...")
        df_result = self.assign_names_and_perks(df_result)
        
        # Step 8: Calculate Metrics (Delegated to MetricsCalculator)
        # Uses the feature matrix (X_cluster) and the final business-aligned labels
        print("\nStep 8/9: Calculating Metrics...")
        metrics = self.metrics_calculator.compute_kmeans_metrics(
            X_cluster, df_result['cluster'].values, df_result
        )

        # Step 9: Visualization & Export (Delegated to Visualizer/Exporter)
        print("\nStep 9/9: Generating Visualizations and Exporting Results...")
        self.visualizer.plot_kmeans_results(
            X_cluster, df_result['cluster'].values, df_result, metrics
        )

        exported_files = self.data_exporter.export_results(
            df_result, "kmeans", metrics, include_features=True
        )

        print("\n✅ Pipeline completed successfully.")
        return df_result, metrics, exported_files