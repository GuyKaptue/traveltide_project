# ============================================================================
# COMPLETE ML CLUSTERING SYSTEM - INTEGRATION GUIDE
# ============================================================================
#
# This document shows how all components work together for K-Means and DBSCAN
# clustering with your dataset.
#
# ============================================================================

"""
MAIN ORCHESTRATOR: clustering_orchestrator.py
==============================================
"""

import os
import pandas as pd # type: ignore
import numpy as np # type: ignore  # noqa: F401
from typing import Dict, Any, Optional

from src.utils import project_root, get_path, load_yaml
from .feature_engineer import FeatureEngineer
from .kmeans_engine import KMeansEngine
from .dbscan_engine import DBSCANEngine
from .perk_assigner import PerkAssigner # type: ignore
from .metrics_calculator import MetricsCalculator
from .visualizer import ClusterVisualizer
from .data_exporter import DataExporter


class ClusteringOrchestrator:
    """
    Main orchestrator for ML-based customer segmentation.
    Supports K-Means and DBSCAN with automatic perk assignment.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        run_name: str = "dbscan"
    ):
        """
        Initialize clustering orchestrator.
        
        Parameters
        ----------
        config_path : str, optional
            Path to config YAML file
        run_name : str
            Name for this clustering run
        """
        self.run_name = run_name
        
        # Load configuration
        self.config_path = config_path or os.path.join(
            project_root, 'config', 'ml_config.yaml'
        )
        self.config = load_yaml(self.config_path)
        
        # Setup output directories
        self._setup_directories()
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(self.config)
        self.kmeans_engine = KMeansEngine(self.config)
        self.dbscan_engine = DBSCANEngine(self.config)
        self.perk_assigner = PerkAssigner(self.config)
        self.metrics_calculator = MetricsCalculator(self.config)
        self.visualizer = ClusterVisualizer(
            self.fig_output_dir, 
            self.config
        )
        self.exporter = DataExporter(self.data_output_dir)
        
        print(f"✅ ClusteringOrchestrator initialized")  # noqa: F541
        print(f"   Run: {self.run_name}")
        print(f"   Config: {self.config_path}")
    
    def _setup_directories(self):
        """Setup output directory structure."""
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
    
    # ========================================================================
    # K-MEANS PIPELINE
    # ========================================================================
    
    def run_kmeans(
        self,
        df: pd.DataFrame,
        n_clusters: Optional[int] = 5
    ) -> Dict[str, Any]:
        """
        Run complete K-Means clustering pipeline.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input customer data
        n_clusters : int, optional
            Number of clusters (uses config if None)
            
        Returns
        -------
        Dict[str, Any]
            Complete results dictionary
        """
        print("\n" + "="*80)
        print("🎯 K-MEANS CLUSTERING PIPELINE")
        print("="*80 + "\n")
        
        # Step 1: Feature Engineering
        print("[STEP 1/7] Feature Engineering...")
        df_eng = self.feature_engineer.engineer_features(df, algorithm='kmeans')
        
        # Step 2: Feature Selection
        print("\n[STEP 2/7] Feature Selection...")
        features = self.feature_engineer.select_features(df_eng, algorithm='kmeans')
        X = df_eng[features]
        
        # Step 3: Winsorization
        print("\n[STEP 3/7] Winsorization...")
        df_wins = self.feature_engineer.winsorize_features(df_eng, features)
        X = df_wins[features]
        
        # Step 4: Scaling
        print("\n[STEP 4/7] Scaling...")
        scaler_method = self.config.get('segmentation', {}).get(
            'clustering', {}
        ).get('scaler_method', 'robust')
        X_scaled = self.feature_engineer.scale_features(X, method=scaler_method)
        
        # Step 5: Clustering
        print("\n[STEP 5/7] K-Means Clustering...")
        clustering_results = self.kmeans_engine.fit_and_assign(
            X_scaled, df_wins, n_clusters
        )
        
        # Step 6: Perk Assignment
        print("\n[STEP 6/7] Perk Assignment...")
        df_result = df_eng.copy()
        df_result['cluster'] = clustering_results['labels']
        df_result = self.perk_assigner.assign_perks_and_names(
            df_result, 
            algorithm='kmeans'
        )
        
        # Step 7: Metrics & Visualization
        print("\n[STEP 7/7] Evaluation & Visualization...")
        metrics = self.metrics_calculator.compute_kmeans_metrics(
            X_scaled, 
            clustering_results['labels'], 
            df_result
        )
        
        self.visualizer.plot_kmeans_results(
            clustering_results.get('X_pca', X_scaled),
            clustering_results['labels'],
            df_result,
            metrics
        )
        
        # Export results
        self.exporter.export_results(df_result, 'kmeans', metrics)
        
        # Compile final results
        final_results = {
            'df_result': df_result,
            'labels': clustering_results['labels'],
            'metrics': metrics,
            'model': clustering_results['model'],
            'features_used': features,
            'X_scaled': X_scaled,
            'X_pca': clustering_results.get('X_pca'),
            'algorithm': 'kmeans'
        }
        
        print("\n✅ K-Means pipeline complete!")
        return final_results
    
    # ========================================================================
    # DBSCAN PIPELINE
    # ========================================================================
    
    def run_dbscan(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run complete DBSCAN clustering pipeline.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input customer data
            
        Returns
        -------
        Dict[str, Any]
            Complete results dictionary
        """
        print("\n" + "="*80)
        print("🔍 DBSCAN CLUSTERING PIPELINE")
        print("="*80 + "\n")
        
        # Step 1: Feature Engineering
        print("[STEP 1/6] Feature Engineering...")
        df_eng = self.feature_engineer.engineer_features(df, algorithm='dbscan')
        
        # Step 2: Feature Selection
        print("\n[STEP 2/6] Feature Selection...")
        features = self.feature_engineer.select_features(df_eng, algorithm='dbscan')
        X = df_eng[features]
        
        # Step 3: Scaling
        print("\n[STEP 3/6] Scaling...")
        X_scaled = self.feature_engineer.scale_features(X, method='robust')
        
        # Step 4: DBSCAN Clustering
        print("\n[STEP 4/6] DBSCAN Clustering...")
        clustering_results = self.dbscan_engine.fit_and_assign(X_scaled, df_eng)
        
        # Step 5: Perk Assignment
        print("\n[STEP 5/6] Perk Assignment...")
        df_result = df_eng.copy()
        df_result['cluster'] = clustering_results['labels']
        df_result['is_noise'] = (clustering_results['labels'] == -1)
        
        df_result = self.perk_assigner.assign_perks_and_names(
            df_result,
            algorithm='dbscan'
        )
        
        # Step 6: Metrics & Visualization
        print("\n[STEP 6/6] Evaluation & Visualization...")
        metrics = self.metrics_calculator.compute_dbscan_metrics(
            X_scaled,
            clustering_results['labels'],
            df_result
        )
        
        self.visualizer.plot_dbscan_results(
            X_scaled,
            clustering_results['labels'],
            df_result,
            metrics
        )
        
        # Export results
        self.exporter.export_results(df_result, 'dbscan', metrics)
        
        # Compile final results
        final_results = {
            'df_result': df_result,
            'labels': clustering_results['labels'],
            'metrics': metrics,
            'model': clustering_results['model'],
            'parameters': clustering_results['parameters'],
            'features_used': features,
            'X_scaled': X_scaled,
            'algorithm': 'dbscan'
        }
        
        print("\n✅ DBSCAN pipeline complete!")
        return final_results
    
    # ========================================================================
    # ALGORITHM COMPARISON
    # ========================================================================
    
    def run_comparison(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run both K-Means and DBSCAN and compare results.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input customer data
            
        Returns
        -------
        Dict[str, Any]
            Results for both algorithms with comparison
        """
        print("\n" + "="*80)
        print("🔬 ALGORITHM COMPARISON: K-MEANS vs DBSCAN")
        print("="*80 + "\n")
        
        results = {}
        
        # Run K-Means
        print("\n[RUNNING K-MEANS]")
        print("-" * 80)
        try:
            results['kmeans'] = self.run_kmeans(df)
        except Exception as e:
            print(f"❌ K-Means failed: {e}")
            results['kmeans'] = None
        
        # Run DBSCAN
        print("\n[RUNNING DBSCAN]")
        print("-" * 80)
        try:
            results['dbscan'] = self.run_dbscan(df)
        except Exception as e:
            print(f"❌ DBSCAN failed: {e}")
            results['dbscan'] = None
        
        # Compare if both succeeded
        if results['kmeans'] and results['dbscan']:
            print("\n[COMPARISON]")
            print("=" * 80)
            self._print_comparison(results)
            self.visualizer.plot_comparison(results)
        
        return results
    
    def _print_comparison(self, results: Dict[str, Any]):
        """Print comparison table between algorithms."""
        kmeans_m = results['kmeans']['metrics']
        dbscan_m = results['dbscan']['metrics']
        
        print(f"\n{'Metric':<30} {'K-Means':<20} {'DBSCAN':<20}")
        print("-" * 70)
        print(f"{'Clusters':<30} {kmeans_m['n_clusters']:<20} {dbscan_m['n_clusters']:<20}")
        print(f"{'Noise Points':<30} {0:<20} {dbscan_m.get('n_noise', 0):<20}")
        print(f"{'Silhouette Score':<30} {kmeans_m.get('silhouette', -1):<20.3f} {dbscan_m.get('silhouette', -1):<20.3f}")
        print(f"{'Davies-Bouldin':<30} {kmeans_m.get('davies_bouldin', 999):<20.3f} {dbscan_m.get('davies_bouldin', 999):<20.3f}")
        print(f"{'Calinski-Harabasz':<30} {kmeans_m.get('calinski', 0):<20.1f} {dbscan_m.get('calinski', 0):<20.1f}")
        
        print("\n" + "=" * 80)
        
        # Recommendation
        km_sil = kmeans_m.get('silhouette', -1)
        db_sil = dbscan_m.get('silhouette', -1)
        
        if km_sil > db_sil:
            print("✅ RECOMMENDATION: K-Means performs better")
        elif db_sil > km_sil and dbscan_m.get('noise_ratio', 1) < 0.3:
            print("✅ RECOMMENDATION: DBSCAN performs better")
        else:
            print("⚖️ RECOMMENDATION: Both algorithms perform similarly")
        
        print("=" * 80)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_usage():
    """
    Example usage patterns for the clustering system.
    """
 
    
    # Load your data
    df = pd.read_csv('path/to/your/data.csv')
    
    # ========================================================================
    # PATTERN 1: Run both algorithms and compare
    # ========================================================================
    
    orchestrator = ClusteringOrchestrator(
        config_path='config/ml_config.yaml',
        run_name='customer_seg_v1'
    )
    
    results = orchestrator.run_comparison(df)
    
    # Access K-Means results
    kmeans_df = results['kmeans']['df_result']  # noqa: F841
    kmeans_metrics = results['kmeans']['metrics']  # noqa: F841
    
    # Access DBSCAN results
    dbscan_df = results['dbscan']['df_result']  # noqa: F841
    dbscan_metrics = results['dbscan']['metrics']  # noqa: F841
    
    # ========================================================================
    # PATTERN 2: Run only K-Means
    # ========================================================================
    
    kmeans_results = orchestrator.run_kmeans(df, n_clusters=5)
    
    # Get segmented customers
    segmented_df = kmeans_results['df_result']
    
    # Get VIP customers
    vip_customers = segmented_df[
        segmented_df['segment_name'] == 'VIP High-Frequency Spenders'
    ]
    
    print(f"VIP customers: {len(vip_customers)}")
    
    # ========================================================================
    # PATTERN 3: Run only DBSCAN
    # ========================================================================
    
    dbscan_results = orchestrator.run_dbscan(df)
    
    # Get customers without noise
    good_clusters = dbscan_results['df_result'][  # noqa: F841
        dbscan_results['df_result']['is_noise'] == False  # noqa: E712
    ]
    
    # ========================================================================
    # PATTERN 4: Export for campaigns
    # ========================================================================
    
    # Export each segment
    for perk in segmented_df['assigned_perk'].unique():
        segment_customers = segmented_df[
            segmented_df['assigned_perk'] == perk
        ]
        
        campaign_file = f"campaign_{perk.replace(' ', '_')}.csv"
        segment_customers[['user_id', 'assigned_perk']].to_csv(
            campaign_file, 
            index=False
        )
        print(f"Exported {len(segment_customers)} customers to {campaign_file}")


if __name__ == "__main__":
    example_usage()


# ============================================================================
# COMPONENT REFERENCE
# ============================================================================

"""
All Components in the System:
------------------------------

1. ClusteringOrchestrator (Main coordinator)
   - Coordinates entire pipeline
   - Manages data flow between components
   - Handles configuration and directories

2. FeatureEngineer
   - Creates derived features
   - Selects optimal feature sets
   - Applies scaling and winsorization

3. KMeansEngine
   - K-Means clustering
   - PCA dimensionality reduction
   - Cluster center calculation

4. DBSCANEngine
   - DBSCAN clustering
   - Automatic parameter optimization
   - Knee detection for epsilon

5. ClusterBalancer (if needed)
   - Rebalances K-Means clusters
   - Ensures target distribution (12-23%)
   - Distance-based reassignment

6. PerkAssigner
   - Maps clusters to business segments
   - Assigns perks based on behavior
   - Handles both K-Means and DBSCAN

7. MetricsCalculator
   - Computes clustering quality metrics
   - Silhouette score, Davies-Bouldin, etc.
   - Business alignment metrics

8. ClusterVisualizer
   - 2D/3D PCA plots
   - Distribution charts
   - Comparison visualizations
   - Interactive HTML plots

9. DataExporter
   - Saves segmented customer data
   - Exports metrics
   - Creates summary reports

File Structure Checklist:
-------------------------
✅ clustering_orchestrator.py (main)
✅ feature_engineer.py
✅ kmeans_engine.py
✅ dbscan_engine.py
✅ cluster_balancer.py
✅ perk_assigner.py
✅ metrics_calculator.py
✅ visualizer.py
✅ data_exporter.py
✅ __init__.py
"""