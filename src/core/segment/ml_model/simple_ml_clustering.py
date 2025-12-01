# core/segment/ml_model/simple_ml_clustering.py
"""
Simplified ML Clustering System - Ready to Use
==============================================

A streamlined implementation for K-Means and DBSCAN clustering with automatic
perk assignment for customer segmentation.

Usage:
------
    from core.segment.ml_model.simple_ml_clustering import MLClustering
    
    # Run both algorithms
    ml = MLClustering(config_path='config/ml_config.yaml')
    results = ml.run_both(df)
    
    # Or run individually
    kmeans_results = ml.run_kmeans(df, n_clusters=5)
    dbscan_results = ml.run_dbscan(df)
"""

import os
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans, DBSCAN # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.preprocessing import RobustScaler # type: ignore
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score # type: ignore
from sklearn.neighbors import NearestNeighbors # type: ignore

from src.utils import load_yaml, project_root, get_path


class MLClustering:
    """
    Complete ML clustering system with K-Means and DBSCAN.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with configuration."""
        self.config_path = config_path or os.path.join(
            project_root, 'config', 'ml_config.yaml'
        )
        self.config = load_yaml(self.config_path)
        
        # Setup directories
        self.output_dir = os.path.join(get_path('processed'), 'segment', 'ml_model')
        self.fig_dir = os.path.join(get_path('reports'), 'segment', 'ml_model')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)
        
        print(f"✅ MLClustering initialized")  # noqa: F541
        print(f"   Config: {self.config_path}")
    
    # ========================================================================
    # FEATURE ENGINEERING
    # ========================================================================
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for clustering."""
        print("[FEATURES] Engineering features...")
        df = df.copy()
        
        # Spending metrics
        if 'total_spend' in df.columns and 'num_trips' in df.columns:
            df['spend_per_trip'] = df['total_spend'] / df['num_trips'].replace(0, 1)
        
        if 'money_spent_hotel_total' in df.columns and 'total_spend' in df.columns:
            df['hotel_preference'] = df['money_spent_hotel_total'] / df['total_spend'].replace(0, 1)
        
        # Efficiency metrics
        if 'num_trips' in df.columns and 'num_sessions' in df.columns:
            df['session_efficiency'] = df['num_trips'] / df['num_sessions'].replace(0, 1)
        
        # Ratios
        if 'num_weekend_trips_agg' in df.columns and 'num_trips' in df.columns:
            df['weekend_ratio'] = df['num_weekend_trips_agg'] / df['num_trips'].replace(0, 1)
        
        if 'num_discount_trips_agg' in df.columns and 'num_trips' in df.columns:
            df['discount_ratio'] = df['num_discount_trips_agg'] / df['num_trips'].replace(0, 1)
        
        print(f"   ✅ Created derived features")  # noqa: F541
        return df
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """Select best features for clustering."""
        # Priority features
        priority_features = [
            'total_spend', 'num_trips', 'num_sessions', 'conversion_rate',
            'browsing_rate', 'business_rate', 'group_rate', 'RFM_score',
            'money_spent_hotel_total', 'avg_bags', 'booking_growth',
            'spend_per_trip', 'hotel_preference', 'session_efficiency',
            'international_ratio', 'avg_km_flown', 'cancellation_rate'
        ]
        
        # Use features that exist
        features = [f for f in priority_features if f in df.columns]
        
        # Add more if needed
        if len(features) < 10:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            exclude = ['user_id', 'cluster', 'index']
            additional = [c for c in numeric_cols if c not in features and c not in exclude]
            features.extend(additional[:15-len(features)])
        
        print(f"[FEATURES] Selected {len(features)} features")
        return features
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        features: List[str],
        apply_pca: bool = True,
        n_components: int = 10
    ) -> Tuple[np.ndarray, Optional[np.ndarray], RobustScaler]:
        """Scale features and optionally apply PCA."""
        print("[SCALING] Preparing data...")
        
        X = df[features].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Scale
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA (optional)
        X_pca = None
        if apply_pca:
            n_components = min(n_components, X_scaled.shape[1], X_scaled.shape[0])
            pca = PCA(n_components=n_components, whiten=True, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            print(f"   ✅ PCA: {pca.explained_variance_ratio_.sum():.1%} variance")
        
        print(f"   ✅ Data prepared")  # noqa: F541
        return X_scaled, X_pca, scaler
    
    # ========================================================================
    # K-MEANS
    # ========================================================================
    
    def run_kmeans(
        self, 
        df: pd.DataFrame, 
        n_clusters: int = 5,
        use_pca: bool = True
    ) -> Dict:
        """Run K-Means clustering pipeline."""
        print("\n" + "="*70)
        print("K-MEANS CLUSTERING")
        print("="*70)
        
        # Feature engineering
        df_eng = self.engineer_features(df)
        features = self.select_features(df_eng)
        
        # Prepare data
        X_scaled, X_pca, scaler = self.prepare_data(
            df_eng, features, apply_pca=use_pca
        )
        X_cluster = X_pca if use_pca else X_scaled
        
        # Cluster
        print(f"\n[K-MEANS] Fitting {n_clusters} clusters...")
        kmeans = KMeans(
            n_clusters=n_clusters, 
            n_init=20, 
            max_iter=500, 
            random_state=42
        )
        labels = kmeans.fit_predict(X_cluster)
        
        # Assign perks
        df_result = df_eng.copy()
        df_result['cluster'] = labels
        df_result = self._assign_kmeans_perks(df_result)
        
        # Metrics
        metrics = self._compute_metrics(X_cluster, labels, 'kmeans')
        
        # Visualize
        self._plot_kmeans(X_cluster, labels, df_result, n_clusters)
        
        # Save
        output_path = os.path.join(self.output_dir, 'kmeans_segmentation.csv')
        df_result.to_csv(output_path, index=False)
        print(f"\n💾 Saved: {output_path}")
        
        print("\n✅ K-Means complete!")
        return {
            'df': df_result,
            'labels': labels,
            'model': kmeans,
            'metrics': metrics,
            'features': features
        }
    
    def _assign_kmeans_perks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign perks to K-Means clusters based on spending."""
        seg_config = self.config.get('segmentation', {})
        perks = seg_config.get('all_perks', [
            "1 night free hotel plus flight",
            "exclusive discounts",
            "free checked bags",
            "free hotel meal",
            "no cancellation fees"
        ])
        names = seg_config.get('group_names', [
            f"Segment {i}" for i in range(len(perks))
        ])
        
        # Sort clusters by avg spend
        cluster_spending = df.groupby('cluster')['total_spend'].mean().sort_values(ascending=False)
        
        # Map highest spend → best perk
        cluster_to_perk = {}
        cluster_to_name = {}
        for idx, cluster_id in enumerate(cluster_spending.index):
            perk_idx = min(idx, len(perks)-1)
            cluster_to_perk[cluster_id] = perks[perk_idx]
            cluster_to_name[cluster_id] = names[perk_idx]
        
        df['assigned_perk'] = df['cluster'].map(cluster_to_perk)
        df['segment_name'] = df['cluster'].map(cluster_to_name)
        
        # Print assignments
        print("\n[PERKS] K-Means assignments:")
        for c in sorted(df['cluster'].unique()):
            count = (df['cluster'] == c).sum()
            avg_spend = df[df['cluster'] == c]['total_spend'].mean()
            print(f"   Cluster {c}: {cluster_to_name[c]}")
            print(f"      {count:,} customers | Avg spend ${avg_spend:,.0f}")
            print(f"      Perk: {cluster_to_perk[c]}")
        
        return df
    
    # ========================================================================
    # DBSCAN
    # ========================================================================
    
    def run_dbscan(self, df: pd.DataFrame) -> Dict:
        """Run DBSCAN clustering pipeline."""
        print("\n" + "="*70)
        print("DBSCAN CLUSTERING")
        print("="*70)
        
        # Feature engineering
        df_eng = self.engineer_features(df)
        features = self.select_features(df_eng)
        
        # Prepare data (no PCA for DBSCAN)
        X_scaled, _, scaler = self.prepare_data(
            df_eng, features, apply_pca=False
        )
        
        # Find optimal parameters
        print("\n[DBSCAN] Optimizing parameters...")
        eps, min_samples = self._optimize_dbscan(X_scaled)
        
        # Cluster
        print(f"\n[DBSCAN] Clustering (eps={eps:.3f}, min_samples={min_samples})...")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"   Found {n_clusters} clusters, {n_noise:,} noise points")
        
        # Assign perks
        df_result = df_eng.copy()
        df_result['cluster'] = labels
        df_result['is_noise'] = (labels == -1)
        df_result = self._assign_dbscan_perks(df_result)
        
        # Metrics
        metrics = self._compute_metrics(X_scaled, labels, 'dbscan')
        metrics['n_noise'] = n_noise
        metrics['noise_ratio'] = n_noise / len(labels)
        
        # Visualize
        self._plot_dbscan(X_scaled, labels, df_result)
        
        # Save
        output_path = os.path.join(self.output_dir, 'dbscan_segmentation.csv')
        df_result.to_csv(output_path, index=False)
        print(f"\n💾 Saved: {output_path}")
        
        print("\n✅ DBSCAN complete!")
        return {
            'df': df_result,
            'labels': labels,
            'model': dbscan,
            'metrics': metrics,
            'features': features,
            'parameters': {'eps': eps, 'min_samples': min_samples}
        }
    
    def _optimize_dbscan(self, X: np.ndarray) -> Tuple[float, int]:
        """Find optimal DBSCAN parameters using knee detection."""
        # Calculate min_samples (2% of data)
        min_samples = max(5, min(50, int(0.02 * X.shape[0])))
        
        # Find epsilon using k-distance graph
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors_fit = neighbors.fit(X)
        distances, _ = neighbors_fit.kneighbors(X)
        
        k_distances = np.sort(distances[:, min_samples-1])
        
        # Use 85th percentile as epsilon
        eps = np.percentile(k_distances, 85)
        eps = max(0.1, min(2.0, eps))  # Reasonable bounds
        
        return eps, min_samples
    
    def _assign_dbscan_perks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign perks to DBSCAN clusters based on profiles."""
        seg_config = self.config.get('segmentation', {})
        perks = seg_config.get('all_perks', [
            "1 night free hotel plus flight",
            "exclusive discounts",
            "free checked bags",
            "free hotel meal",
            "no cancellation fees"
        ])
        names = seg_config.get('group_names', [
            f"Segment {i}" for i in range(len(perks))
        ])
        
        # Get non-noise clusters
        valid_clusters = sorted([c for c in df['cluster'].unique() if c != -1])
        
        # Sort by avg spend
        cluster_spending = df[df['cluster'].isin(valid_clusters)].groupby(
            'cluster'
        )['total_spend'].mean().sort_values(ascending=False)
        
        # Assign perks
        cluster_to_perk = {}
        cluster_to_name = {}
        for idx, cluster_id in enumerate(cluster_spending.index):
            perk_idx = min(idx, len(perks)-1)
            cluster_to_perk[cluster_id] = perks[perk_idx]
            cluster_to_name[cluster_id] = names[perk_idx]
        
        # Noise gets baseline perk
        cluster_to_perk[-1] = perks[-1]
        cluster_to_name[-1] = "Unassigned (Noise)"
        
        df['assigned_perk'] = df['cluster'].map(cluster_to_perk)
        df['segment_name'] = df['cluster'].map(cluster_to_name)
        
        # Print assignments
        print("\n[PERKS] DBSCAN assignments:")
        for c in sorted(valid_clusters):
            count = (df['cluster'] == c).sum()
            avg_spend = df[df['cluster'] == c]['total_spend'].mean()
            print(f"   Cluster {c}: {cluster_to_name[c]}")
            print(f"      {count:,} customers | Avg spend ${avg_spend:,.0f}")
            print(f"      Perk: {cluster_to_perk[c]}")
        
        if -1 in df['cluster'].values:
            noise_count = (df['cluster'] == -1).sum()
            print(f"   Noise: {noise_count:,} customers → {perks[-1]}")
        
        return df
    
    # ========================================================================
    # METRICS & VISUALIZATION
    # ========================================================================
    
    def _compute_metrics(
        self, 
        X: np.ndarray, 
        labels: np.ndarray,
        algorithm: str
    ) -> Dict:
        """Compute clustering quality metrics."""
        metrics = {'algorithm': algorithm}
        
        # Handle noise points
        mask = labels != -1
        X_valid = X[mask]
        labels_valid = labels[mask]
        
        if len(set(labels_valid)) > 1:
            try:
                metrics['silhouette'] = silhouette_score(X_valid, labels_valid)
                metrics['davies_bouldin'] = davies_bouldin_score(X_valid, labels_valid)
                metrics['calinski'] = calinski_harabasz_score(X_valid, labels_valid)
            except:  # noqa: E722
                metrics['silhouette'] = -1
                metrics['davies_bouldin'] = 999
                metrics['calinski'] = 0
        else:
            metrics['silhouette'] = -1
            metrics['davies_bouldin'] = 999
            metrics['calinski'] = 0
        
        metrics['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
        
        return metrics
    
    def _plot_kmeans(
        self, 
        X: np.ndarray, 
        labels: np.ndarray, 
        df: pd.DataFrame,
        n_clusters: int
    ):
        """Create K-Means visualizations."""
        # 2D projection
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X)
        else:
            X_2d = X[:, :2]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter plot
        scatter = ax1.scatter(
            X_2d[:, 0], X_2d[:, 1], 
            c=labels, cmap='viridis', 
            s=30, alpha=0.6
        )
        ax1.set_title(f'K-Means Clusters (n={n_clusters})', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Component 1')
        ax1.set_ylabel('Component 2')
        plt.colorbar(scatter, ax=ax1, label='Cluster')
        
        # Distribution
        cluster_counts = df['cluster'].value_counts().sort_index()
        bars = ax2.bar(range(len(cluster_counts)), cluster_counts.values, color='steelblue')
        ax2.set_title('Cluster Sizes', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Count')
        ax2.set_xticks(range(len(cluster_counts)))
        
        # Add labels
        for bar, count in zip(bars, cluster_counts.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{count:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = os.path.join(self.fig_dir, 'kmeans_results.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"📊 Saved plot: {save_path}")
    
    def _plot_dbscan(self, X: np.ndarray, labels: np.ndarray, df: pd.DataFrame):
        """Create DBSCAN visualizations."""
        # 2D projection
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter plot
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                color = 'gray'
                alpha = 0.2
            else:
                alpha = 0.6
            
            mask = labels == label
            ax1.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=[color], s=30, alpha=alpha,
                label=f'Cluster {label}' if label != -1 else 'Noise'
            )
        
        ax1.set_title('DBSCAN Clusters', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Component 1')
        ax1.set_ylabel('Component 2')
        ax1.legend(loc='best', fontsize=8)
        
        # Distribution
        cluster_data = df[df['cluster'] != -1]
        cluster_counts = cluster_data['cluster'].value_counts().sort_index()
        
        bars = ax2.bar(range(len(cluster_counts)), cluster_counts.values, color='steelblue')
        ax2.set_title('Cluster Sizes (excluding noise)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Count')
        ax2.set_xticks(range(len(cluster_counts)))
        
        for bar, count in zip(bars, cluster_counts.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{count:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = os.path.join(self.fig_dir, 'dbscan_results.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"📊 Saved plot: {save_path}")
    
    # ========================================================================
    # RUN BOTH
    # ========================================================================
    
    def run_both(self, df: pd.DataFrame, n_clusters: int = 5) -> Dict:
        """Run both K-Means and DBSCAN and compare."""
        print("\n" + "="*70)
        print("RUNNING BOTH ALGORITHMS")
        print("="*70)
        
        results = {}
        
        # K-Means
        try:
            results['kmeans'] = self.run_kmeans(df, n_clusters=n_clusters)
        except Exception as e:
            print(f"❌ K-Means failed: {e}")
            results['kmeans'] = None
        
        # DBSCAN
        try:
            results['dbscan'] = self.run_dbscan(df)
        except Exception as e:
            print(f"❌ DBSCAN failed: {e}")
            results['dbscan'] = None
        
        # Compare
        if results['kmeans'] and results['dbscan']:
            self._compare_algorithms(results)
        
        return results
    
    def _compare_algorithms(self, results: Dict):
        """Print comparison between algorithms."""
        print("\n" + "="*70)
        print("ALGORITHM COMPARISON")
        print("="*70)
        
        km = results['kmeans']['metrics']
        db = results['dbscan']['metrics']
        
        print(f"\n{'Metric':<25} {'K-Means':<15} {'DBSCAN':<15}")
        print("-" * 55)
        print(f"{'Clusters':<25} {km['n_clusters']:<15} {db['n_clusters']:<15}")
        print(f"{'Silhouette':<25} {km['silhouette']:<15.3f} {db['silhouette']:<15.3f}")
        print(f"{'Davies-Bouldin':<25} {km['davies_bouldin']:<15.3f} {db['davies_bouldin']:<15.3f}")
        print(f"{'Calinski-Harabasz':<25} {km['calinski']:<15.1f} {db['calinski']:<15.1f}")
        
        if 'noise_ratio' in db:
            print(f"{'Noise Ratio':<25} {'0%':<15} {db['noise_ratio']:<15.1%}")
        
        print("\n" + "="*70)


# ============================================================================
# QUICK START EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('path/to/your/data.csv')
    
    # Initialize
    ml = MLClustering(config_path='config/ml_config.yaml')
    
    # Run both algorithms
    results = ml.run_both(df, n_clusters=5)
    
    # Access results
    kmeans_customers = results['kmeans']['df']
    dbscan_customers = results['dbscan']['df']
    
    print("\n✅ Clustering complete!")
    print(f"   K-Means: {len(kmeans_customers):,} customers")
    print(f"   DBSCAN: {len(dbscan_customers):,} customers")