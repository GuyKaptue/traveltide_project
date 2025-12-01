# core/segment/ml_model/metrics_calculator.py

import numpy as np # type: ignore
import pandas as pd # type: ignore
from typing import Dict, Any
from sklearn.metrics import ( # type: ignore
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score
)


class MetricsCalculator:
    """
    Calculates clustering quality metrics for both K-Means and DBSCAN.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metrics calculator.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
        """
        self.config = config
        
        # Get validation thresholds from config
        validation_config = config.get('segmentation', {}).get('metrics', {}).get(
            'validation_thresholds', {}
        )
        
        self.thresholds = {
            'min_silhouette': validation_config.get('min_silhouette', 0.25),
            'max_davies_bouldin': validation_config.get('max_davies_bouldin', 2.0),
            'min_calinski': validation_config.get('min_calinski_harabasz', 100),
            'max_noise_ratio': validation_config.get('max_noise_ratio', 0.3)
        }
    
    def compute_kmeans_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute comprehensive metrics for K-Means clustering.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        labels : np.ndarray
            Cluster labels
        df : pd.DataFrame
            Original dataframe with assignments
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of metrics
        """
        print("[METRICS] Computing K-Means metrics...")
        
        metrics = {
            'algorithm': 'kmeans',
            'n_clusters': len(set(labels)),
            'n_samples': len(labels)
        }
        
        # Internal clustering metrics
        try:
            metrics['silhouette'] = silhouette_score(X, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
            metrics['calinski'] = calinski_harabasz_score(X, labels)
        except Exception as e:
            print(f"   ⚠️ Warning computing metrics: {e}")
            metrics['silhouette'] = -1
            metrics['davies_bouldin'] = 999
            metrics['calinski'] = 0
        
        # Cluster balance
        cluster_sizes = [np.sum(labels == i) for i in range(metrics['n_clusters'])]
        if np.mean(cluster_sizes) > 0:
            metrics['cluster_balance'] = 1 - (np.std(cluster_sizes) / np.mean(cluster_sizes))
        else:
            metrics['cluster_balance'] = 0
        
        metrics['cluster_sizes'] = cluster_sizes
        metrics['min_cluster_size'] = min(cluster_sizes)
        metrics['max_cluster_size'] = max(cluster_sizes)
        
        # Business metrics
        if 'segment_name' in df.columns:
            metrics.update(self._compute_business_metrics(df))
        
        # Validation
        metrics['validation'] = self._validate_kmeans_metrics(metrics)
        
        self._print_metrics(metrics, 'K-Means')
        
        return metrics
    
    def compute_dbscan_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute comprehensive metrics for DBSCAN clustering.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        labels : np.ndarray
            Cluster labels (-1 for noise)
        df : pd.DataFrame
            Original dataframe with assignments
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of metrics
        """
        print("[METRICS] Computing DBSCAN metrics...")
        
        # Basic counts
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        n_total = len(labels)
        noise_ratio = n_noise / n_total
        
        metrics = {
            'algorithm': 'dbscan',
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio,
            'n_samples': n_total
        }
        
        # Internal clustering metrics (excluding noise)
        if n_clusters >= 2:
            mask = labels != -1
            X_valid = X[mask]
            labels_valid = labels[mask]
            
            if len(set(labels_valid)) > 1 and len(X_valid) > n_clusters:
                try:
                    metrics['silhouette'] = silhouette_score(X_valid, labels_valid)
                    metrics['davies_bouldin'] = davies_bouldin_score(X_valid, labels_valid)
                    metrics['calinski'] = calinski_harabasz_score(X_valid, labels_valid)
                except Exception as e:
                    print(f"   ⚠️ Warning computing metrics: {e}")
                    metrics['silhouette'] = -1
                    metrics['davies_bouldin'] = 999
                    metrics['calinski'] = 0
            else:
                metrics['silhouette'] = -1
                metrics['davies_bouldin'] = 999
                metrics['calinski'] = 0
        else:
            metrics['silhouette'] = -1
            metrics['davies_bouldin'] = 999
            metrics['calinski'] = 0
        
        # Cluster sizes (excluding noise)
        valid_clusters = [l for l in set(labels) if l != -1]  # noqa: E741
        cluster_sizes = [np.sum(labels == i) for i in valid_clusters]
        
        if cluster_sizes:
            metrics['cluster_sizes'] = cluster_sizes
            metrics['min_cluster_size'] = min(cluster_sizes)
            metrics['max_cluster_size'] = max(cluster_sizes)
            
            # Cluster balance (excluding noise)
            if np.mean(cluster_sizes) > 0:
                metrics['cluster_balance'] = 1 - (np.std(cluster_sizes) / np.mean(cluster_sizes))
            else:
                metrics['cluster_balance'] = 0
        else:
            metrics['cluster_sizes'] = []
            metrics['min_cluster_size'] = 0
            metrics['max_cluster_size'] = 0
            metrics['cluster_balance'] = 0
        
        # DBSCAN-specific metrics
        metrics.update(self._compute_dbscan_quality_metrics(X, labels))
        
        # Business metrics
        if 'segment_name' in df.columns:
            metrics.update(self._compute_business_metrics(df))
        
        # Validation
        metrics['validation'] = self._validate_dbscan_metrics(metrics)
        
        self._print_metrics(metrics, 'DBSCAN')
        
        return metrics
    
    def _compute_dbscan_quality_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute DBSCAN-specific quality metrics.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        labels : np.ndarray
            Cluster labels
            
        Returns
        -------
        Dict[str, float]
            DBSCAN-specific metrics
        """
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters < 2:
            return {
                'cluster_stability': 0,
                'density_connectedness': 0,
                'meaningful_cluster_ratio': 0
            }
        
        # Cluster stability (based on intra-cluster distances)
        stability_scores = []
        
        for cluster_id in range(n_clusters):
            if cluster_id in labels:
                cluster_points = X[labels == cluster_id]
                if len(cluster_points) > 1:
                    centroid = cluster_points.mean(axis=0)
                    distances = np.linalg.norm(cluster_points - centroid, axis=1)
                    stability = 1 / (1 + np.mean(distances))
                    stability_scores.append(stability)
        
        cluster_stability = np.mean(stability_scores) if stability_scores else 0
        
        # Meaningful clusters (with stability > 0.5)
        meaningful_clusters = sum(1 for s in stability_scores if s > 0.5)
        meaningful_ratio = meaningful_clusters / n_clusters if n_clusters > 0 else 0
        
        return {
            'cluster_stability': cluster_stability,
            'density_connectedness': cluster_stability,  # Proxy metric
            'meaningful_cluster_ratio': meaningful_ratio
        }
    
    def _compute_business_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute business-aligned metrics.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with segment assignments
            
        Returns
        -------
        Dict[str, Any]
            Business metrics
        """
        business_metrics = {}
        
        # VIP segment metrics
        if 'total_spend' in df.columns and 'num_trips' in df.columns:
            spend_threshold = df['total_spend'].quantile(0.9)
            trips_threshold = df['num_trips'].quantile(0.9)
            
            vips = df[
                (df['total_spend'] >= spend_threshold) &
                (df['num_trips'] >= trips_threshold)
            ]
            
            if len(vips) > 0 and 'cluster' in df.columns:
                # Check if VIPs are concentrated in one cluster
                vip_cluster_dist = vips['cluster'].value_counts()
                if len(vip_cluster_dist) > 0:
                    business_metrics['vip_cluster_purity'] = vip_cluster_dist.max() / len(vips)
                else:
                    business_metrics['vip_cluster_purity'] = 0
            else:
                business_metrics['vip_cluster_purity'] = 0
        
        # Segment distinctness
        key_dimensions = ['total_spend', 'num_trips', 'business_rate', 'conversion_rate']
        distinctness_scores = []
        
        if 'cluster' in df.columns:
            for dim in key_dimensions:
                if dim in df.columns:
                    cluster_means = df.groupby('cluster')[dim].mean()
                    if len(cluster_means) > 1 and cluster_means.std() > 0:
                        cv = cluster_means.std() / cluster_means.mean() if cluster_means.mean() != 0 else 0
                        distinctness_scores.append(min(cv, 1.0))
        
        business_metrics['segment_distinctness'] = (
            np.mean(distinctness_scores) if distinctness_scores else 0
        )
        
        # Overall business alignment score
        business_metrics['business_alignment'] = (
            business_metrics.get('vip_cluster_purity', 0) * 0.4 +
            business_metrics['segment_distinctness'] * 0.6
        )
        
        return business_metrics
    
    def _validate_kmeans_metrics(self, metrics: Dict) -> Dict[str, bool]:
        """
        Validate K-Means metrics against thresholds.
        
        Parameters
        ----------
        metrics : Dict
            Computed metrics
            
        Returns
        -------
        Dict[str, bool]
            Validation results
        """
        validation = {
            'silhouette_ok': metrics.get('silhouette', -1) >= self.thresholds['min_silhouette'],
            'davies_bouldin_ok': metrics.get('davies_bouldin', 999) <= self.thresholds['max_davies_bouldin'],
            'calinski_ok': metrics.get('calinski', 0) >= self.thresholds['min_calinski'],
            'cluster_size_ok': metrics.get('min_cluster_size', 0) >= 50
        }
        
        validation['overall_quality'] = all(validation.values())
        
        return validation
    
    def _validate_dbscan_metrics(self, metrics: Dict) -> Dict[str, bool]:
        """
        Validate DBSCAN metrics against thresholds.
        
        Parameters
        ----------
        metrics : Dict
            Computed metrics
            
        Returns
        -------
        Dict[str, bool]
            Validation results
        """
        validation = {
            'silhouette_ok': metrics.get('silhouette', -1) >= self.thresholds['min_silhouette'],
            'noise_ratio_ok': metrics.get('noise_ratio', 1) <= self.thresholds['max_noise_ratio'],
            'cluster_count_ok': 3 <= metrics.get('n_clusters', 0) <= 8,
            'cluster_size_ok': metrics.get('min_cluster_size', 0) >= 50
        }
        
        validation['overall_quality'] = all(validation.values())
        
        return validation
    
    def _print_metrics(self, metrics: Dict, algorithm: str):
        """Print formatted metrics summary."""
        print(f"\n   ✅ {algorithm} Metrics:")
        print("   " + "="*60)
        
        # Core metrics
        print(f"   Clusters: {metrics['n_clusters']}")
        print(f"   Silhouette Score: {metrics.get('silhouette', -1):.3f}")
        print(f"   Davies-Bouldin: {metrics.get('davies_bouldin', 999):.3f}")
        print(f"   Calinski-Harabasz: {metrics.get('calinski', 0):.1f}")
        
        # Algorithm-specific
        if algorithm == 'DBSCAN':
            print(f"   Noise Points: {metrics['n_noise']:,} ({metrics['noise_ratio']:.1%})")
            print(f"   Cluster Stability: {metrics.get('cluster_stability', 0):.3f}")
        else:
            print(f"   Cluster Balance: {metrics.get('cluster_balance', 0):.3f}")
        
        # Business metrics
        if 'business_alignment' in metrics:
            print(f"   Business Alignment: {metrics['business_alignment']:.3f}")
        
        # Validation
        validation = metrics.get('validation', {})
        if validation.get('overall_quality'):
            print("   ✅ Quality: PASS")
        else:
            print("   ⚠️ Quality: NEEDS IMPROVEMENT")
            failed = [k for k, v in validation.items() if not v and k != 'overall_quality']
            if failed:
                print(f"      Failed checks: {', '.join(failed)}")
        
        print("   " + "="*60)
    
    def compare_algorithms(
        self,
        kmeans_metrics: Dict,
        dbscan_metrics: Dict
    ) -> Dict[str, Any]:
        """
        Compare metrics between K-Means and DBSCAN.
        
        Parameters
        ----------
        kmeans_metrics : Dict
            K-Means metrics
        dbscan_metrics : Dict
            DBSCAN metrics
            
        Returns
        -------
        Dict[str, Any]
            Comparison results with recommendation
        """
        comparison = {
            'kmeans_score': self._calculate_overall_score(kmeans_metrics),
            'dbscan_score': self._calculate_overall_score(dbscan_metrics),
        }
        
        # Determine winner
        if comparison['kmeans_score'] > comparison['dbscan_score']:
            comparison['recommendation'] = 'kmeans'
            comparison['reason'] = 'Better overall clustering quality'
        elif comparison['dbscan_score'] > comparison['kmeans_score']:
            comparison['recommendation'] = 'dbscan'
            comparison['reason'] = 'Better cluster separation and quality'
        else:
            comparison['recommendation'] = 'tie'
            comparison['reason'] = 'Similar performance, choose based on business needs'
        
        # Add noise consideration for DBSCAN
        if dbscan_metrics.get('noise_ratio', 0) > 0.3:
            comparison['warning'] = 'DBSCAN has high noise ratio (>30%)'
        
        return comparison
    
    def _calculate_overall_score(self, metrics: Dict) -> float:
        """
        Calculate overall quality score from metrics.
        
        Parameters
        ----------
        metrics : Dict
            Metrics dictionary
            
        Returns
        -------
        float
            Overall score (0-1)
        """
        # Normalize silhouette (from [-1, 1] to [0, 1])
        sil_score = (metrics.get('silhouette', -1) + 1) / 2
        
        # Normalize Davies-Bouldin (lower is better, cap at 3)
        db_score = 1 - min(metrics.get('davies_bouldin', 3), 3) / 3
        
        # Normalize Calinski (higher is better, cap at 1000)
        ch_score = min(metrics.get('calinski', 0), 1000) / 1000
        
        # Cluster balance/stability
        if metrics['algorithm'] == 'kmeans':
            balance_score = metrics.get('cluster_balance', 0)
        else:
            balance_score = metrics.get('cluster_stability', 0)
        
        # Business alignment
        business_score = metrics.get('business_alignment', 0.5)
        
        # Weighted average
        overall = (
            sil_score * 0.3 +
            db_score * 0.2 +
            ch_score * 0.2 +
            balance_score * 0.15 +
            business_score * 0.15
        )
        
        # Penalty for DBSCAN with high noise
        if metrics['algorithm'] == 'dbscan':
            noise_penalty = metrics.get('noise_ratio', 0) * 0.5
            overall = overall * (1 - noise_penalty)
        
        return overall