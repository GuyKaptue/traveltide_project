# core/segment/ml_model/kmeans_engine.py

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import Dict, Any, Optional
from sklearn.cluster import KMeans  # type: ignore
from sklearn.decomposition import PCA  # type: ignore

from .cluster_balancer import ClusterBalancer  # type: ignore


class KMeansEngine:
    """
    K-Means clustering engine with optional PCA and rebalancing.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize K-Means engine.
        """
        self.config = config
        self.model = None
        self.pca_model = None
        self.labels = None

        # Extract K-Means config
        clustering_config = config.get('segmentation', {}).get('clustering', {})
        self.kmeans_config = clustering_config.get('kmeans', {})
        self.n_clusters = clustering_config.get('n_clusters', 5)
        self.random_state = clustering_config.get('random_state', 42)
        self.enable_pca = clustering_config.get('enable_pca', True)
        self.n_components = clustering_config.get('n_components', 10)

    def apply_pca(
        self,
        X_scaled: np.ndarray,
        n_components: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply PCA dimensionality reduction.
        """
        if not self.enable_pca:
            print("[PCA] Skipped (disabled in config)")
            return X_scaled

        if n_components is None:
            n_components = self.n_components

        print(f"[PCA] Starting PCA with {X_scaled.shape[0]} samples and {X_scaled.shape[1]} features")
        print(f"[PCA] Target components: {n_components}")

        n_components = min(n_components, X_scaled.shape[1], X_scaled.shape[0])
        print(f"[PCA] Adjusted components: {n_components}")

        pca_whiten = self.config.get('segmentation', {}).get('clustering', {}).get('pca_whiten', True)
        print(f"[PCA] Whitening enabled: {pca_whiten}")

        self.pca_model = PCA(
            n_components=n_components,
            whiten=pca_whiten,
            random_state=self.random_state
        )

        X_pca = self.pca_model.fit_transform(X_scaled)
        explained_var = self.pca_model.explained_variance_ratio_.sum()
        print(f"[PCA] ✅ PCA complete - {explained_var:.1%} variance explained")

        return X_pca

    def fit(
        self,
        X: np.ndarray,
        n_clusters: Optional[int] = None
    ) -> np.ndarray:
        """
        Fit K-Means clustering model.
        """
        if n_clusters is None:
            n_clusters = self.n_clusters

        print(f"[K-MEANS] Starting fit with {X.shape[0]} samples and {X.shape[1]} features")
        print(f"[K-MEANS] Parameters: n_clusters={n_clusters}, "
              f"n_init={self.kmeans_config.get('n_init', 20)}, "
              f"max_iter={self.kmeans_config.get('max_iter', 500)}, "
              f"algorithm={self.kmeans_config.get('algorithm', 'lloyd')}")

        self.model = KMeans(
            n_clusters=n_clusters,
            n_init=self.kmeans_config.get('n_init', 20),
            max_iter=self.kmeans_config.get('max_iter', 500),
            algorithm=self.kmeans_config.get('algorithm', 'lloyd'),
            random_state=self.random_state
        )

        self.labels = self.model.fit_predict(X)

        unique, counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)

        print(f"[K-MEANS] ✅ Fit complete - inertia={self.model.inertia_:.2f}")
        print("[K-MEANS] Initial distribution:")
        for cluster_id, count in zip(unique, counts):
            pct = (count / total) * 100
            print(f"   Cluster {cluster_id}: {count:,} ({pct:.1f}%)")

        return self.labels

    def get_cluster_centers(self) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted yet")
        return self.model.cluster_centers_

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)

    def get_inertia(self) -> float:
        if self.model is None:
            raise ValueError("Model not fitted yet")
        return self.model.inertia_

    def fit_and_assign(
        self,
        X_scaled: np.ndarray,
        df: pd.DataFrame,
        n_clusters: Optional[int] = 5
    ) -> Dict[str, Any]:
        """
        Complete K-Means pipeline: PCA (optional) → fit → assign.
        """
        print("\n" + "="*60)
        print("K-MEANS CLUSTERING ENGINE")
        print("="*60)
        print(f"[PIPELINE] Input shape: {X_scaled.shape}, DataFrame rows: {len(df)}")

        # Step 1: PCA
        X_pca = None
        if self.enable_pca:
            print("[PIPELINE] PCA enabled")
            X_pca = self.apply_pca(X_scaled)
            X_cluster = X_pca
        else:
            print("[PIPELINE] PCA disabled")
            X_cluster = X_scaled

        # Step 2: Fit K-Means
        labels = self.fit(X_cluster, n_clusters)

        # Step 3: Rebalancing
        rebalancing_enabled = self.kmeans_config.get('rebalancing', {}).get('enabled', False)
        print(f"[PIPELINE] Rebalancing enabled: {rebalancing_enabled}")
        if rebalancing_enabled:
            print("[REBALANCING] Running ClusterBalancer...")
            balancer = ClusterBalancer(self.config)
            labels = balancer.rebalance_if_needed(X_cluster, df, labels, self.n_clusters)

        # Step 4: Results
        results = {
            'labels': labels,
            'model': self.model,
            'pca_model': self.pca_model,
            'X_pca': X_pca,
            'X_scaled': X_scaled,
            'n_clusters': n_clusters or self.n_clusters,
            'inertia': self.get_inertia(),
            'cluster_centers': self.get_cluster_centers()
        }

        print("[PIPELINE] ✅ K-Means pipeline complete")
        return results
