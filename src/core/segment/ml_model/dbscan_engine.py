# core/segment/ml_model/dbscan_engine.py

import numpy as np # type: ignore
import pandas as pd # type: ignore
from typing import Dict, Any, Optional
from sklearn.cluster import DBSCAN # type: ignore
from sklearn.neighbors import NearestNeighbors # type: ignore
from scipy.signal import argrelextrema, savgol_filter # type: ignore


class DBSCANEngine:
    """
    DBSCAN clustering engine with automatic parameter optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DBSCAN engine.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
        """
        self.config = config
        self.model = None
        self.labels = None
        self.best_eps = None
        self.best_min_samples = None
        
        # Extract DBSCAN config
        clustering_config = config.get('segmentation', {}).get('clustering', {})
        self.dbscan_config = clustering_config.get('dbscan', {})
        self.random_state = clustering_config.get('random_state', 42)
        
    def calculate_min_samples(self, n_samples: int) -> int:
        """
        Calculate optimal min_samples parameter.
        
        Parameters
        ----------
        n_samples : int
            Number of data points
            
        Returns
        -------
        int
            Optimal min_samples value
        """
        strategy = self.dbscan_config.get('min_samples_strategy', 'percentage')
        
        if strategy == 'percentage':
            percentage = self.dbscan_config.get('min_samples_percentage', 0.02)
            min_samples = max(5, min(50, int(n_samples * percentage)))
        elif strategy == 'sqrt':
            min_samples = max(5, min(50, int(np.sqrt(n_samples))))
        else:
            min_samples = self.dbscan_config.get('min_samples_fixed', 10)
        
        return min_samples
    
    def find_optimal_eps_knee(self, X: np.ndarray) -> float:
        """
        Find optimal epsilon using knee detection in k-distance graph.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
            
        Returns
        -------
        float
            Optimal epsilon value
        """
        print("   [EPS] Using knee detection method...")
        
        min_samples = self.calculate_min_samples(X.shape[0])
        n_neighbors = min_samples
        
        # Compute k-nearest neighbors
        neighbors = NearestNeighbors(n_neighbors=n_neighbors)
        neighbors_fit = neighbors.fit(X)
        distances, _ = neighbors_fit.kneighbors(X)
        
        # Sort k-distances
        k_distances = np.sort(distances[:, n_neighbors-1])
        
        try:
            # Smooth the curve
            window = min(51, len(k_distances)//2)
            if window % 2 == 0:
                window += 1
            
            k_distances_smooth = savgol_filter(
                k_distances, 
                window_length=window, 
                polyorder=2
            )
            
            # Find knee point (maximum curvature)
            second_deriv = np.gradient(np.gradient(k_distances_smooth))
            knee_points = argrelextrema(second_deriv, np.less)[0]
            
            if len(knee_points) > 0:
                optimal_eps = k_distances_smooth[knee_points[0]]
            else:
                # Fallback to 85th percentile
                optimal_eps = np.percentile(k_distances, 85)
        
        except Exception as e:
            print(f"      ⚠️ Knee detection failed: {e}")
            # Fallback to percentile
            optimal_eps = np.percentile(k_distances, 85)
        
        # Ensure reasonable bounds
        optimal_eps = max(0.1, min(2.0, optimal_eps))
        
        return optimal_eps
    
    def find_optimal_eps_grid(
        self, 
        X: np.ndarray,
        eps_range: tuple = (0.1, 2.0),
        step: float = 0.1
    ) -> float:
        """
        Find optimal epsilon using grid search.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        eps_range : tuple
            (min_eps, max_eps) range to search
        step : float
            Step size for grid search
            
        Returns
        -------
        float
            Optimal epsilon value
        """
        print("   [EPS] Using grid search method...")
        
        min_samples = self.calculate_min_samples(X.shape[0])
        min_eps, max_eps = eps_range
        
        best_eps = 0.5
        best_score = -1
        
        for eps in np.arange(min_eps, max_eps, step):
            try:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise_ratio = np.sum(labels == -1) / len(labels)
                
                # Score based on cluster count and noise
                if 3 <= n_clusters <= 8 and noise_ratio < 0.4:
                    score = n_clusters * (1 - noise_ratio)
                    
                    if score > best_score:
                        best_score = score
                        best_eps = eps
            
            except Exception:
                continue
        
        return best_eps
    
    def optimize_parameters(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Optimize DBSCAN parameters (eps and min_samples).
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with optimal parameters
        """
        print("[DBSCAN] Optimizing parameters...")
        
        # Auto-tuning config
        optimization_config = self.config.get('segmentation', {}).get(  # noqa: F841
            'dbscan_optimization', {}
        )
        
        auto_eps = self.dbscan_config.get('auto_eps', True)
        
        if auto_eps:
            # Use knee detection by default
            eps = self.find_optimal_eps_knee(X)
        else:
            # Use configured eps
            eps = self.dbscan_config.get('eps', 0.5)
        
        min_samples = self.calculate_min_samples(X.shape[0])
        metric = self.dbscan_config.get('metric', 'euclidean')
        
        self.best_eps = eps
        self.best_min_samples = min_samples
        
        params = {
            'eps': eps,
            'min_samples': min_samples,
            'metric': metric
        }
        
        print(f"   ✅ Optimal parameters:")  # noqa: F541
        print(f"      EPS: {eps:.3f}")
        print(f"      min_samples: {min_samples}")
        print(f"      metric: {metric}")
        
        return params
    
    def fit(
        self, 
        X: np.ndarray,
        eps: Optional[float] = None,
        min_samples: Optional[int] = None,
        metric: str = 'euclidean'
    ) -> np.ndarray:
        """
        Fit DBSCAN clustering model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        eps : float, optional
            Epsilon parameter
        min_samples : int, optional
            Min samples parameter
        metric : str
            Distance metric
            
        Returns
        -------
        np.ndarray
            Cluster labels (-1 for noise)
        """
        # Use optimized or provided parameters
        if eps is None:
            eps = self.best_eps or 0.5
        if min_samples is None:
            min_samples = self.best_min_samples or self.calculate_min_samples(X.shape[0])
        
        print(f"[DBSCAN] Fitting model...")  # noqa: F541
        print(f"   Parameters: eps={eps:.3f}, min_samples={min_samples}")
        
        self.model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric
        )
        
        self.labels = self.model.fit_predict(X)
        
        # Print distribution
        unique_labels = set(self.labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(self.labels).count(-1)
        noise_ratio = n_noise / len(self.labels)
        
        print(f"   ✅ DBSCAN complete")  # noqa: F541
        print(f"      Clusters found: {n_clusters}")
        print(f"      Noise points: {n_noise:,} ({noise_ratio:.1%})")
        
        # Print cluster sizes
        for label in sorted(unique_labels):
            if label != -1:
                count = np.sum(self.labels == label)
                pct = count / len(self.labels) * 100
                print(f"      Cluster {label}: {count:,} ({pct:.1f}%)")
        
        return self.labels
    
    def fit_and_assign(
        self,
        X_scaled: np.ndarray,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Complete DBSCAN pipeline: optimize → fit → assign.
        
        Parameters
        ----------
        X_scaled : np.ndarray
            Scaled feature matrix
        df : pd.DataFrame
            Original dataframe (for metadata)
            
        Returns
        -------
        Dict[str, Any]
            Results dictionary with labels, parameters, etc.
        """
        print("\n" + "="*60)
        print("DBSCAN CLUSTERING ENGINE")
        print("="*60)
        
        # Step 1: Optimize parameters
        params = self.optimize_parameters(X_scaled)
        
        # Step 2: Fit DBSCAN
        labels = self.fit(
            X_scaled,
            eps=params['eps'],
            min_samples=params['min_samples'],
            metric=params['metric']
        )
        
        # Step 3: Create results dictionary
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)
        
        results = {
            'labels': labels,
            'model': self.model,
            'parameters': params,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': n_noise / len(labels),
            'X_scaled': X_scaled
        }
        
        print("\n✅ DBSCAN pipeline complete")
        return results
    
    def predict_new(self, X: np.ndarray) -> np.ndarray:
        """
        Predict clusters for new data (approximate for DBSCAN).
        Uses nearest fitted point to assign cluster.
        
        Parameters
        ----------
        X : np.ndarray
            New data points
            
        Returns
        -------
        np.ndarray
            Predicted cluster labels
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # DBSCAN doesn't have a predict method
        # Use nearest neighbor approach
        
        
        # Get core samples
        core_samples_mask = np.zeros_like(self.labels, dtype=bool)
        core_samples_mask[self.model.core_sample_indices_] = True
        
        # Find nearest core sample for each new point
        X_core = X[core_samples_mask]
        labels_core = self.labels[core_samples_mask]
        
        nbrs = NearestNeighbors(n_neighbors=1).fit(X_core)
        distances, indices = nbrs.kneighbors(X)
        
        # Assign label of nearest core sample
        predicted_labels = labels_core[indices.flatten()]
        
        return predicted_labels