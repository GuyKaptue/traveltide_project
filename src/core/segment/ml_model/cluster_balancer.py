# core/segment/ml_model/cluster_balancer.py

import numpy as np # type: ignore
import pandas as pd # type: ignore
from typing import Dict, Any, List, Tuple
from IPython.display import display # type: ignore

class ClusterBalancer:
    """
    Rebalance clusters to meet distribution and business constraints.

    Reads parameters from config/ml_config.yaml:
    - segmentation.target_distribution (min_pct, max_pct, preferred)
    - segmentation.clustering.kmeans.rebalancing (enabled, method, max_iterations)
    - segmentation.business_rules (baseline_max_percentage, segment_min_percentage, enforce_vip_spend_threshold)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Distribution constraints
        td_cfg = config.get("segmentation", {}).get("target_distribution", {})
        self.min_pct = td_cfg.get("min_pct", 0.0) / 100.0
        self.max_pct = td_cfg.get("max_pct", 1.0) / 100.0
        self.preferred = td_cfg.get("preferred", [])

        # Rebalancing settings
        reb_cfg = (config.get("segmentation", {})
                          .get("clustering", {})
                          .get("kmeans", {})
                          .get("rebalancing", {}))
        self.enabled = reb_cfg.get("enabled", False)
        self.method = reb_cfg.get("method", "distance_based")
        self.max_iterations = reb_cfg.get("max_iterations", 3)

        # Business rules
        br_cfg = config.get("segmentation", {}).get("business_rules", {})
        self.baseline_max_pct = br_cfg.get("baseline_max_percentage", 25.0) / 100.0
        self.segment_min_pct = br_cfg.get("segment_min_percentage", 10.0) / 100.0
        self.enforce_vip_spend = br_cfg.get("enforce_vip_spend_threshold", True)

    # ---------------- Public API ----------------

    def rebalance_if_needed(
        self,
        X: np.ndarray,
        df: pd.DataFrame,
        labels: np.ndarray,
        n_clusters: int
    ) -> np.ndarray:
        """
        Rebalance clusters if distribution violates config constraints.
        """
        if not self.enabled:
            print("   → Rebalancing disabled; skipping.")
            return labels

        shares = self._compute_shares(labels, n_clusters)
        self._print_distribution("   Current distribution", shares)

        if not self._is_imbalanced(shares):
            print("   → Distribution within bounds; no rebalancing needed.")
            return labels

        # Iterative rebalancing
        for it in range(1, self.max_iterations + 1):
            print(f"[REBALANCING] Iteration {it}/{self.max_iterations}")
            overfull, underfull = self._find_pressure_clusters(shares)

            if not overfull and not underfull:
                print("   → No clusters to adjust; stopping.")
                break

            labels = self._rebalance_step(X, labels, overfull, underfull)
            shares = self._compute_shares(labels, n_clusters)
            self._print_distribution("   Updated distribution", shares)

            if not self._is_imbalanced(shares):
                print("   ✅ Rebalancing achieved within bounds.")
                break

        # Apply business rules
        labels = self._apply_business_rules(df, labels)

        return labels

    # ---------------- Internals ----------------

    def _compute_shares(self, labels: np.ndarray, n_clusters: int) -> np.ndarray:
        counts = np.array([np.sum(labels == k) for k in range(n_clusters)], dtype=float)
        total = len(labels)
        return counts / max(total, 1)

    def _is_imbalanced(self, shares: np.ndarray) -> bool:
        return bool(np.any(shares < self.min_pct) or np.any(shares > self.max_pct))

    def _find_pressure_clusters(self, shares: np.ndarray) -> Tuple[List[int], List[int]]:
        overfull = [i for i, s in enumerate(shares) if s > self.max_pct]
        underfull = [i for i, s in enumerate(shares) if s < self.min_pct]
        return overfull, underfull

    def _rebalance_step(self, X: np.ndarray, labels: np.ndarray,
                        overfull: List[int], underfull: List[int]) -> np.ndarray:
        # Simple distance-based reassignment
        centers = [X[labels == k].mean(axis=0) for k in set(labels)]
        centers = np.vstack(centers)

        for c_over in overfull:
            idx_over = np.where(labels == c_over)[0]
            if len(idx_over) == 0:
                continue

            # Distances to all centers
            dists = np.sqrt(((X[idx_over][:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
            nearest = np.argmin(dists, axis=1)

            for i, target in zip(idx_over, nearest):
                if target in underfull:
                    labels[i] = target

        return labels

    def _apply_business_rules(self, df: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
        n_clusters = len(np.unique(labels))
        shares = self._compute_shares(labels, n_clusters)

        # Baseline max percentage rule
        baseline_idx = np.where(labels == 4)[0]  # assuming cluster 4 = baseline
        if len(baseline_idx) / len(labels) > self.baseline_max_pct:
            print("⚠️ Baseline cluster exceeds max percentage; consider threshold adjustment.")

        # Segment min percentage rule
        if np.any(shares < self.segment_min_pct):
            print("⚠️ Some clusters below minimum percentage; consider rebalancing thresholds.")

        # VIP existence & spend rule
        vip_idx = np.where(labels == 0)[0]
        if len(vip_idx) == 0:
            print("⚠️ VIP cluster is empty → reassigning top spenders to VIP")
            # Sort by spend and take top N users
            top_spenders = df['total_spend'].nlargest(int(len(df) * self.segment_min_pct)).index
            labels[top_spenders] = 0
            vip_idx = np.where(labels == 0)[0]

        if self.enforce_vip_spend and "total_spend" in df.columns:
            vip_spend = df.loc[vip_idx, "total_spend"].mean()
            other_means = df.loc[labels != 0].groupby(labels[labels != 0])["total_spend"].mean()
            if len(other_means) > 0 and vip_spend <= other_means.max():
                print("⚠️ VIP cluster does not have highest spend → reassigning top spenders")
                # Find cluster with highest spend
                max_cluster = other_means.idxmax()
                candidates = df.loc[labels == max_cluster].sort_values("total_spend", ascending=False)
                # Move top X% into VIP
                n_move = max(1, int(len(df) * self.segment_min_pct))
                move_idx = candidates.head(n_move).index
                labels[move_idx] = 0

        # Print final distribution summary
        final_shares = self._compute_shares(labels, n_clusters)
        self._print_distribution("   Final distribution after business rules", final_shares)

        # Stakeholder-friendly summary table
        summary_rows = []
        for cid in range(n_clusters):
            cluster_data = df[labels == cid]
            summary_rows.append({
                "Cluster": cid,
                "Size": len(cluster_data),
                "Percentage": f"{(len(cluster_data)/len(df))*100:.1f}%",
                "Avg Spend": f"${cluster_data['total_spend'].mean():.0f}" if 'total_spend' in df else "N/A"
            })
        display(pd.DataFrame(summary_rows))

        return labels

    def _print_distribution(self, title: str, shares: np.ndarray) -> None:
        print(title)
        for k, pct in enumerate(shares * 100):
            print(f"      Cluster {k}: {pct:.1f}%")
