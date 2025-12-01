# core/segment/ml_model/perk_assigner.py

import pandas as pd # type: ignore
import numpy as np # type: ignore  # noqa: F401
from typing import Dict, Any, List  # noqa: F401
from IPython.display import display  # type: ignore


class PerkAssigner:
    """
    Assigns perks and segment names to clusters based on behavioral profiles.
    Works for both K-Means and DBSCAN algorithms.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        seg_config = config.get('segmentation', {})

        # Load perks, names, and perk_groups
        self.all_perks = seg_config.get('all_perks', [])
        self.group_names = seg_config.get('group_names', [])
        self.perk_groups = seg_config.get('perk_groups', [])

        # Build mapping dictionaries from perk_groups
        self.group_to_perk = {pg['group']: pg['perk'] for pg in self.perk_groups}
        self.perk_to_group = {pg['perk']: pg['group'] for pg in self.perk_groups}

        # Thresholds for scoring
        self.thresholds = self._extract_thresholds(seg_config)

    def _extract_thresholds(self, seg_config: Dict) -> Dict[str, float]:
        """Extract threshold values from config and print descriptions."""
        threshold_defs = seg_config.get('threshold_definitions', {})
        rows = []
        thresholds = {}
        for key, defn in threshold_defs.items():
            val = defn.get('fallback', 0)
            thresholds[key.lower()] = val
            rows.append({
                "Metric": key,
                "Column": defn.get('column'),
                "Threshold": val,
                "Quantile": defn.get('quantile'),
                "Description": defn.get('description', '')
            })
        print("[THRESHOLDS] Loaded fallback thresholds:")
        display(pd.DataFrame(rows))
        return {
            'spend': thresholds.get('total_spend', 4000),
            'trips': thresholds.get('trip_count', 1.5),
            'browsing': thresholds.get('browsing_rate', 0.6),
            'hotel': thresholds.get('money_spent_hotel_total', 1200),
            'business': thresholds.get('business_rate', 0.2),
            'group': thresholds.get('group_rate', 0.1),
            'bags': thresholds.get('avg_bags', 1.0),
        }

    def assign_perks_and_names(self, df: pd.DataFrame, algorithm: str = 'kmeans') -> pd.DataFrame:
        print(f"[PERK ASSIGNMENT] Assigning perks for {algorithm}...")
        if algorithm == 'kmeans':
            return self._assign_kmeans_perks(df)
        else:
            return self._assign_dbscan_perks(df)

    def _assign_kmeans_perks(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        unique_clusters = sorted(df['cluster'].unique())

        # Rank clusters by avg spend
        profiles = [self._compute_cluster_profile(df, cid) for cid in unique_clusters]
        sorted_profiles = sorted(profiles, key=lambda x: x['avg_spend'], reverse=True)

        cluster_to_perk, cluster_to_name = {}, {}
        for idx, profile in enumerate(sorted_profiles):
            cluster_id = profile['cluster_id']
            if idx < len(self.perk_groups):
                perk = self.perk_groups[idx]['perk']
                group = self.perk_groups[idx]['group']
            else:
                perk = self.all_perks[min(idx, len(self.all_perks)-1)]
                group = self.group_names[min(idx, len(self.group_names)-1)]
            cluster_to_perk[cluster_id] = perk
            cluster_to_name[cluster_id] = group

        df['assigned_perk'] = df['cluster'].map(cluster_to_perk)
        df['segment_name'] = df['cluster'].map(cluster_to_name)
        self._print_assignment_summary(df, cluster_to_name, cluster_to_perk, 'K-Means')
        return df

    def _assign_dbscan_perks(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Get all unique cluster IDs, excluding noise (-1)
        unique_clusters = sorted([c for c in df['cluster'].unique() if c != -1])

        if not unique_clusters:
            # Handle case where only noise or no data exists
            default_perk = self.all_perks[-1] if self.all_perks else "Standard Perk"
            df['assigned_perk'] = default_perk
            df['segment_name'] = "Unassigned (Noise)"
            return df

        profiles = []
        for cid in unique_clusters:
            profile = self._compute_cluster_profile(df, cid)
            profile['scores'] = self._compute_segment_scores(profile)
            profiles.append(profile)

        # 1. Greedy matching using perk_groups (Original logic)
        cluster_to_perk, cluster_to_name = {}, {}
        segment_types = ['vip','browser','group','hotel_business','baseline']
        
        # This can assign multiple segment types to the same cluster ID,
        # leaving other cluster IDs (like 1) unassigned.
        for seg_type, pg in zip(segment_types, self.perk_groups):
            best_cluster = max(profiles, key=lambda p: p['scores'].get(seg_type,0))
            cluster_to_perk[best_cluster['cluster_id']] = pg['perk']
            cluster_to_name[best_cluster['cluster_id']] = pg['group']

        # 2. FIX: Ensure all unique clusters have an assignment (Resolves KeyError)
        
        # Define a fallback perk/name (using the last available or a default)
        default_perk = self.all_perks[-1] if self.all_perks else "Standard Discount"
        default_name = "Unmatched Cluster"
        
        # Find all cluster IDs that were missed by the greedy matching
        unassigned_clusters = set(unique_clusters) - set(cluster_to_perk.keys())
        
        if unassigned_clusters:
            print(f"   ⚠️ DBSCAN found {len(unassigned_clusters)} cluster(s) {list(unassigned_clusters)} not matched by the greedy assignment logic. Assigning default segment: '{default_name}'.")
            for cluster_id in unassigned_clusters:
                # Add the missing cluster ID (e.g., np.int64(1)) to the dictionaries
                cluster_to_perk[cluster_id] = default_perk 
                cluster_to_name[cluster_id] = default_name

        # 3. Noise (-1) assignment
        if -1 in df['cluster'].values:
            noise_perk = self.all_perks[-1] if self.all_perks else "No Perk"
            cluster_to_perk[-1] = noise_perk
            cluster_to_name[-1] = "Unassigned (Noise)"

        # 4. Apply the mappings
        df['assigned_perk'] = df['cluster'].map(cluster_to_perk)
        df['segment_name'] = df['cluster'].map(cluster_to_name)
        self._print_assignment_summary(df, cluster_to_name, cluster_to_perk, 'DBSCAN')
        return df

    def _compute_cluster_profile(self, df: pd.DataFrame, cluster_id: int) -> Dict[str, float]:
        cluster_data = df[df['cluster'] == cluster_id]
        def safe_mean(col, default=0):
            return float(cluster_data[col].mean()) if col in cluster_data.columns else default
        return {
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'avg_spend': safe_mean('total_spend'),
            'avg_trips': safe_mean('num_trips'),
            'avg_browsing_rate': safe_mean('browsing_rate'),
            'avg_hotel_spend': safe_mean('money_spent_hotel_total'),
            'avg_business_rate': safe_mean('business_rate'),
            'avg_group_rate': safe_mean('group_rate'),
            'avg_bags': safe_mean('avg_bags'),
            'avg_conversion': safe_mean('conversion_rate'),
        }

    def _compute_segment_scores(self, profile: Dict) -> Dict[str, float]:
        T = self.thresholds
        scores = {
            'vip': (profile['avg_spend']/T['spend'])*0.5 + (profile['avg_trips']/T['trips'])*0.3 + (profile['avg_conversion']/0.5)*0.2,
            'browser': (profile['avg_browsing_rate']/T['browsing'])*0.5 + (profile['avg_spend']/T['spend'])*0.3 + (profile['avg_conversion']/0.5)*0.2,
            'group': (profile['avg_group_rate']/T['group'])*0.5 + (profile['avg_bags']/T['bags'])*0.3 + (profile['avg_trips']/T['trips'])*0.2,
            'hotel_business': (profile['avg_hotel_spend']/T['hotel'])*0.4 + (profile['avg_business_rate']/T['business'])*0.4 + (profile['avg_spend']/T['spend'])*0.2,
        }
        scores['baseline'] = 1.0 / (1 + sum(scores.values()))
        return scores

    def _print_assignment_summary(
        self,
        df: pd.DataFrame,
        cluster_to_name: Dict,
        cluster_to_perk: Dict,
        algorithm: str
    ):
        """Print summary of perk assignments."""
        print(f"\n📊 {algorithm} Perk Assignments")
        print("="*60)
        summary_rows = []

        # Normal clusters
        for cid in sorted([c for c in df['cluster'].unique() if c != -1]):
            cluster_data = df[df['cluster'] == cid]
            count, pct = len(cluster_data), (len(cluster_data)/len(df))*100
            summary_rows.append({
                "Cluster": cid,
                "Segment": cluster_to_name[cid],
                "Perk": cluster_to_perk[cid],
                "Size": count,
                "Percentage": f"{pct:.1f}%",
                "Avg Spend": f"${cluster_data['total_spend'].mean():.0f}" if 'total_spend' in df else "N/A",
                "Avg Trips": f"{cluster_data['num_trips'].mean():.1f}" if 'num_trips' in df else "N/A"
            })

        # Noise cluster (DBSCAN)
        if -1 in df['cluster'].values:
            noise_count = (df['cluster'] == -1).sum()
            noise_pct = (noise_count / len(df)) * 100
            summary_rows.append({
                "Cluster": -1,
                "Segment": "Unassigned (Noise)",
                "Perk": cluster_to_perk.get(-1, "N/A"),
                "Size": noise_count,
                "Percentage": f"{noise_pct:.1f}%",
                "Avg Spend": "N/A",
                "Avg Trips": "N/A"
            })

        # Display summary table
        summary_df = pd.DataFrame(summary_rows)
        display(summary_df)
        print("="*60)
        
    def get_segment_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for each segment.
        """
        if 'segment_name' not in df.columns:
            raise ValueError("DataFrame must have 'segment_name' column")

        # Aggregation dictionary
        agg_dict = {}
        if 'user_id' in df.columns:
            agg_dict['user_id'] = 'count'

        numeric_cols = [
            'total_spend', 'num_trips', 'conversion_rate',
            'business_rate', 'group_rate', 'browsing_rate'
        ]
        for col in numeric_cols:
            if col in df.columns:
                agg_dict[col] = ['mean', 'median']

        # Group by segment + perk
        summary = df.groupby(['segment_name', 'assigned_perk']).agg(agg_dict)

        # Flatten multi-index columns
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
        summary = summary.reset_index()

        # Add percentage of total
        total = len(df)
        count_col = [c for c in summary.columns if 'count' in c.lower()]
        if count_col:
            summary['percentage'] = (summary[count_col[0]] / total * 100).round(1)

        display(summary)
        return summary
