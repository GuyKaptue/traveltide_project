# core/segment/ml_model/data_exporter.py

import os
import json
import pandas as pd  # type: ignore
import numpy as np   # type: ignore
from typing import Dict, Any, Optional  # noqa: F401


class DataExporter:
    """
    Handles all data export operations for clustering results.
    """

    def __init__(self, output_dir: str):
        """
        Initialize data exporter.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_results(
        self,
        df: pd.DataFrame,
        algorithm: str,
        metrics: Dict[str, Any],
        include_features: bool = True
    ) -> Dict[str, str]:
        """
        Export complete clustering results.
        """
        print(f"[EXPORT] Saving {algorithm} results...")

        exported_files = {}

        # Create algorithm-specific subdirectory
        algo_dir = os.path.join(self.output_dir, algorithm)
        os.makedirs(algo_dir, exist_ok=True)

        # 1. Main segmentation file
        main_path = self._export_main_segmentation(df, algo_dir, algorithm, include_features)
        exported_files['main'] = main_path

        # 2. Segment summary
        summary_path = self._export_segment_summary(df, algo_dir, algorithm)
        exported_files['summary'] = summary_path

        # 3. Metrics
        metrics_path = self._export_metrics(metrics, algo_dir, algorithm)
        exported_files['metrics'] = metrics_path

        # 4. Perk distribution
        perk_path = self._export_perk_distribution(df, algo_dir, algorithm)
        exported_files['perks'] = perk_path

        # 5. Campaign exports
        campaign_dir = os.path.join(algo_dir, 'campaigns')
        os.makedirs(campaign_dir, exist_ok=True)
        campaign_files = self._export_campaign_files(df, campaign_dir)
        exported_files['campaigns'] = campaign_files

        print(f"   ✅ All files saved to: {algo_dir}")
        return exported_files

    def _export_main_segmentation(
        self,
        df: pd.DataFrame,
        output_dir: str,
        algorithm: str,
        include_all: bool
    ) -> str:
        """Export main segmentation file with all assignments."""
        if include_all:
            export_df = df.copy()
        else:
            key_columns = [
                'user_id', 'cluster', 'segment_name', 'assigned_perk',
                'total_spend', 'num_trips', 'conversion_rate',
                'browsing_rate', 'business_rate', 'group_rate'
            ]
            if algorithm == 'dbscan' and 'is_noise' in df.columns:
                key_columns.append('is_noise')
            available_columns = [c for c in key_columns if c in df.columns]
            export_df = df[available_columns]

        file_path = os.path.join(output_dir, f'{algorithm}_segmentation.csv')
        export_df.to_csv(file_path, index=False)

        print(f"   📄 Segmentation: {file_path}")
        print(f"      {len(export_df):,} rows, {len(export_df.columns)} columns")

        return file_path

    def _export_segment_summary(self, df: pd.DataFrame, output_dir: str, algorithm: str) -> str:
        """Export segment summary statistics."""
        if 'segment_name' not in df.columns:
            return ""

        agg_dict = {'user_id': 'count' if 'user_id' in df.columns else lambda x: len(x)}
        numeric_cols = [
            'total_spend', 'num_trips', 'conversion_rate',
            'business_rate', 'group_rate', 'browsing_rate',
            'money_spent_hotel_total', 'avg_bags'
        ]
        for col in numeric_cols:
            if col in df.columns:
                agg_dict[col] = ['mean', 'median', 'std']

        summary = df.groupby(['segment_name', 'assigned_perk']).agg(agg_dict)
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
        summary = summary.reset_index()

        total = len(df)
        count_col = [c for c in summary.columns if 'count' in c.lower()][0]
        summary['percentage'] = (summary[count_col] / total * 100).round(2)

        file_path = os.path.join(output_dir, f'{algorithm}_segment_summary.csv')
        summary.to_csv(file_path, index=False)

        print(f"   📄 Summary: {file_path}")
        return file_path

    def _export_metrics(self, metrics: Dict[str, Any], output_dir: str, algorithm: str) -> str:
        """Export metrics to JSON and CSV."""
        json_path = os.path.join(output_dir, f'{algorithm}_metrics.json')

        metrics_clean = self._clean_metrics_for_json(metrics)

        with open(json_path, 'w') as f:
            json.dump(metrics_clean, f, indent=2)

        print(f"   📄 Metrics (JSON): {json_path}")

        csv_path = os.path.join(output_dir, f'{algorithm}_metrics.csv')
        metrics_flat = self._flatten_dict(metrics_clean)
        pd.DataFrame([metrics_flat]).to_csv(csv_path, index=False)

        print(f"   📄 Metrics (CSV): {csv_path}")
        return json_path

    def _export_perk_distribution(self, df: pd.DataFrame, output_dir: str, algorithm: str) -> str:
        """Export perk distribution summary."""
        if 'assigned_perk' not in df.columns:
            return ""

        perk_dist = df.groupby(['assigned_perk', 'segment_name']).size().reset_index(name='count')
        total = len(df)
        perk_dist['percentage'] = (perk_dist['count'] / total * 100).round(2)

        numeric_cols = ['total_spend', 'num_trips', 'conversion_rate']
        for col in numeric_cols:
            if col in df.columns:
                avg_values = df.groupby('assigned_perk')[col].mean()
                perk_dist[f'avg_{col}'] = perk_dist['assigned_perk'].map(avg_values)

        file_path = os.path.join(output_dir, f'{algorithm}_perk_distribution.csv')
        perk_dist.to_csv(file_path, index=False)

        print(f"   📄 Perk distribution: {file_path}")
        return file_path

    def _export_campaign_files(self, df: pd.DataFrame, campaign_dir: str) -> list:
        """Export separate files for each segment (campaign use)."""
        if 'segment_name' not in df.columns:
            return []

        exported_files = []
        for segment in df['segment_name'].unique():
            segment_df = df[df['segment_name'] == segment]
            campaign_columns = ['user_id', 'segment_name', 'assigned_perk']
            optional_cols = ['email', 'total_spend', 'num_trips', 'home_country']
            for col in optional_cols:
                if col in df.columns:
                    campaign_columns.append(col)

            available_columns = [c for c in campaign_columns if c in segment_df.columns]
            campaign_df = segment_df[available_columns]

            safe_name = segment.replace(' ', '_').replace('/', '_').lower()
            file_path = os.path.join(campaign_dir, f'campaign_{safe_name}.csv')
            campaign_df.to_csv(file_path, index=False)
            exported_files.append(file_path)

            print(f"   📄 Campaign file: {safe_name}.csv ({len(campaign_df):,} customers)")

        return exported_files

    def _clean_metrics_for_json(self, metrics: Dict) -> Dict:
        """Clean metrics dictionary for JSON export."""
        cleaned = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer,)):
                cleaned[key] = int(value)
            elif isinstance(value, (np.floating,)):
                cleaned[key] = float(value)
            elif isinstance(value, (np.bool_)):
                cleaned[key] = bool(value)
            elif isinstance(value, np.ndarray):
                cleaned[key] = value.tolist()
            elif isinstance(value, dict):
                cleaned[key] = self._clean_metrics_for_json(value)
            elif isinstance(value, list):
                cleaned[key] = [
                    int(v) if isinstance(v, np.integer) else
                    float(v) if isinstance(v, np.floating) else
                    bool(v) if isinstance(v, np.bool_) else v
                    for v in value
                ]
            else:
                cleaned[key] = value
        return cleaned

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
