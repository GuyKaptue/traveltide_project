# core/segment/ml_model/visualizer.py

import os
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from typing import Dict, Any, Optional  # noqa: F401
from sklearn.decomposition import PCA # type: ignore

try:
    import plotly.graph_objects as go # type: ignore  # noqa: F401
    import plotly.express as px # type: ignore  # noqa: F401
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class ClusterVisualizer:
    """
    Handles all visualization tasks for clustering results.
    """
    
    def __init__(self, output_dir: str, config: Dict[str, Any]):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        output_dir : str
            Directory for saving figures
        config : Dict[str, Any]
            Configuration dictionary
        """
        self.output_dir = output_dir
        self.config = config
        self.html_dir = os.path.join(output_dir, 'html')
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.html_dir, exist_ok=True)
        
        # Get visualization settings
        viz_config = config.get('segmentation', {}).get('visualization', {})
        self.enable_2d = viz_config.get('enable_2d_plot', True)
        self.enable_3d = viz_config.get('enable_3d_plot', False)
        
        # Set style
        sns.set_style("whitegrid")
    
    def plot_kmeans_results(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        df: pd.DataFrame,
        metrics: Dict
    ):
        """
        Create comprehensive K-Means visualizations.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (scaled, possibly PCA)
        labels : np.ndarray
            Cluster labels
        df : pd.DataFrame
            Original data with assignments
        metrics : Dict
            Computed metrics
        """
        print("[VISUALIZATION] Creating K-Means plots...")
        
        # Segment summary table 
        if 'assigned_perk' in df.columns:
            self._plot_segment_summary(df, 'kmeans')
        
        # 2D scatter + distribution
        self._plot_clusters_2d(X, labels, df, 'kmeans', metrics)
        
        # Cluster profiles heatmap
        if 'segment_name' in df.columns:
            self._plot_cluster_profiles(df, 'kmeans')
        
        # Metrics dashboard
        self._plot_metrics_dashboard(metrics, 'kmeans')
        
        print(f"   ✅ Visualizations saved to {self.output_dir}")
    
    def plot_dbscan_results(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        df: pd.DataFrame,
        metrics: Dict
    ):
        """
        Create comprehensive DBSCAN visualizations.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (scaled)
        labels : np.ndarray
            Cluster labels
        df : pd.DataFrame
            Original data with assignments
        metrics : Dict
            Computed metrics
        """
        print("[VISUALIZATION] Creating DBSCAN plots...")
        # Segment summary table 
        if 'assigned_perk' in df.columns:
            self._plot_segment_summary(df, 'dbscan')
        
        # 2D scatter + distribution
        self._plot_clusters_2d(X, labels, df, 'dbscan', metrics)
        
        # Cluster profiles heatmap
        if 'segment_name' in df.columns:
            self._plot_cluster_profiles(df, 'dbscan')
        
        # Metrics dashboard
        self._plot_metrics_dashboard(metrics, 'dbscan')
        
        # DBSCAN-specific: noise analysis
        self._plot_noise_analysis(df)
        
        print(f"   ✅ Visualizations saved to {self.output_dir}")
    
    def _plot_segment_summary(self, df: pd.DataFrame, algorithm: str):
        """
        Create interactive segment summary table (HTML + PNG export).
        """
        if not PLOTLY_AVAILABLE:
            print("   ⚠️ Plotly is required for the Segment Summary table. Skipping.")
            return
        if 'assigned_perk' not in df.columns:
            print("   ⚠️ DataFrame must contain 'assigned_perk' column for summary table. Skipping.")
            return

        print(f"   [TABLE] Generating {algorithm} segment summary...")

        # 1. Prepare summary data by grouping on final assignments
        # Use a list of tuples to ensure the segment order is retained as found in the DF
        summary_group = df.groupby(['assigned_perk', 'segment_name']).agg(
            count=('segment_name', 'size'),
            avg_spend=('total_spend', 'mean') if 'total_spend' in df.columns else ('segment_name', 'size'),
            avg_trips=('num_trips', 'mean') if 'num_trips' in df.columns else ('segment_name', 'size'),
        ).reset_index()

        total_users = len(df)
        summary_group['Percentage'] = (summary_group['count'] / total_users * 100).round(1).astype(str) + '%'
        
        # Format columns, handling cases where the column might be the dummy 'size' column
        summary_group['Avg Spend'] = summary_group['avg_spend'].apply(
            lambda x: f"${x:.0f}" if isinstance(x, (int, float)) and 'total_spend' in df.columns else "N/A"
        )
        summary_group['Avg Trips'] = summary_group['avg_trips'].apply(
            lambda x: f"{x:.1f}" if isinstance(x, (int, float)) and 'num_trips' in df.columns else "N/A"
        )
        
        # Rename and select final columns
        summary_df = summary_group.rename(columns={
            'segment_name': 'Segment',
            'assigned_perk': 'Assigned Perk',
            'count': 'Count'
        })
        summary_df = summary_df[['Segment', 'Assigned Perk', 'Count', 'Percentage', 'Avg Spend', 'Avg Trips']]

        # --- Zebra row colors ---
        row_colors = ['#ffffff' if i % 2 == 0 else '#f9f9f9' for i in range(len(summary_df))]

        # --- Plotly Table ---
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(summary_df.columns),
                fill_color="#4CAF50",
                font=dict(color="white", size=13),
                align="center"
            ),
            cells=dict(
                values=[summary_df[col] for col in summary_df.columns],
                fill_color=[row_colors],
                align="center",
                font=dict(size=11)
            )
        )])

        fig.update_layout(
            title=f"{algorithm.upper()} Segment Summary with Perks",
            margin=dict(l=10, r=10, t=40, b=10)
        )

        # --- Save HTML and PNG ---
        
        html_path = os.path.join(self.html_dir, f"{algorithm}_segment_summary_table.html")
        png_path = os.path.join(self.output_dir, f"{algorithm}_segment_summary_table.png")

        fig.write_html(html_path, include_plotlyjs="cdn")

        # PNG (requires kaleido)
        try:
            # We use the class output directory for PNG
            fig.write_image(png_path) 
            print(f"   🖼 PNG saved: {png_path}")
        except Exception as e:
            print(f"   ⚠ PNG export failed for {algorithm}: {e}")
            print("   Install kaleido using: pip install -U kaleido")

        print(f"   📊 HTML saved: {html_path}")

        # Console output
        print("-" * 80)
        print(f"{algorithm.upper()} SEGMENT SUMMARY")
        print("-" * 80)
        print(summary_df.to_string(index=False))
    
    def _plot_clusters_2d(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        df: pd.DataFrame,
        algorithm: str,
        metrics: Dict
    ):
        """Create 2D scatter plot and distribution charts."""
        # Reduce to 2D if needed
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X)
            var_explained = pca.explained_variance_ratio_
        else:
            X_2d = X[:, :2]
            var_explained = [1.0, 1.0]
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Plot 1: Scatter plot
        ax1 = axes[0]
        
        if algorithm == 'dbscan':
            # Special coloring for DBSCAN (noise in gray)
            unique_labels = set(labels)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(sorted(unique_labels), colors):
                if label == -1:
                    color = 'gray'
                    alpha = 0.2
                    label_text = 'Noise'
                else:
                    alpha = 0.6
                    label_text = f'Cluster {label}'
                
                mask = labels == label
                ax1.scatter(
                    X_2d[mask, 0], X_2d[mask, 1],
                    c=[color], s=30, alpha=alpha, label=label_text
                )
        else:
            # K-Means: standard coloring
            scatter = ax1.scatter(
                X_2d[:, 0], X_2d[:, 1],
                c=labels, cmap='viridis',
                s=30, alpha=0.6
            )
            plt.colorbar(scatter, ax=ax1, label='Cluster')
        
        ax1.set_title(
            f'{algorithm.upper()} Clusters\n'
            f'Silhouette: {metrics.get("silhouette", -1):.3f}',
            fontsize=12, fontweight='bold'
        )
        ax1.set_xlabel(f'PC1 ({var_explained[0]:.1%} var)')
        ax1.set_ylabel(f'PC2 ({var_explained[1]:.1%} var)')
        if algorithm == 'dbscan':
            ax1.legend(loc='best', fontsize=8)
        
        # Plot 2: Cluster sizes
        ax2 = axes[1]
        
        if algorithm == 'dbscan':
            # Exclude noise for main chart
            valid_labels = [l for l in labels if l != -1]  # noqa: E741
            if valid_labels:
                unique, counts = np.unique(valid_labels, return_counts=True)
                bars = ax2.bar(range(len(unique)), counts, color='steelblue')
                ax2.set_xticks(range(len(unique)))
                ax2.set_xticklabels([f'C{l}' for l in unique])  # noqa: E741
        else:
            unique, counts = np.unique(labels, return_counts=True)
            bars = ax2.bar(range(len(unique)), counts, color='steelblue')
            ax2.set_xticks(range(len(unique)))
            ax2.set_xticklabels([f'C{l}' for l in unique])  # noqa: E741
        
        ax2.set_title('Cluster Sizes', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Count')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax2.text(
                bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{count:,}', ha='center', va='bottom', fontsize=9
            )
        
        # Plot 3: Segment distribution (pie)
        ax3 = axes[2]
        
        if 'segment_name' in df.columns:
            segment_counts = df['segment_name'].value_counts()
            colors_pie = plt.cm.Set3(range(len(segment_counts)))
            
            wedges, texts, autotexts = ax3.pie(
                segment_counts.values,
                labels=None,
                autopct='%1.1f%%',
                colors=colors_pie,
                startangle=90
            )
            
            # Add legend
            ax3.legend(
                [f'{name[:20]}...' if len(name) > 20 else name 
                 for name in segment_counts.index],
                loc='center left',
                bbox_to_anchor=(1, 0, 0.5, 1),
                fontsize=8
            )
            
            ax3.set_title('Segment Distribution', fontsize=12, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No segment data', ha='center', va='center')
            ax3.set_title('Segment Distribution', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, f'{algorithm}_clusters_2d.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"   📊 Saved: {save_path}")
    
    def _plot_cluster_profiles(self, df: pd.DataFrame, algorithm: str):
        """Create heatmap of cluster profiles."""
        # Select key features for profiling
        profile_features = [
            'total_spend', 'num_trips', 'conversion_rate',
            'browsing_rate', 'business_rate', 'group_rate',
            'money_spent_hotel_total', 'avg_bags'
        ]
        
        available_features = [f for f in profile_features if f in df.columns]
        
        if not available_features or 'cluster' not in df.columns:
            return
        
        # Compute mean values per cluster
        cluster_profiles = df.groupby('cluster')[available_features].mean()
        
        # Normalize for better visualization
        cluster_profiles_norm = (cluster_profiles - cluster_profiles.min()) / (
            cluster_profiles.max() - cluster_profiles.min()
        )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(
            cluster_profiles_norm.T,
            annot=cluster_profiles.T,
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Normalized Value'},
            ax=ax
        )
        
        ax.set_title(
            f'{algorithm.upper()} Cluster Profiles',
            fontsize=14, fontweight='bold'
        )
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, f'{algorithm}_cluster_profiles.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"   📊 Saved: {save_path}")
    
    def _plot_metrics_dashboard(self, metrics: Dict, algorithm: str):
        """Create metrics dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Metric 1: Silhouette score (gauge-style)
        ax1 = axes[0, 0]
        sil_score = metrics.get('silhouette', -1)
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        sil_normalized = (sil_score + 1) / 2  # -1 to 1 → 0 to 1
        
        ax1.barh([0], [sil_normalized], color=colors[int(sil_normalized * 4)], height=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_yticks([])
        ax1.set_xlabel('Score')
        ax1.set_title(f'Silhouette Score: {sil_score:.3f}', fontweight='bold')
        ax1.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Metric 2: Cluster sizes
        ax2 = axes[0, 1]
        cluster_sizes = metrics.get('cluster_sizes', [])
        if cluster_sizes:
            ax2.bar(range(len(cluster_sizes)), cluster_sizes, color='skyblue')
            ax2.set_xlabel('Cluster')
            ax2.set_ylabel('Size')
            ax2.set_title('Cluster Size Distribution', fontweight='bold')
            ax2.axhline(
                np.mean(cluster_sizes), 
                color='red', 
                linestyle='--', 
                label=f'Mean: {np.mean(cluster_sizes):.0f}'
            )
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No cluster size data', ha='center', va='center')
            ax2.set_title('Cluster Size Distribution', fontweight='bold')
        
        # Metric 3: Quality metrics comparison
        ax3 = axes[1, 0]
        metric_names = ['Silhouette', 'Davies-Bouldin', 'Calinski-H']
        metric_values = [
            (metrics.get('silhouette', -1) + 1) / 2,  # Normalize
            1 - min(metrics.get('davies_bouldin', 3), 3) / 3,  # Invert and normalize
            min(metrics.get('calinski', 0), 1000) / 1000  # Normalize
        ]
        
        bars = ax3.barh(metric_names, metric_values, color=['green', 'blue', 'purple'])
        ax3.set_xlim(0, 1)
        ax3.set_xlabel('Normalized Score')
        ax3.set_title('Quality Metrics (normalized)', fontweight='bold')
        
        for bar, val in zip(bars, metric_values):
            ax3.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center')
        
        # Metric 4: Summary text
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
{algorithm.upper()} CLUSTERING SUMMARY
{'='*35}

Clusters: {metrics['n_clusters']}
Total Samples: {metrics['n_samples']:,}

Quality Metrics:
  • Silhouette: {metrics.get('silhouette', -1):.3f}
  • Davies-Bouldin: {metrics.get('davies_bouldin', 999):.3f}
  • Calinski-Harabasz: {metrics.get('calinski', 0):.1f}
"""
        
        if algorithm == 'dbscan':
            summary_text += f"""
DBSCAN Specific:
  • Noise Points: {metrics.get('n_noise', 0):,}
  • Noise Ratio: {metrics.get('noise_ratio', 0):.1%}
  • Cluster Stability: {metrics.get('cluster_stability', 0):.3f}
"""
        else:
            summary_text += f"""
K-Means Specific:
  • Cluster Balance: {metrics.get('cluster_balance', 0):.3f}
  • Min Cluster Size: {metrics.get('min_cluster_size', 0):,}
  • Max Cluster Size: {metrics.get('max_cluster_size', 0):,}
"""
        
        validation = metrics.get('validation', {})
        if validation.get('overall_quality'):
            summary_text += "\n✅ Quality: PASS"
        else:
            summary_text += "\n⚠️  Quality: NEEDS IMPROVEMENT"
        
        ax4.text(
            0.1, 0.9, summary_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, f'{algorithm}_metrics_dashboard.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"   📊 Saved: {save_path}")
    
    def _plot_noise_analysis(self, df: pd.DataFrame):
        """Create noise analysis plot for DBSCAN."""
        if 'is_noise' not in df.columns:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Noise vs non-noise counts
        ax1 = axes[0]
        noise_counts = df['is_noise'].value_counts()
        
        bars = ax1.bar(
            ['Valid Clusters', 'Noise'],
            [noise_counts.get(False, 0), noise_counts.get(True, 0)],
            color=['green', 'red']
        )
        
        ax1.set_title('Noise Analysis', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Count')
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2, height,
                f'{int(height):,}', ha='center', va='bottom'
            )
        
        # Plot 2: Noise characteristics
        ax2 = axes[1]
        
        if noise_counts.get(True, 0) > 0:
            noise_df = df[df['is_noise'] == True]  # noqa: E712
            regular_df = df[df['is_noise'] == False]  # noqa: E712
            
            # Compare key metrics
            metrics_compare = []
            for col in ['total_spend', 'num_trips', 'conversion_rate']:
                if col in df.columns:
                    noise_mean = noise_df[col].mean()
                    regular_mean = regular_df[col].mean()
                    metrics_compare.append((col, noise_mean, regular_mean))
            
            if metrics_compare:
                x = np.arange(len(metrics_compare))
                width = 0.35
                
                noise_vals = [m[1] for m in metrics_compare]
                regular_vals = [m[2] for m in metrics_compare]
                
                # Normalize for comparison
                max_vals = [max(n, r) for n, r in zip(noise_vals, regular_vals)]
                noise_norm = [n/m if m > 0 else 0 for n, m in zip(noise_vals, max_vals)]
                regular_norm = [r/m if m > 0 else 0 for r, m in zip(regular_vals, max_vals)]
                
                ax2.bar(x - width/2, noise_norm, width, label='Noise', color='red', alpha=0.7)
                ax2.bar(x + width/2, regular_norm, width, label='Valid', color='green', alpha=0.7)
                
                ax2.set_xlabel('Metric')
                ax2.set_ylabel('Normalized Value')
                ax2.set_title('Noise vs Valid Cluster Characteristics', fontweight='bold')
                ax2.set_xticks(x)
                ax2.set_xticklabels([m[0] for m in metrics_compare], rotation=45, ha='right')
                ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No noise points found', ha='center', va='center')
            ax2.set_title('Noise Characteristics', fontweight='bold')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'dbscan_noise_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"   📊 Saved: {save_path}")
    
    def plot_comparison(self, results: Dict):
        """Create comparison visualization between algorithms."""
        if not results.get('kmeans') or not results.get('dbscan'):
            print("   ⚠️ Both algorithms needed for comparison")
            return
        
        km_metrics = results['kmeans']['metrics']
        db_metrics = results['dbscan']['metrics']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Metric comparison
        ax1 = axes[0, 0]
        metrics_to_compare = ['silhouette', 'davies_bouldin', 'calinski']
        x = np.arange(len(metrics_to_compare))
        width = 0.35
        
        km_vals = [
            km_metrics.get('silhouette', -1),
            1 / km_metrics.get('davies_bouldin', 1),  # Invert for comparison
            km_metrics.get('calinski', 0) / 100
        ]
        db_vals = [
            db_metrics.get('silhouette', -1),
            1 / db_metrics.get('davies_bouldin', 1),
            db_metrics.get('calinski', 0) / 100
        ]
        
        ax1.bar(x - width/2, km_vals, width, label='K-Means', color='blue', alpha=0.7)
        ax1.bar(x + width/2, db_vals, width, label='DBSCAN', color='green', alpha=0.7)
        
        ax1.set_ylabel('Score')
        ax1.set_title('Quality Metrics Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Silhouette', 'DB (inv)', 'Calinski/100'])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Cluster count comparison
        ax2 = axes[0, 1]
        ax2.bar(
            ['K-Means', 'DBSCAN'],
            [km_metrics['n_clusters'], db_metrics['n_clusters']],
            color=['blue', 'green'], alpha=0.7
        )
        ax2.set_ylabel('Number of Clusters')
        ax2.set_title('Clusters Found', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (algo, count) in enumerate([('K-Means', km_metrics['n_clusters']), 
                                            ('DBSCAN', db_metrics['n_clusters'])]):
            ax2.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Segment distribution comparison
        ax3 = axes[1, 0]
        
        km_segments = results['kmeans']['df']['segment_name'].value_counts()
        db_segments = results['dbscan']['df']['segment_name'].value_counts()
        
        all_segments = sorted(set(km_segments.index) | set(db_segments.index))
        x = np.arange(len(all_segments))
        width = 0.35
        
        km_counts = [km_segments.get(s, 0) for s in all_segments]
        db_counts = [db_segments.get(s, 0) for s in all_segments]
        
        ax3.barh(x - width/2, km_counts, width, label='K-Means', color='blue', alpha=0.7)
        ax3.barh(x + width/2, db_counts, width, label='DBSCAN', color='green', alpha=0.7)
        
        ax3.set_yticks(x)
        ax3.set_yticklabels([s[:30] for s in all_segments], fontsize=9)
        ax3.set_xlabel('Count')
        ax3.set_title('Segment Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        
        # Plot 4: Recommendation text
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Determine better algorithm
        km_score = (km_metrics.get('silhouette', -1) + 1) / 2
        db_score = (db_metrics.get('silhouette', -1) + 1) / 2
        
        if db_metrics.get('noise_ratio', 0) > 0.3:
            db_score *= 0.7  # Penalize high noise
        
        comparison_text = f"""
ALGORITHM COMPARISON
{'='*40}

K-Means:
  • Silhouette: {km_metrics.get('silhouette', -1):.3f}
  • Clusters: {km_metrics['n_clusters']}
  • All users assigned

DBSCAN:
  • Silhouette: {db_metrics.get('silhouette', -1):.3f}
  • Clusters: {db_metrics['n_clusters']}
  • Noise: {db_metrics.get('noise_ratio', 0):.1%}

RECOMMENDATION:
"""
        
        if km_score > db_score:
            comparison_text += "  ✅ K-Means performs better\n"
            comparison_text += "  Reasons:\n"
            comparison_text += "  • Better silhouette score\n"
            comparison_text += "  • All customers assigned\n"
        elif db_score > km_score:
            comparison_text += "  ✅ DBSCAN performs better\n"
            comparison_text += "  Reasons:\n"
            comparison_text += "  • Better cluster separation\n"
            comparison_text += "  • Natural groupings found\n"
        else:
            comparison_text += "  ⚖️  Similar performance\n"
            comparison_text += "  Choose based on:\n"
            comparison_text += "  • Business requirements\n"
            comparison_text += "  • Noise tolerance\n"
        
        ax4.text(
            0.1, 0.9, comparison_text,
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        )
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'algorithm_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"   📊 Saved: {save_path}")