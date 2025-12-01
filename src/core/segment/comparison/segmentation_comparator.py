# core/segment/comparison/segmentation_comparator.py

import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from typing import Dict, List, Any, Optional
from sklearn.metrics import ( # type: ignore
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
from scipy.stats import chi2_contingency # type: ignore  # noqa: F401
import warnings
warnings.filterwarnings('ignore')

from src.utils import get_path  # noqa: E402
from .comparison_viz import ComparisonVisualizer  # noqa: E402


class SegmentationComparator:
    """
    Professional comparison tool for manual vs ML-based customer segmentation.
    
    Compares manual (rule-based) vs K-Means (ML-based) segmentations with:
    - Statistical alignment metrics
    - Visual comparisons
    - Business insights
    - Actionable recommendations
    """
    
    def __init__(
        self, 
        manual_segmentation: pd.DataFrame,
        ml_segmentation: pd.DataFrame,
        manual_segment_col: str = 'persona_type',
        manual_group_col: str = 'assigned_group',
        ml_segment_col: str = 'segment_name',
        ml_cluster_col: str = 'cluster',
        output_dir: Optional[str] = None
    ):
        """Initialize comparator with segmentation datasets."""
        self.manual_seg = manual_segmentation.copy()
        self.ml_seg = ml_segmentation.copy()
        
        # Column mappings
        self.manual_segment_col = manual_segment_col
        self.manual_group_col = manual_group_col
        self.ml_segment_col = ml_segment_col
        self.ml_cluster_col = ml_cluster_col
        
        # Merged data
        self.merged_data = None
        
        # Results storage
        self.metrics = {}
        self.analysis_results = {}
        
        # Output directory
        self.output_dir = output_dir or os.path.join(
            get_path('reports'), 'segment', 'comparison'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("✅ SegmentationComparator initialized")
        print(f"   Manual: {len(self.manual_seg):,} users")
        print(f"   ML: {len(self.ml_seg):,} users")
    
    def merge_datasets(self) -> pd.DataFrame:
        """Merge manual and ML segmentations on user_id."""
        print("[MERGE] Combining datasets...")
        
        # Select columns
        manual_cols = ['user_id', self.manual_segment_col, self.manual_group_col]
        ml_cols = ['user_id', self.ml_cluster_col, self.ml_segment_col]
        
        # Add perks
        if 'assigned_perk' in self.manual_seg.columns:
            manual_cols.append('assigned_perk')
        if 'assigned_perk' in self.ml_seg.columns:
            ml_cols.append('assigned_perk')
        
        # Add common features
        common_features = ['total_spend', 'num_trips', 'conversion_rate', 
                          'RFM_score', 'browsing_rate', 'business_rate']
        
        for col in common_features:
            if col in self.manual_seg.columns and col not in manual_cols:
                manual_cols.append(col)
            if col in self.ml_seg.columns and col not in ml_cols:
                ml_cols.append(col)
        
        # Filter to available
        manual_cols = [c for c in manual_cols if c in self.manual_seg.columns]
        ml_cols = [c for c in ml_cols if c in self.ml_seg.columns]
        
        # Rename perks
        manual_df = self.manual_seg[manual_cols].copy()
        if 'assigned_perk' in manual_df.columns:
            manual_df.rename(columns={'assigned_perk': 'manual_perk'}, inplace=True)
        
        ml_df = self.ml_seg[ml_cols].copy()
        if 'assigned_perk' in ml_df.columns:
            ml_df.rename(columns={'assigned_perk': 'ml_perk'}, inplace=True)
        
        # Merge
        self.merged_data = pd.merge(
            manual_df, ml_df, on='user_id', how='inner',
            suffixes=('_manual', '_ml')
        )
        
        print(f"   ✅ Merged: {len(self.merged_data):,} users")
        return self.merged_data
    
    def calculate_alignment_metrics(self) -> Dict[str, float]:
        """Calculate statistical alignment metrics."""
        print("[METRICS] Calculating alignment...")
        
        if self.merged_data is None:
            self.merge_datasets()
        
        # Convert to numeric labels
        manual_labels = pd.Categorical(
            self.merged_data[self.manual_segment_col]
        ).codes
        ml_labels = pd.Categorical(
            self.merged_data[self.ml_segment_col]
        ).codes
        
        metrics = {
            'adjusted_rand_index': adjusted_rand_score(manual_labels, ml_labels),
            'normalized_mutual_info': normalized_mutual_info_score(manual_labels, ml_labels),
            'fowlkes_mallows_score': fowlkes_mallows_score(manual_labels, ml_labels),
            'homogeneity': homogeneity_score(manual_labels, ml_labels),
            'completeness': completeness_score(manual_labels, ml_labels),
            'v_measure': v_measure_score(manual_labels, ml_labels),
        }
        
        self.metrics = metrics
        
        print(f"   ✅ Adjusted Rand Index: {metrics['adjusted_rand_index']:.3f}")
        print(f"   ✅ Normalized Mutual Info: {metrics['normalized_mutual_info']:.3f}")
        
        return metrics
    
    def create_cross_tabulation(self) -> pd.DataFrame:
        """Create cross-tabulation matrix."""
        if self.merged_data is None:
            self.merge_datasets()
        
        cross_tab = pd.crosstab(
            self.merged_data[self.manual_segment_col],
            self.merged_data[self.ml_segment_col],
            margins=True,
            margins_name="Total"
        )
        
        return cross_tab
    
    def analyze_segment_overlap(self) -> Dict[str, Dict]:
        """Analyze how segments overlap between approaches."""
        if self.merged_data is None:
            self.merge_datasets()
        
        overlap_analysis = {}
        
        for manual_seg in self.merged_data[self.manual_segment_col].unique():
            seg_data = self.merged_data[
                self.merged_data[self.manual_segment_col] == manual_seg
            ]
            
            ml_dist = seg_data[self.ml_segment_col].value_counts(normalize=True)
            
            overlap_analysis[manual_seg] = {
                'size': len(seg_data),
                'dominant_ml_segment': ml_dist.index[0] if len(ml_dist) > 0 else None,
                'dominant_percentage': ml_dist.iloc[0] if len(ml_dist) > 0 else 0,
                'distribution': ml_dist.to_dict(),
                'purity': ml_dist.iloc[0] if len(ml_dist) > 0 else 0
            }
        
        self.analysis_results['overlap'] = overlap_analysis
        return overlap_analysis
    
    def compare_perk_assignments(self) -> pd.DataFrame:
        """Compare perk assignments between approaches."""
        if self.merged_data is None:
            self.merge_datasets()
        
        if 'manual_perk' not in self.merged_data.columns or 'ml_perk' not in self.merged_data.columns:
            print("   ⚠️ Perk columns not available")
            return pd.DataFrame()
        
        # Count agreements
        agreement = self.merged_data['manual_perk'] == self.merged_data['ml_perk']
        agreement_rate = agreement.sum() / len(self.merged_data)
        
        # Create comparison
        perk_comp = self.merged_data.groupby(['manual_perk', 'ml_perk']).size()
        perk_comp = perk_comp.reset_index(name='count')
        perk_comp['percentage'] = perk_comp['count'] / len(self.merged_data) * 100
        
        print(f"\n   Perk Agreement Rate: {agreement_rate:.1%}")
        
        return perk_comp
    
    def analyze_feature_differences(self) -> Dict[str, Any]:
        """Analyze feature distributions across segments."""
        if self.merged_data is None:
            self.merge_datasets()
        
        # Key features to compare
        features = ['total_spend', 'num_trips', 'conversion_rate', 'RFM_score']
        available_features = [f for f in features if f in self.merged_data.columns]
        
        feature_comparison = {}
        
        for feature in available_features:
            # Handle duplicated columns
            feat_col = feature
            if f'{feature}_manual' in self.merged_data.columns:
                feat_col = f'{feature}_manual'
            elif f'{feature}_ml' in self.merged_data.columns:
                feat_col = f'{feature}_ml'
            
            if feat_col in self.merged_data.columns:
                manual_means = self.merged_data.groupby(
                    self.manual_segment_col
                )[feat_col].mean()
                
                ml_means = self.merged_data.groupby(
                    self.ml_segment_col
                )[feat_col].mean()
                
                feature_comparison[feature] = {
                    'manual_means': manual_means.to_dict(),
                    'ml_means': ml_means.to_dict(),
                    'manual_std': self.merged_data.groupby(
                        self.manual_segment_col
                    )[feat_col].std().to_dict(),
                    'ml_std': self.merged_data.groupby(
                        self.ml_segment_col
                    )[feat_col].std().to_dict(),
                }
        
        self.analysis_results['features'] = feature_comparison
        return feature_comparison
    
    def calculate_segment_stability(self) -> Dict[str, float]:
        """Calculate segment stability metrics."""
        if self.merged_data is None:
            self.merge_datasets()
        
        # Purity: % of dominant ML segment in each manual segment
        purity_scores = []
        for manual_seg in self.merged_data[self.manual_segment_col].unique():
            seg_data = self.merged_data[
                self.merged_data[self.manual_segment_col] == manual_seg
            ]
            ml_dist = seg_data[self.ml_segment_col].value_counts(normalize=True)
            purity_scores.append(ml_dist.iloc[0] if len(ml_dist) > 0 else 0)
        
        avg_purity = np.mean(purity_scores)
        
        # Consistency: % of users staying in similar segments
        stability = {
            'average_purity': avg_purity,
            'purity_scores': dict(zip(
                self.merged_data[self.manual_segment_col].unique(),
                purity_scores
            )),
            'min_purity': min(purity_scores) if purity_scores else 0,
            'max_purity': max(purity_scores) if purity_scores else 0,
        }
        
        return stability
    
    def visualize_comparison(self, save: bool = True) -> plt.Figure:
        """Create comprehensive comparison visualizations."""
        print("[VISUALIZATION] Creating plots...")
        
        if self.merged_data is None:
            self.merge_datasets()
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Segment size comparison
        ax1 = fig.add_subplot(gs[0, 0])
        manual_dist = self.merged_data[self.manual_segment_col].value_counts()
        ml_dist = self.merged_data[self.ml_segment_col].value_counts()
        
        x = np.arange(max(len(manual_dist), len(ml_dist)))
        width = 0.35
        
        ax1.bar(x[:len(manual_dist)] - width/2, manual_dist.values, 
                width, label='Manual', alpha=0.8, color='steelblue')
        ax1.bar(x[:len(ml_dist)] + width/2, ml_dist.values, 
                width, label='ML', alpha=0.8, color='coral')
        ax1.set_title('Segment Size Comparison', fontweight='bold')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Cross-tabulation heatmap
        ax2 = fig.add_subplot(gs[0, 1:])
        cross_tab = pd.crosstab(
            self.merged_data[self.manual_segment_col],
            self.merged_data[self.ml_segment_col]
        )
        
        sns.heatmap(
            cross_tab, annot=True, fmt='d', cmap='YlOrRd',
            ax=ax2, cbar_kws={'label': 'User Count'}
        )
        ax2.set_title('Segment Alignment Matrix', fontweight='bold')
        ax2.set_xlabel('ML Segments')
        ax2.set_ylabel('Manual Segments')
        
        # 3. Overlap distribution (normalized)
        ax3 = fig.add_subplot(gs[1, :2])
        overlap_norm = pd.crosstab(
            self.merged_data[self.manual_segment_col],
            self.merged_data[self.ml_segment_col],
            normalize='index'
        )
        
        overlap_norm.plot(kind='bar', stacked=True, ax=ax3, 
                         colormap='tab10', alpha=0.8)
        ax3.set_title('Manual Segments Distribution in ML Segments', 
                     fontweight='bold')
        ax3.set_ylabel('Proportion')
        ax3.set_xlabel('Manual Segments')
        ax3.legend(title='ML Segments', bbox_to_anchor=(1.05, 1), 
                  loc='upper left', fontsize=8)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Metrics dashboard
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis('off')
        
        if self.metrics:
            metrics_text = "ALIGNMENT METRICS\n" + "="*25 + "\n\n"
            metrics_text += f"Adjusted Rand: {self.metrics['adjusted_rand_index']:.3f}\n"
            metrics_text += f"Norm. Mutual Info: {self.metrics['normalized_mutual_info']:.3f}\n"
            metrics_text += f"Fowlkes-Mallows: {self.metrics['fowlkes_mallows_score']:.3f}\n"
            metrics_text += f"Homogeneity: {self.metrics['homogeneity']:.3f}\n"
            metrics_text += f"Completeness: {self.metrics['completeness']:.3f}\n"
            metrics_text += f"V-Measure: {self.metrics['v_measure']:.3f}\n"
        else:
            self.calculate_alignment_metrics()
            metrics_text = "Run calculate_alignment_metrics()"
        
        ax4.text(
            0.1, 0.9, metrics_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        )
        
        # 5. Feature comparison (if available)
        ax5 = fig.add_subplot(gs[2, :2])
        
        feature_col = None
        for col in ['total_spend', 'RFM_score', 'num_trips']:
            if col in self.merged_data.columns:
                feature_col = col
                break
            elif f'{col}_manual' in self.merged_data.columns:
                feature_col = f'{col}_manual'
                break
        
        if feature_col:
            manual_means = self.merged_data.groupby(
                self.manual_segment_col
            )[feature_col].mean()
            ml_means = self.merged_data.groupby(
                self.ml_segment_col
            )[feature_col].mean()
            
            x = np.arange(max(len(manual_means), len(ml_means)))
            width = 0.35
            
            ax5.bar(x[:len(manual_means)] - width/2, manual_means.values,
                   width, label='Manual', alpha=0.8, color='steelblue')
            ax5.bar(x[:len(ml_means)] + width/2, ml_means.values,
                   width, label='ML', alpha=0.8, color='coral')
            
            ax5.set_title(f'Average {feature_col.replace("_", " ").title()} by Segment',
                         fontweight='bold')
            ax5.set_ylabel(feature_col.replace('_', ' ').title())
            ax5.legend()
            ax5.grid(axis='y', alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No feature data available', 
                    ha='center', va='center')
        
        # 6. Perk agreement (if available)
        ax6 = fig.add_subplot(gs[2, 2])
        
        if 'manual_perk' in self.merged_data.columns and 'ml_perk' in self.merged_data.columns:
            agreement = self.merged_data['manual_perk'] == self.merged_data['ml_perk']
            agree_pct = agreement.sum() / len(self.merged_data) * 100
            disagree_pct = 100 - agree_pct
            
            ax6.pie(
                [agree_pct, disagree_pct],
                labels=['Agreement', 'Disagreement'],
                autopct='%1.1f%%',
                colors=['lightgreen', 'lightcoral'],
                startangle=90
            )
            ax6.set_title('Perk Assignment\nAgreement', fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'No perk data', ha='center', va='center')
            ax6.set_title('Perk Assignment', fontweight='bold')
        
        plt.suptitle(
            'Segmentation Comparison: Manual vs ML',
            fontsize=16, fontweight='bold', y=0.98
        )
        
        if save:
            save_path = os.path.join(self.output_dir, 'segmentation_comparison.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Saved: {save_path}")
        
        plt.show()
        return fig
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        print("\n" + "="*70)
        print("SEGMENTATION COMPARISON REPORT")
        print("="*70 + "\n")

        if self.merged_data is None:
            self.merge_datasets()

        metrics = self.calculate_alignment_metrics()
        overlap = self.analyze_segment_overlap()
        stability = self.calculate_segment_stability()
        cross_tab = self.create_cross_tabulation()

        # NEW: Chi-square test
        chi_square = self.chi_square_test()

        print(f"Total Users Compared: {len(self.merged_data):,}\n")

        print("ALIGNMENT METRICS")
        print("-" * 70)
        for metric, value in metrics.items():
            print(f"  {metric.replace('_', ' ').title()}: {value:.3f}")

        print("\nCHI-SQUARE INDEPENDENCE TEST")
        print("-" * 70)
        print(f"  Chi-square: {chi_square['chi2']:.3f}")
        print(f"  p-value: {chi_square['p_value']:.4f}")
        print(f"  Degrees of freedom: {chi_square['degrees_of_freedom']}")
        print("  Interpretation:", "Dependent (p < 0.05)" if chi_square['p_value'] < 0.05 else "No evidence of dependence (p ≥ 0.05)")

        print("\nSEGMENT STABILITY")
        print("-" * 70)
        print(f"  Average Purity: {stability['average_purity']:.1%}")
        print(f"  Min Purity: {stability['min_purity']:.1%}")
        print(f"  Max Purity: {stability['max_purity']:.1%}")

        print("\nSEGMENT OVERLAP ANALYSIS")
        print("-" * 70)
        for manual_seg, analysis in overlap.items():
            print(f"\n  {manual_seg}:")
            print(f"    Size: {analysis['size']:,}")
            print(f"    → Maps to: {analysis['dominant_ml_segment']}")
            print(f"    → Alignment: {analysis['dominant_percentage']:.1%}")

        print("\nCROSS-TABULATION")
        print("-" * 70)
        print(cross_tab)

        print("\nPERK ASSIGNMENTS")
        print("-" * 70)
        perk_comp = self.compare_perk_assignments()
        if not perk_comp.empty:
            print(perk_comp.to_string(index=False))

        print("\n" + "="*70)

        return {
            'metrics': metrics,
            'overlap': overlap,
            'stability': stability,
            'cross_tab': cross_tab,
            'perk_comparison': perk_comp,
            'chi_square': {
                'chi2': chi_square['chi2'],
                'p_value': chi_square['p_value'],
                'degrees_of_freedom': chi_square['degrees_of_freedom']
            }
        }

    def chi_square_test(self) -> Dict[str, Any]:
        """Run Chi-square test of independence on manual vs ML segmentation."""
        if self.merged_data is None:
            self.merge_datasets()

        # Build contingency table without margins
        contingency = pd.crosstab(
            self.merged_data[self.manual_segment_col],
            self.merged_data[self.ml_segment_col]
        )

        chi2, p, dof, expected = chi2_contingency(contingency)

        result = {
            "chi2": float(chi2),
            "p_value": float(p),
            "degrees_of_freedom": int(dof),
            "expected_freq": pd.DataFrame(
                expected,
                index=contingency.index,
                columns=contingency.columns
            )
        }

        print(f"   Chi-square: {chi2:.3f}, p-value: {p:.4f}, dof: {dof}")
        return result

    
    def get_recommendations(self) -> List[str]:
        """Generate actionable recommendations."""
        if self.merged_data is None:
            self.merge_datasets()
        
        if not self.metrics:
            self.calculate_alignment_metrics()
        
        recommendations = []
        
        ari = self.metrics['adjusted_rand_index']
        nmi = self.metrics['normalized_mutual_info']  # noqa: F841
        
        # Alignment assessment
        if ari < 0.3:
            recommendations.append(
                "⚠️ LOW ALIGNMENT (ARI < 0.3): Manual and ML segmentations "
                "show significant differences. Consider:\n"
                "   - Reviewing feature selection for ML clustering\n"
                "   - Validating manual segmentation rules\n"
                "   - Investigating which segments diverge most"
            )
        elif ari < 0.6:
            recommendations.append(
                "✅ MODERATE ALIGNMENT (0.3 ≤ ARI < 0.6): Some agreement exists. "
                "Recommended actions:\n"
                "   - Analyze segments with low purity\n"
                "   - Consider hybrid approach combining both methods\n"
                "   - A/B test different segmentation strategies"
            )
        else:
            recommendations.append(
                "✅ HIGH ALIGNMENT (ARI ≥ 0.6): Strong agreement between methods. "
                "This validates both approaches.\n"
                "   - Both segmentations can be used confidently\n"
                "   - Consider using ML for scalability\n"
                "   - Manual rules provide good interpretability"
            )
        
        # Balance check
        manual_dist = self.merged_data[self.manual_segment_col].value_counts(normalize=True)
        ml_dist = self.merged_data[self.ml_segment_col].value_counts(normalize=True)
        
        if manual_dist.min() < 0.05:
            recommendations.append(
                "⚠️ UNBALANCED MANUAL SEGMENTS: Some segments < 5% of users. "
                "Consider merging small segments for campaign viability."
            )
        
        if ml_dist.min() < 0.05:
            recommendations.append(
                "⚠️ UNBALANCED ML SEGMENTS: Some segments < 5% of users. "
                "Consider adjusting cluster count or using rebalancing."
            )
        
        # Perk agreement
        if 'manual_perk' in self.merged_data.columns and 'ml_perk' in self.merged_data.columns:
            perk_agreement = (
                self.merged_data['manual_perk'] == self.merged_data['ml_perk']
            ).sum() / len(self.merged_data)
            
            if perk_agreement < 0.5:
                recommendations.append(
                    f"⚠️ LOW PERK AGREEMENT ({perk_agreement:.1%}): Different "
                    "segmentation approaches assign different perks to same users. "
                    "Consider business impact before choosing approach."
                )
        
        return recommendations
    
    def export_results(self, filename: str = 'comparison_results.csv'):
        """Export comparison results to CSV."""
        if self.merged_data is None:
            self.merge_datasets()
        
        export_path = os.path.join(self.output_dir, filename)
        self.merged_data.to_csv(export_path, index=False)
        
        print(f"✅ Exported results to: {export_path}")
        return export_path
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete comparison analysis."""
        print("\n" + "="*70)
        print("RUNNING COMPLETE SEGMENTATION COMPARISON")
        print("="*70)
        
        # Run all analyses
        self.merge_datasets()
        metrics = self.calculate_alignment_metrics()
        overlap = self.analyze_segment_overlap()
        stability = self.calculate_segment_stability()
        features = self.analyze_feature_differences()
        perk_comp = self.compare_perk_assignments()
        
        # Generate visualizations
        self.visualize_comparison(save=True)
        
        # Generate report
        report = self.generate_comprehensive_report()  # noqa: F841
        
        # Get recommendations
        recommendations = self.get_recommendations()
        
        print("\nRECOMMENDATIONS")
        print("="*70)
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")
        
        # Export
        self.export_results()
        
        # NEW: Scatter comparison plots
        viz = ComparisonVisualizer(self.output_dir)
        scatter_paths = viz.generate_all_comparison_plots(self.merged_data)

        # Attach scatter plots to report
        report['scatter_plots'] = scatter_paths

        print("📊 Scatter plots saved:")
        for k, v in scatter_paths.items():
            print(f"   {k}: {v}")
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70)
        
        return {
            'metrics': metrics,
            'overlap': overlap,
            'stability': stability,
            'features': features,
            'perk_comparison': perk_comp,
            'recommendations': recommendations
        }