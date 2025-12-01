# core/segment/non_ml/visualizer.py

import os
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch # type: ignore
from typing import Dict, Any, Optional  # noqa: F401
from IPython.display import display # type: ignore

try:
    import plotly.graph_objects as go # type: ignore
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class SegmentationVisualizer:
    """
    Handles all visualization tasks for customer segmentation.
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
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectory for HTML files
        self.html_dir = os.path.join(output_dir, "html")
        os.makedirs(self.html_dir, exist_ok=True)
    
    def plot_segment_summary_table(self, summary_df: pd.DataFrame) -> None:
        """
        Create interactive segment summary table.
        
        Parameters
        ----------
        summary_df : pd.DataFrame
            Summary dataframe with segment statistics
        """
        if not PLOTLY_AVAILABLE:
            self._plot_text_summary(summary_df)
            return
        
        # Zebra row colors
        row_colors = [
            '#ffffff' if i % 2 == 0 else '#f9f9f9' 
            for i in range(len(summary_df))
        ]
        
        # Create Plotly table
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
            title="Segment Summary with Perks",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        # Save HTML
        html_path = os.path.join(self.html_dir, "segment_summary_table.html")
        fig.write_html(html_path, include_plotlyjs="cdn")
        print(f"📊 HTML saved: {html_path}")
        
        # Try to save PNG
        try:
            png_path = os.path.join(self.output_dir, "segment_summary_table.png")
            fig.write_image(png_path)
            print(f"🖼 PNG saved: {png_path}")
        except Exception as e:
            print(f"⚠ PNG export failed: {e}")
            print("Install kaleido using: pip install -U kaleido")
        
        fig.show()
    
    def _plot_text_summary(self, summary_df: pd.DataFrame) -> None:
        """Fallback text-based summary when plotly unavailable"""
        print("\n" + "=" * 80)
        print("SEGMENT SUMMARY")
        print("=" * 80)
        display(summary_df.to_string(index=False))
        print("⚠️ Plotly not installed. Install with: pip install plotly")
    
    def plot_perk_distribution(self, df: pd.DataFrame) -> plt.Figure:
        """
        Create matplotlib visualization of perk distribution.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with assigned perks
            
        Returns
        -------
        plt.Figure
            Figure object
        """
        print("[VISUALIZATION] Creating perk distribution plot...")
        
        perk_counts = df['assigned_perk'].value_counts()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Bar plot
        colors = ['gold', 'lightcoral', 'lightgreen', 'plum', 'lightgray']
        bars = ax1.bar(
            range(len(perk_counts)), 
            perk_counts.values, 
            color=colors
        )
        ax1.set_xlabel('Perk Type', fontsize=12)
        ax1.set_ylabel('Number of Users', fontsize=12)
        ax1.set_title(
            'Customer Segmentation: Perk Distribution', 
            fontsize=14, 
            fontweight='bold'
        )
        ax1.set_xticks(range(len(perk_counts)))
        ax1.set_xticklabels(
            [p.replace(' ', '\n') for p in perk_counts.index], 
            rotation=45, 
            ha='right'
        )
        
        # Add value labels
        for bar, count in zip(bars, perk_counts.values):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2., 
                height + 5,
                f'{count:,}', 
                ha='center', 
                va='bottom', 
                fontweight='bold'
            )
        
        # Pie chart
        ax2.pie(
            perk_counts.values, 
            labels=perk_counts.index, 
            autopct='%1.1f%%',
            colors=colors, 
            startangle=90
        )
        ax2.set_title(
            'Perk Distribution Percentage', 
            fontsize=14, 
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "perk_distribution_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {plot_path}")
        
        plt.show()
        return fig
    
    def plot_threshold_coverage(
        self, 
        df: pd.DataFrame, 
        thresholds: Dict[str, float]
    ) -> plt.Figure:
        """
        Visualize how many users meet each threshold.
        
        Parameters
        ----------
        df : pd.DataFrame
            User dataframe
        thresholds : Dict[str, float]
            Threshold values
            
        Returns
        -------
        plt.Figure
            Figure object
        """
        print("[VISUALIZATION] Creating threshold coverage plot...")
        
        threshold_defs = self.config['threshold_definitions']
        n_thresholds = len(threshold_defs)
        
        fig, axes = plt.subplots(
            2, 4, 
            figsize=(16, 8)
        )
        axes = axes.flatten()
        
        for idx, (name, config) in enumerate(threshold_defs.items()):
            if idx >= len(axes):
                break
            
            column = config['column']
            threshold = thresholds[name]
            
            if column in df.columns:
                ax = axes[idx]
                
                # Plot histogram
                df[column].hist(
                    bins=50, 
                    ax=ax, 
                    alpha=0.7, 
                    edgecolor='black'
                )
                
                # Add threshold line
                ax.axvline(
                    threshold, 
                    color='red', 
                    linestyle='--', 
                    linewidth=2,
                    label=f'Threshold: {threshold:.2f}'
                )
                
                # Calculate percentage above threshold
                pct_above = (df[column] >= threshold).sum() / len(df) * 100
                
                ax.set_title(f'{name}\n{pct_above:.1f}% above threshold')
                ax.set_xlabel(column)
                ax.legend()
        
        # Hide unused subplots
        for idx in range(n_thresholds, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, "threshold_coverage.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path}")
        
        fig.show()
        return fig
    
    def draw_decision_tree(
        self, 
        thresholds: Dict[str, float]
    ) -> plt.Figure:
        """
        Create modern decision tree visualization.
        
        Parameters
        ----------
        thresholds : Dict[str, float]
            Threshold values
            
        Returns
        -------
        plt.Figure
            Figure object
        """
        print("[VISUALIZATION] Creating decision tree...")
        
        perk_groups = self.config['perk_groups']
        T = thresholds
        
        fig, ax = plt.subplots(figsize=(18, 12), facecolor='white')
        ax.axis('off')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        COLORS = {
            'start': '#6366F1',
            'decision': '#F59E0B',
            'success': '#10B981',
            'fallback': '#8B5CF6',
            'text_dark': '#1F2937',
            'text_light': '#FFFFFF'
        }
        
        # Define nodes
        nodes = self._create_decision_nodes(perk_groups, T)
        
        # Draw nodes
        for node in nodes.values():
            self._draw_node(ax, node, COLORS)
        
        # Draw arrows
        self._draw_decision_arrows(ax, nodes)
        
        # Title
        ax.text(
            5, 9.8, 
            'TravelTide Perk Assignment Decision Tree',
            fontsize=16, 
            fontweight='bold', 
            ha='center'
        )
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'decision_tree.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path}")
        
        fig.show()
        return fig
    
    def _create_decision_nodes(
        self, 
        perk_groups: list, 
        T: Dict[str, float]
    ) -> Dict:
        """Create node definitions for decision tree"""
        
        class Node:
            def __init__(self, x, y, w, h, text, node_type):
                self.x, self.y, self.w, self.h = x, y, w, h
                self.text, self.node_type = text, node_type
        
        return {
            'start': Node(
                5, 9, 2.5, 0.8,
                'START\nUser Profile Analysis',
                'start'
            ),
            'tier1': Node(
                5, 7.5, 2.5, 0.8,
                f"Spend ≥ ${T['TOTAL_SPEND']:.0f}\nTrips ≥ {T['TRIP_COUNT']:.0f}",
                'decision'
            ),
            'tier1_perk': Node(
                2, 6, 3, 0.8,
                f"✈️ {perk_groups[0]['perk']}\n{perk_groups[0]['group']}",
                'success'
            ),
            'tier2': Node(
                7, 6, 2.5, 0.8,
                f"Browsing ≥ {T['BROWSING_RATE']:.2f}",
                'decision'
            ),
            'tier2_perk': Node(
                2, 4.5, 3, 0.8,
                f"🛍️ {perk_groups[1]['perk']}\n{perk_groups[1]['group']}",
                'success'
            ),
            'tier3': Node(
                7, 4.5, 2.5, 0.8,
                f"Bags ≥ {T['AVG_BAGS']:.1f} OR\nGroup ≥ {T['GROUP_RATE']:.2f}",
                'decision'
            ),
            'tier3_perk': Node(
                2, 3, 3, 0.8,
                f"🎒 {perk_groups[2]['perk']}\n{perk_groups[2]['group']}",
                'success'
            ),
            'tier4': Node(
                7, 3, 2.5, 0.8,
                f"Hotel ≥ ${T['HOTEL_SPEND']:.0f} OR\nBusiness ≥ {T['BUSINESS_RATE']:.2f}",
                'decision'
            ),
            'tier4_perk': Node(
                2, 1.5, 3, 0.8,
                f"🏨 {perk_groups[3]['perk']}\n{perk_groups[3]['group']}",
                'success'
            ),
            'baseline': Node(
                7, 1.5, 3, 0.8,
                f"🛡️ {perk_groups[4]['perk']}\n{perk_groups[4]['group']}",
                'fallback'
            )
        }
    
    def _draw_node(self, ax, node, colors):
        """Draw a single node"""
        if node.node_type == 'start':
            color, edge, text_color = colors['start'], '#4F46E5', colors['text_light']
        elif node.node_type == 'decision':
            color, edge, text_color = colors['decision'], '#D97706', colors['text_dark']
        elif node.node_type == 'success':
            color, edge, text_color = colors['success'], '#059669', colors['text_light']
        else:
            color, edge, text_color = colors['fallback'], '#7C3AED', colors['text_light']
        
        box = FancyBboxPatch(
            (node.x - node.w/2, node.y - node.h/2),
            node.w, node.h,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor=edge,
            linewidth=2.5,
            zorder=2
        )
        ax.add_patch(box)
        ax.text(
            node.x, node.y, node.text,
            fontsize=10,
            fontweight='bold',
            ha='center',
            va='center',
            color=text_color
        )
    
    def _draw_decision_arrows(self, ax, nodes):
        """Draw arrows connecting nodes"""
        def arrow(n1, n2, label=''):
            arr = FancyArrowPatch(
                (n1.x, n1.y-0.4),
                (n2.x, n2.y+0.4),
                arrowstyle='->',
                linewidth=2,
                color='#6B7280',
                alpha=0.7
            )
            ax.add_patch(arr)
            if label:
                ax.text(
                    (n1.x+n2.x)/2,
                    (n1.y+n2.y)/2,
                    label,
                    fontsize=9,
                    fontweight='bold',
                    ha='center',
                    color='#10B981' if label=='YES' else '#EF4444'
                )
        
        # Connect flow
        arrow(nodes['start'], nodes['tier1'])
        arrow(nodes['tier1'], nodes['tier1_perk'], 'YES')
        arrow(nodes['tier1'], nodes['tier2'], 'NO')
        arrow(nodes['tier2'], nodes['tier2_perk'], 'YES')
        arrow(nodes['tier2'], nodes['tier3'], 'NO')
        arrow(nodes['tier3'], nodes['tier3_perk'], 'YES')
        arrow(nodes['tier3'], nodes['tier4'], 'NO')
        arrow(nodes['tier4'], nodes['tier4_perk'], 'YES')
        arrow(nodes['tier4'], nodes['baseline'], 'NO')
    
    # def draw_horizontal_decision_tree(self, thresholds: Dict[str, float]) -> plt.Figure:
    #     """
    #     Horizontal layout decision tree visualization.
    #     """
    #     perk_groups = self.config['perk_groups']
    #     T = thresholds

    #     fig, ax = plt.subplots(figsize=(20, 10), facecolor='white')
    #     ax.axis('off')
    #     ax.set_xlim(0, 14)
    #     ax.set_ylim(0, 10)

    #     # Define nodes (left to right flow)
    #     nodes = {
    #         'start': self._create_node(1, 5, 2.5, 1.2, 'START\nUser Analysis', 'start'),
    #         'tier1': self._create_node(3.5, 8, 2.5, 1,
    #                                 f"Spend ≥ {T['TOTAL_SPEND']:.0f}\nTrips ≥ {T['TRIP_COUNT']:.0f}", 'decision'),
    #         'tier1_perk': self._create_node(6, 8, 3, 1,
    #                                         f"✈️ {perk_groups[0]['perk']}\n{perk_groups[0]['group']}", 'success'),
    #         'tier2': self._create_node(3.5, 6, 2.5, 1,
    #                                 f"Browsing ≥ {T['BROWSING_RATE']:.2f}", 'decision'),
    #         'tier2_perk': self._create_node(6, 6, 3, 1,
    #                                         f"🛍️ {perk_groups[4]['perk']}\n{perk_groups[4]['group']}", 'fallback'),
    #         'tier3': self._create_node(3.5, 4, 2.5, 1,
    #                                 f"Bags ≥ {T['AVG_BAGS']:.1f}\nGroup ≥ {T['GROUP_RATE']:.2f}", 'decision'),
    #         'tier3_perk': self._create_node(6, 4, 3, 1,
    #                                         f"🎒 {perk_groups[1]['perk']}\n{perk_groups[1]['group']}", 'success'),
    #         'tier4': self._create_node(3.5, 2, 2.5, 1,
    #                                 f"Hotel ≥ {T['HOTEL_SPEND']:.0f}\nBusiness ≥ {T['BUSINESS_RATE']:.2f}", 'decision'),
    #         'tier4_perk': self._create_node(6, 2, 3, 1,
    #                                         f"🏨 {perk_groups[2]['perk']}\n{perk_groups[2]['group']}", 'success'),
    #         'baseline': self._create_node(9, 2, 3, 1,
    #                                     f"🛡️ {perk_groups[3]['perk']}\n{perk_groups[3]['group']}", 'success')
    #     }

    #     # Draw nodes
    #     for node in nodes.values():
    #         self._draw_node(ax, node, {
    #             'start': '#6366F1',
    #             'decision': '#F59E0B',
    #             'success': '#10B981',
    #             'fallback': '#8B5CF6',
    #             'text_dark': '#1F2937',
    #             'text_light': '#FFFFFF'
    #         })

    #     # Draw arrows
    #     def arrow(n1, n2, label=''):
    #         arr = FancyArrowPatch(
    #             (n1.x + n1.w/2, n1.y),
    #             (n2.x - n2.w/2, n2.y),
    #             arrowstyle='->',
    #             linewidth=2,
    #             color='#6B7280',
    #             alpha=0.7
    #         )
    #         ax.add_patch(arr)
    #         if label:
    #             ax.text((n1.x+n2.x)/2, (n1.y+n2.y)/2 + 0.3, label,
    #                     fontsize=9, fontweight='bold',
    #                     ha='center', color='#10B981' if label == 'YES' else '#EF4444')

    #     arrow(nodes['start'], nodes['tier1'])
    #     arrow(nodes['tier1'], nodes['tier1_perk'], 'YES')
    #     arrow(nodes['tier1'], nodes['tier2'], 'NO')
    #     arrow(nodes['tier2'], nodes['tier2_perk'], 'YES')
    #     arrow(nodes['tier2'], nodes['tier3'], 'NO')
    #     arrow(nodes['tier3'], nodes['tier3_perk'], 'YES')
    #     arrow(nodes['tier3'], nodes['tier4'], 'NO')
    #     arrow(nodes['tier4'], nodes['tier4_perk'], 'YES')
    #     arrow(nodes['tier4'], nodes['baseline'], 'NO')

    #     ax.text(7, 9.5, 'TravelTide Perk Assignment Decision Tree (Horizontal Layout)',
    #             fontsize=16, fontweight='bold', ha='center')

    #     plt.tight_layout()
    #     save_path = os.path.join(self.output_dir, 'decision_tree_horizontal.png')
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     print(f"   ✅ Saved: {save_path}")
    #     plt.show()
    #     return fig
