# core/segment/comparison/comparison_viz.py

import os
import matplotlib.pyplot as plt # type: ignore



class ComparisonVisualizer:
    """
    Creates professional scatter plots comparing manual vs ML segmentation metrics.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ============================================================
    # Internal helper: professional scatter plot
    # ============================================================
    def _create_scatter_plot(self, df, x_col, y_col, title, filename):
        plt.figure(figsize=(7, 6))
        plt.scatter(
            df[x_col],
            df[y_col],
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
            s=60
        )

        # Diagonal reference line
        min_val = min(df[x_col].min(), df[y_col].min())
        max_val = max(df[x_col].max(), df[y_col].max())
        plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')

        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel(x_col.replace("_", " ").title())
        plt.ylabel(y_col.replace("_", " ").title())
        plt.grid(alpha=0.3)

        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

        return save_path

    # ============================================================
    # Individual Plots
    # ============================================================

    def plot_total_spend(self, df):
        return self._create_scatter_plot(
            df,
            "total_spend_manual",
            "total_spend_ml",
            "Total Spend — Manual vs. ML",
            "plot_total_spend_manual_total_spend_ml.png"
        )

    def plot_num_trips(self, df):
        return self._create_scatter_plot(
            df,
            "num_trips_manual",
            "num_trips_ml",
            "Number of Trips — Manual vs. ML",
            "plot_num_trips_manual_num_trips_ml.png"
        )

    def plot_conversion_rate(self, df):
        return self._create_scatter_plot(
            df,
            "conversion_rate_manual",
            "conversion_rate_ml",
            "Conversion Rate — Manual vs. ML",
            "plot_conversion_rate_manual_conversion_rate_ml.png"
        )

    def plot_rfm_score(self, df):
        return self._create_scatter_plot(
            df,
            "RFM_score_manual",
            "RFM_score_ml",
            "RFM Score — Manual vs. ML",
            "plot_RFM_score_manual_RFM_score_ml.png"
        )

    def plot_browsing_rate(self, df):
        return self._create_scatter_plot(
            df,
            "browsing_rate_manual",
            "browsing_rate_ml",
            "Browsing Rate — Manual vs. ML",
            "plot_browsing_rate_manual_browsing_rate_ml.png"
        )

    def plot_business_rate(self, df):
        return self._create_scatter_plot(
            df,
            "business_rate_manual",
            "business_rate_ml",
            "Business Rate — Manual vs. ML",
            "plot_business_rate_manual_business_rate_ml.png"
        )

    # ============================================================
    # Master function: run all visual comparisons
    # ============================================================
    def generate_all_comparison_plots(self, df):
        """
        Generates all 6 scatter plots and returns file paths.
        """
        print("📊 Generating all comparison scatter plots...")

        results = {
            "total_spend": self.plot_total_spend(df),
            "num_trips": self.plot_num_trips(df),
            "conversion_rate": self.plot_conversion_rate(df),
            "rfm_score": self.plot_rfm_score(df),
            "browsing_rate": self.plot_browsing_rate(df),
            "business_rate": self.plot_business_rate(df),
        }

        print("✅ All comparison plots generated.")
        return results
