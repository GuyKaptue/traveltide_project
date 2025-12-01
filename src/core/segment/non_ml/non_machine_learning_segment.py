# core/segment/non_ml/segmentation.py

import os
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from typing import Dict, Any, Tuple
from IPython.display import display  # type: ignore

from src.utils import project_root, get_path, load_yaml
from .metrics import MetricsComputer
from .threshold_manager import ThresholdManager
from .perk_assigner import PerkAssigner
from .data_manager import DataManager
from .analyzer import SegmentationAnalyzer
from .visualizer import SegmentationVisualizer


class NonMachineLearningSegment:
    """
    Customer segmentation orchestrator using rule-based tier hierarchy.
    Coordinates:
    - Metric computation
    - Threshold calculation
    - Perk assignment
    - Analysis and visualization
    - Data export
    """

    REQUIRED_CONFIG_SECTIONS = ["threshold_definitions","perk_groups"]

    def __init__(self, users: pd.DataFrame) -> None:
        if users.empty:
            raise ValueError("Input DataFrame is empty.")

        self.df = users.copy()
        self.thresholds: Dict[str, float] = {}
        self.config: Dict[str, Any] = {}

        # Setup directories
        self.data_output_dir = os.path.join(get_path("processed"), "segment", "non_ml")
        self.fig_output_dir = os.path.join(get_path("reports"), "segment", "non_ml")
        os.makedirs(self.data_output_dir, exist_ok=True)
        os.makedirs(self.fig_output_dir, exist_ok=True)

        # Load configuration
        self.config_path = os.path.join(project_root, "config", "non_ml_config.yaml")
        self._load_configuration()

        # Extract perk groups
        self.perk_groups = self.config["perk_groups"]
        self.perks = [pg["perk"] for pg in self.perk_groups]
        self.groups = [pg["group"] for pg in self.perk_groups]

        # Initialize components
        self.metrics_computer = None
        self.threshold_manager = None
        self.perk_assigner = None
        self.data_manager = None
        self.analyzer = None
        self.visualizer = None

        self._print_initialization_summary()

    # ---------------- Configuration ----------------

    def _load_configuration(self) -> None:
        """Load segmentation configuration from YAML file."""
        try:
            self.config = load_yaml(self.config_path)

            # If YAML has a top-level 'segmentation' key, unwrap it
            if "segmentation" in self.config:
                self.config = self.config["segmentation"]

            print("✅ Configuration loaded successfully")

            # Validate required sections
            for section in self.REQUIRED_CONFIG_SECTIONS:
                if section not in self.config:
                    raise ValueError(f"Missing required configuration section: {section}")

        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            print("⚠️ Using default configuration...")
            self._set_default_config()


    def _set_default_config(self) -> None:
        self.config = {
            "threshold_definitions": {
                "TOTAL_SPEND": {"column": "total_spend", "quantile": 0.8, "fallback": 4000},
                "TRIP_COUNT": {"column": "num_trips", "quantile": 0.8, "fallback": 1.5},
                "BROWSING_RATE": {"column": "browsing_rate", "quantile": 0.8, "fallback": 0.6},
                "HOTEL_SPEND": {"column": "money_spent_hotel_total", "quantile": 0.8, "fallback": 1200},
                "BUSINESS_RATE": {"column": "business_rate", "quantile": 0.8, "fallback": 0.2},
                "GROUP_RATE": {"column": "group_rate", "quantile": 0.8, "fallback": 0.1},
                "AVG_BAGS": {"column": "avg_bags", "quantile": 0.8, "fallback": 1.0},
            },
            "perk_groups": [
                {"perk": "1 night free hotel plus flight", "group": "VIP High-Frequency Spenders"},
                {"perk": "exclusive discounts", "group": "High-Intent Browsers & Spenders"},
                {"perk": "free checked bags", "group": "Group & Family Travelers / Heavy Baggage"},
                {"perk": "free hotel meal", "group": "Hotel & Business Focused Travelers"},
                {"perk": "no cancellation fees", "group": "Baseline Travelers"},
            ],
        }

    def _print_initialization_summary(self) -> None:
        print("✅ NonMachineLearningSegment initialized")
        print(f"   - Total users: {len(self.df):,}")
        print(f"   - Data output: {self.data_output_dir}")
        print(f"   - Fig output: {self.fig_output_dir}")
        print(f"   - Config: {self.config_path}")

    # ---------------- Pipeline ----------------

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:
        """Execute complete segmentation pipeline."""
        print("\n" + "=" * 80)
        print("🚀 CUSTOMER SEGMENTATION PIPELINE")
        print("=" * 80 + "\n")

        self.compute_intermediate_metrics()
        self.compute_thresholds()
        perk_distribution = self.assign_perks()
        self.save_customer_segmentation()
        analysis_df = self.analyze_perk_assignments()
        self.data_manager.save_analysis_results(analysis_df)
        decision_tree = self.create_visualizations()

        self._print_final_summary(perk_distribution)
        return self.df, perk_distribution, decision_tree

    def _print_final_summary(self, perk_distribution: pd.DataFrame) -> None:
        print("\n" + "=" * 80)
        print("🎉 PIPELINE COMPLETE!")
        print("=" * 80)
        for _, row in perk_distribution.iterrows():
            print(f"   - {row['assigned_group']}: {row['Count']:,} users ({row['Percentage']:.1f}%)")
        print("=" * 80)

    # ---------------- Convenience ----------------

    def compute_intermediate_metrics(self) -> None:
        self.metrics_computer = MetricsComputer(self.df)
        self.df = self.metrics_computer.compute_all_metrics()

    def compute_thresholds(self) -> None:
        self.threshold_manager = ThresholdManager(self.df, self.config)
        self.thresholds = self.threshold_manager.compute_thresholds()

    def assign_perks(self) -> pd.DataFrame:
        self.perk_assigner = PerkAssigner(self.df, self.thresholds, self.config)
        distribution = self.perk_assigner.assign_perks()
        self.df = self.perk_assigner.get_dataframe()
        return distribution

    def save_customer_segmentation(self) -> str:
        self.data_manager = DataManager(self.data_output_dir)
        return self.data_manager.save_all_results(self.df, self.thresholds, self.config)

    def analyze_perk_assignments(self) -> pd.DataFrame:
        self.analyzer = SegmentationAnalyzer(self.df, self.thresholds, self.config)
        return self.analyzer.analyze_perk_assignments()

    def create_visualizations(self) -> plt.Figure:
        self.visualizer = SegmentationVisualizer(self.fig_output_dir, self.config)
        summary_df = self.analyzer.create_segment_summary()
        self.visualizer.plot_segment_summary_table(summary_df)
        self.visualizer.plot_perk_distribution(self.df)
        self.visualizer.plot_threshold_coverage(self.df, self.thresholds)
        return self.visualizer.draw_decision_tree(self.thresholds)

    # ---------------- Usage Demos ----------------

    def run_quickstart(self, users_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:
        """Demonstrate quickstart usage with full pipeline and step-by-step execution."""
        segmenter = NonMachineLearningSegment(users_df)
        segmented_df, distribution, decision_tree = segmenter.run()
        return segmented_df, distribution, decision_tree

    def advanced_usage(self, users_df: pd.DataFrame) -> None:
        """Demonstrate advanced usage patterns (threshold overrides, filtering, exports)."""
        segmenter = NonMachineLearningSegment(users_df)
        segmenter.compute_intermediate_metrics()
        segmenter.compute_thresholds()

        # Example threshold override
        print(f"Original TOTAL_SPEND threshold: {segmenter.thresholds['TOTAL_SPEND']}")
        segmenter.thresholds["TOTAL_SPEND"] = 5000
        print(f"Adjusted TOTAL_SPEND threshold: {segmenter.thresholds['TOTAL_SPEND']}")

        distribution = segmenter.assign_perks()
        print("\nPerk distribution summary:")
        display(distribution.to_string(index=False)) 

        # Filtering examples
        vip_users = segmenter.df[segmenter.df["assigned_group"] == "VIP High-Frequency Spenders"]
        print(f"VIP Users: {len(vip_users)} | Avg Spend: {vip_users['total_spend'].mean():.2f}")

        baseline_users = segmenter.df[segmenter.df["assigned_perk"] == "no cancellation fees"]
        print(f"Baseline Users: {len(baseline_users)}")

        # Export segments for campaigns
        print("\n[ADVANCED] Export for email campaigns...")
        for pg in segmenter.perk_groups:
            perk = pg["perk"]
            seg_df = segmenter.df[segmenter.df["assigned_perk"] == perk]
            if {"user_id", "email"}.issubset(seg_df.columns):
                export_df = seg_df[["user_id", "email", "assigned_perk"]]
                filename = perk.replace(" ", "_") + "_users.csv"
                # Uncomment to actually save:
                export_df.to_csv(os.path.join(self.data_output_dir, filename), index=False)
                print(f"   - {filename}: {len(export_df)} users")

        print("\n[ADVANCED] Usage demonstration complete.")
