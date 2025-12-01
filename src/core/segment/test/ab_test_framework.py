# core/segment/test/ab_test_framework.py
import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from typing import Dict, Optional, Any
from scipy.stats import chi2_contingency, fisher_exact, ttest_ind, mannwhitneyu, ks_2samp, anderson_ksamp # type: ignore  # noqa: F401

from src.utils import get_path  

import warnings
warnings.filterwarnings('ignore')


class ABTestFramework:
    """
    A/B Testing framework for comparing perk assignment strategies.
    Compares:
    - Group A: Manual (rule-based) perk assignment
    - Group B: ML (K-Means) perk assignment
    - Group C: Random perk assignment (control)
    Tests:
    - Subscription rate (primary metric)
    - Spending increase (secondary metric)
    - User engagement (tertiary metric)
    """

    def __init__(
        self,
        manual_segmentation: pd.DataFrame,
        ml_segmentation: pd.DataFrame,
        test_ratio: float = 0.33,
        random_state: int = 42,
        output_dir: Optional[str] = None
    ):
        """
        Initialize A/B test framework.
        Parameters
        ----------
        manual_segmentation : pd.DataFrame
            Manual segmentation with assigned perks
        ml_segmentation : pd.DataFrame
            ML segmentation with assigned perks
        test_ratio : float
            Proportion of users in each test group (default: 33% each)
        random_state : int
            Random seed for reproducibility
        output_dir : str, optional
            Directory for saving outputs
        """
        self.manual_seg = manual_segmentation.copy()
        self.ml_seg = ml_segmentation.copy()
        self.test_ratio = test_ratio
        self.random_state = random_state
        # Test groups
        self.test_groups = None
        # Results storage
        self.test_results = {}
        self.statistical_tests = {}
        # Available perks
        self.available_perks = [
            "1 night free hotel plus flight",
            "free hotel meal",
            "free checked bags",
            "exclusive discounts",
            "no cancellation fees"
        ]
        # Output directory
        self.output_dir = output_dir or os.path.join(
            get_path('reports'), 'segment', 'ab_test'
        )
        self.data_path = get_path('ab_test')

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        print("✅ ABTestFramework initialized")
        print(f"   Manual segmentation: {len(self.manual_seg):,} users")
        print(f"   ML segmentation: {len(self.ml_seg):,} users")
        print(f"   Test ratio: {self.test_ratio:.0%} per group")

    # ========================================================================
    # TEST SETUP
    # ========================================================================

    def create_test_groups(self) -> pd.DataFrame:
        """
        Create three test groups: Manual, ML, and Random.
        Returns
        -------
        pd.DataFrame
            Combined dataset with test group assignments
        """
        print("[SETUP] Creating test groups...")
        # Merge datasets to get common users
        merged = pd.merge(
            self.manual_seg[['user_id', 'assigned_perk']],
            self.ml_seg[['user_id', 'assigned_perk']],
            on='user_id',
            how='inner',
            suffixes=('_manual', '_ml')
        )
        print(f"   Common users: {len(merged):,}")
        # Shuffle users
        np.random.seed(self.random_state)
        shuffled = merged.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        # Split into three groups
        n = len(shuffled)
        group_size = int(n * self.test_ratio)
        # Group A: Manual perks
        group_a = shuffled.iloc[:group_size].copy()
        group_a['test_group'] = 'A_Manual'
        group_a['assigned_perk'] = group_a['assigned_perk_manual']
        # Group B: ML perks
        group_b = shuffled.iloc[group_size:2*group_size].copy()
        group_b['test_group'] = 'B_ML'
        group_b['assigned_perk'] = group_b['assigned_perk_ml']
        # Group C: Random perks
        group_c = shuffled.iloc[2*group_size:3*group_size].copy()
        group_c['test_group'] = 'C_Random'
        group_c['assigned_perk'] = np.random.choice(
            self.available_perks,
            size=len(group_c),
            replace=True
        )
        # Combine groups
        self.test_groups = pd.concat([group_a, group_b, group_c], ignore_index=True)
        # Add metadata
        self.test_groups['test_start_date'] = pd.Timestamp.now()
        print(f"\n   ✅ Test groups created:")  # noqa: F541
        print(f"      Group A (Manual): {len(group_a):,} users")
        print(f"      Group B (ML): {len(group_b):,} users")
        print(f"      Group C (Random): {len(group_c):,} users")
        return self.test_groups

    def simulate_outcomes(
        self,
        base_subscription_rate: float = 0.15,
        manual_lift: float = 0.05,
        ml_lift: float = 0.08,
        noise: float = 0.02
    ) -> pd.DataFrame:
        """
        Simulate subscription outcomes for testing.
        In production, replace this with actual user behavior data.
        Parameters
        ----------
        base_subscription_rate : float
            Baseline subscription rate for random assignment
        manual_lift : float
            Additional lift from manual assignment
        ml_lift : float
            Additional lift from ML assignment
        noise : float
            Random noise in outcomes
        Returns
        -------
        pd.DataFrame
            Test groups with simulated outcomes
        """
        print("[SIMULATION] Generating outcome data...")
        print("   ⚠️ This is simulated data for demonstration")
        print("   ⚠️ Replace with actual user behavior in production")
        if self.test_groups is None:
            self.create_test_groups()
        np.random.seed(self.random_state)
        # Simulate subscription outcomes
        subscriptions = []
        for _, row in self.test_groups.iterrows():
            # Base rate + lift based on group + noise
            if row['test_group'] == 'A_Manual':
                prob = base_subscription_rate + manual_lift
            elif row['test_group'] == 'B_ML':
                prob = base_subscription_rate + ml_lift
            else:  # Group C: Random
                prob = base_subscription_rate
            # Add noise
            prob += np.random.uniform(-noise, noise)
            # Ensure probability is between 0 and 1
            prob = max(0, min(1, prob))
            # Simulate subscription
            subscriptions.append(np.random.binomial(1, prob))
        self.test_groups['subscribed'] = subscriptions
        # Simulate spending increase
        self.test_groups['spending_increase'] = np.where(
            self.test_groups['subscribed'] == 1,
            np.random.normal(1.2, 0.3, len(self.test_groups)),
            np.random.normal(1.0, 0.1, len(self.test_groups))
        )
        # Simulate user engagement
        self.test_groups['engagement_score'] = np.where(
            self.test_groups['subscribed'] == 1,
            np.random.normal(8, 1, len(self.test_groups)),
            np.random.normal(5, 1, len(self.test_groups))
        )
        print(f"   ✅ Simulated outcomes added to test groups")  # noqa: F541
        return self.test_groups

    # ========================================================================
    # STATISTICAL TESTS
    # ========================================================================

    def run_chi_square_test(self) -> Dict[str, Any]:
        """
        Run Chi-Square test on subscription rates.
        Returns
        -------
        Dict[str, Any]
            Chi-Square test results
        """
        if self.test_groups is None:
            self.create_test_groups()
            self.simulate_outcomes()
        print("[TEST] Running Chi-Square test on subscription rates...")
        # Create contingency table
        contingency = pd.crosstab(
            self.test_groups['test_group'],
            self.test_groups['subscribed']
        )
        # Run Chi-Square test
        chi2, p, dof, expected = chi2_contingency(contingency)
        self.statistical_tests['chi_square'] = {
            'chi2': chi2,
            'p_value': p,
            'degrees_of_freedom': dof,
            'contingency_table': contingency,
            'expected_freq': expected
        }
        print(f"   ✅ Chi-Square test completed")  # noqa: F541
        print(f"      Chi2: {chi2:.3f}, p-value: {p:.4f}")
        print(f"      Interpretation: {'Significant difference' if p < 0.05 else 'No significant difference'}")
        return self.statistical_tests['chi_square']

    def run_fisher_exact_test(self) -> Dict[str, Any]:
        """
        Run Fisher's Exact test on subscription rates.
        Returns
        -------
        Dict[str, Any]
            Fisher's Exact test results
        """
        if self.test_groups is None:
            self.create_test_groups()
            self.simulate_outcomes()
        print("[TEST] Running Fisher's Exact test on subscription rates...")
        # Create contingency table
        contingency = pd.crosstab(
            self.test_groups['test_group'],
            self.test_groups['subscribed']
        )
        # Run Fisher's Exact test
        odds_ratio, p_value = fisher_exact(contingency)

        self.statistical_tests['fisher_exact'] = {
            'odds_ratio': odds_ratio,
            'p_value': p_value,
            'contingency_table': contingency
        }
        print(f"   ✅ Fisher's Exact test completed")  # noqa: F541
        print(f"      Odds Ratio: {odds_ratio:.3f}, p-value: {p_value:.4f}")
        print(f"      Interpretation: {'Significant difference' if p_value < 0.05 else 'No significant difference'}")
        return self.statistical_tests['fisher_exact']

    def run_t_tests(self) -> Dict[str, Dict[str, Any]]:
        """
        Run t-tests on spending increase and engagement score.
        Returns
        -------
        Dict[str, Dict[str, Any]]
            T-test results for each metric
        """
        if self.test_groups is None:
            self.create_test_groups()
            self.simulate_outcomes()
        print("[TEST] Running t-tests on spending and engagement...")
        metrics = ['spending_increase', 'engagement_score']
        t_test_results = {}
        for metric in metrics:
            # Group A vs Group B
            a_data = self.test_groups[self.test_groups['test_group'] == 'A_Manual'][metric]
            b_data = self.test_groups[self.test_groups['test_group'] == 'B_ML'][metric]
            t_stat, p_val = ttest_ind(a_data, b_data, equal_var=False)
            t_test_results[f'{metric}_A_vs_B'] = {
                't_stat': t_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
            # Group A vs Group C
            c_data = self.test_groups[self.test_groups['test_group'] == 'C_Random'][metric]
            t_stat, p_val = ttest_ind(a_data, c_data, equal_var=False)
            t_test_results[f'{metric}_A_vs_C'] = {
                't_stat': t_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
            # Group B vs Group C
            t_stat, p_val = ttest_ind(b_data, c_data, equal_var=False)
            t_test_results[f'{metric}_B_vs_C'] = {
                't_stat': t_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
        self.statistical_tests['t_tests'] = t_test_results
        print(f"   ✅ T-tests completed")  # noqa: F541
        for test, result in t_test_results.items():
            print(f"      {test}: t={result['t_stat']:.3f}, p={result['p_value']:.4f}")
        return self.statistical_tests['t_tests']

    def run_mann_whitney_u_test(self) -> Dict[str, Dict[str, Any]]:
        """
        Run Mann-Whitney U test on spending increase and engagement score.
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Mann-Whitney U test results for each metric
        """
        if self.test_groups is None:
            self.create_test_groups()
            self.simulate_outcomes()
        print("[TEST] Running Mann-Whitney U tests on spending and engagement...")
        metrics = ['spending_increase', 'engagement_score']
        u_test_results = {}
        for metric in metrics:
            # Group A vs Group B
            a_data = self.test_groups[self.test_groups['test_group'] == 'A_Manual'][metric]
            b_data = self.test_groups[self.test_groups['test_group'] == 'B_ML'][metric]
            u_stat, p_val = mannwhitneyu(a_data, b_data)
            u_test_results[f'{metric}_A_vs_B'] = {
                'u_stat': u_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
            # Group A vs Group C
            c_data = self.test_groups[self.test_groups['test_group'] == 'C_Random'][metric]
            u_stat, p_val = mannwhitneyu(a_data, c_data)
            u_test_results[f'{metric}_A_vs_C'] = {
                'u_stat': u_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
            # Group B vs Group C
            u_stat, p_val = mannwhitneyu(b_data, c_data)
            u_test_results[f'{metric}_B_vs_C'] = {
                'u_stat': u_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
        self.statistical_tests['mann_whitney_u'] = u_test_results
        print(f"   ✅ Mann-Whitney U tests completed")  # noqa: F541
        for test, result in u_test_results.items():
            print(f"      {test}: U={result['u_stat']:.3f}, p={result['p_value']:.4f}")
        return self.statistical_tests['mann_whitney_u']

    def run_ks_test(self) -> Dict[str, Dict[str, Any]]:
        """
        Run Kolmogorov-Smirnov test on spending increase and engagement score.
        Returns
        -------
        Dict[str, Dict[str, Any]]
            KS test results for each metric
        """
        if self.test_groups is None:
            self.create_test_groups()
            self.simulate_outcomes()
        print("[TEST] Running Kolmogorov-Smirnov tests on spending and engagement...")
        metrics = ['spending_increase', 'engagement_score']
        ks_test_results = {}
        for metric in metrics:
            # Group A vs Group B
            a_data = self.test_groups[self.test_groups['test_group'] == 'A_Manual'][metric]
            b_data = self.test_groups[self.test_groups['test_group'] == 'B_ML'][metric]
            ks_stat, p_val = ks_2samp(a_data, b_data)
            ks_test_results[f'{metric}_A_vs_B'] = {
                'ks_stat': ks_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
            # Group A vs Group C
            c_data = self.test_groups[self.test_groups['test_group'] == 'C_Random'][metric]
            ks_stat, p_val = ks_2samp(a_data, c_data)
            ks_test_results[f'{metric}_A_vs_C'] = {
                'ks_stat': ks_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
            # Group B vs Group C
            ks_stat, p_val = ks_2samp(b_data, c_data)
            ks_test_results[f'{metric}_B_vs_C'] = {
                'ks_stat': ks_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
        self.statistical_tests['ks_test'] = ks_test_results
        print(f"   ✅ Kolmogorov-Smirnov tests completed")  # noqa: F541
        for test, result in ks_test_results.items():
            print(f"      {test}: KS={result['ks_stat']:.3f}, p={result['p_value']:.4f}")
        return self.statistical_tests['ks_test']

    def run_anderson_ksamp_test(self) -> Dict[str, Dict[str, Any]]:
        """
        Run Anderson-Darling test on spending increase and engagement score.
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Anderson-Darling test results for each metric
        """
        if self.test_groups is None:
            self.create_test_groups()
            self.simulate_outcomes()
        print("[TEST] Running Anderson-Darling tests on spending and engagement...")
        metrics = ['spending_increase', 'engagement_score']
        anderson_test_results = {}
        for metric in metrics:
            # Group A vs Group B vs Group C
            a_data = self.test_groups[self.test_groups['test_group'] == 'A_Manual'][metric]
            b_data = self.test_groups[self.test_groups['test_group'] == 'B_ML'][metric]
            c_data = self.test_groups[self.test_groups['test_group'] == 'C_Random'][metric]
            stat, crit, sig_level = anderson_ksamp([a_data, b_data, c_data])
            anderson_test_results[metric] = {
                'statistic': stat,
                'critical_values': crit,
                'significance_level': sig_level,
                'significant': stat > crit[-1]
            }
        self.statistical_tests['anderson_ksamp'] = anderson_test_results
        print(f"   ✅ Anderson-Darling tests completed")  # noqa: F541
        for metric, result in anderson_test_results.items():
            print(f"      {metric}: Statistic={result['statistic']:.3f}, Critical={result['critical_values'][-1]:.3f}")
        return self.statistical_tests['anderson_ksamp']

    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================

    def plot_subscription_rates(self) -> str:
        """
        Plot subscription rates by test group.
        Returns
        -------
        str
            Path to saved plot
        """
        if self.test_groups is None:
            self.create_test_groups()
            self.simulate_outcomes()
        print("[VISUALIZATION] Plotting subscription rates...")
        # Calculate subscription rates
        subscription_rates = self.test_groups.groupby('test_group')['subscribed'].mean().reset_index()
        plt.figure(figsize=(8, 6))
        sns.barplot(
            data=subscription_rates,
            x='test_group',
            y='subscribed',
            palette='viridis'
        )
        plt.title('Subscription Rates by Test Group')
        plt.ylabel('Subscription Rate')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        save_path = os.path.join(self.output_dir, 'subscription_rates.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        plt.close()
        print(f"   ✅ Saved subscription rates plot: {save_path}")
        return save_path

    def plot_spending_increase(self) -> str:
        """
        Plot spending increase by test group.
        Returns
        -------
        str
            Path to saved plot
        """
        if self.test_groups is None:
            self.create_test_groups()
            self.simulate_outcomes()
        print("[VISUALIZATION] Plotting spending increase...")
        plt.figure(figsize=(8, 6))
        sns.boxplot(
            data=self.test_groups,
            x='test_group',
            y='spending_increase',
            palette='viridis'
        )
        plt.title('Spending Increase by Test Group')
        plt.ylabel('Spending Increase Factor')
        plt.grid(axis='y', alpha=0.3)
        save_path = os.path.join(self.output_dir, 'spending_increase.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        plt.close()
        print(f"   ✅ Saved spending increase plot: {save_path}")
        return save_path

    def plot_engagement_scores(self) -> str:
        """
        Plot engagement scores by test group.
        Returns
        -------
        str
            Path to saved plot
        """
        if self.test_groups is None:
            self.create_test_groups()
            self.simulate_outcomes()
        print("[VISUALIZATION] Plotting engagement scores...")
        plt.figure(figsize=(8, 6))
        sns.boxplot(
            data=self.test_groups,
            x='test_group',
            y='engagement_score',
            palette='viridis'
        )
        plt.title('Engagement Scores by Test Group')
        plt.ylabel('Engagement Score')
        plt.grid(axis='y', alpha=0.3)
        save_path = os.path.join(self.output_dir, 'engagement_scores.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        plt.close()
        print(f"   ✅ Saved engagement scores plot: {save_path}")
        return save_path

    # ========================================================================
    # REPORTING
    # ========================================================================

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive A/B test report.
        Returns
        -------
        Dict[str, Any]
            A/B test report
        """
        if self.test_groups is None:
            self.create_test_groups()
            self.simulate_outcomes()
        print("\n[REPORT] Generating comprehensive A/B test report...")
        # Run statistical tests
        chi_square = self.run_chi_square_test()
        fisher_exact_test = self.run_fisher_exact_test()
        t_tests = self.run_t_tests()
        mann_whitney_u_tests = self.run_mann_whitney_u_test()
        ks_tests = self.run_ks_test()
        anderson_tests = self.run_anderson_ksamp_test()

        # Generate visualizations
        subscription_plot = self.plot_subscription_rates()
        spending_plot = self.plot_spending_increase()
        engagement_plot = self.plot_engagement_scores()

        # Calculate metrics
        metrics = {
            'subscription_rates': self.test_groups.groupby('test_group')['subscribed'].mean().to_dict(),
            'avg_spending_increase': self.test_groups.groupby('test_group')['spending_increase'].mean().to_dict(),
            'avg_engagement_score': self.test_groups.groupby('test_group')['engagement_score'].mean().to_dict(),
            'user_counts': self.test_groups['test_group'].value_counts().to_dict()
        }

        # Compile report
        report = {
            'metadata': {
                'test_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'user_count': len(self.test_groups),
                'test_ratio': self.test_ratio,
                'random_state': self.random_state
            },
            'metrics': metrics,
            'statistical_tests': {
                'chi_square': chi_square,
                'fisher_exact': fisher_exact_test,
                't_tests': t_tests,
                'mann_whitney_u': mann_whitney_u_tests,
                'ks_test': ks_tests,
                'anderson_ksamp': anderson_tests
            },
            'visualizations': {
                'subscription_rates': subscription_plot,
                'spending_increase': spending_plot,
                'engagement_scores': engagement_plot
            }
        }

        self.test_results = report
        print(f"   ✅ A/B test report generated")  # noqa: F541
        return report

    def export_results(self, filename: str = 'ab_test_results.csv') -> str:
        """
        Export test results to CSV.
        Parameters
        ----------
        filename : str
            Name of the output file
        Returns
        -------
        str
            Path to saved file
        """
        if self.test_groups is None:
            self.create_test_groups()
            self.simulate_outcomes()
        print(f"[EXPORT] Saving test results to {filename}...")
        save_path = os.path.join(self.data_path, filename)
        self.test_groups.to_csv(save_path, index=False)
        print(f"   ✅ Saved test results to: {save_path}")
        return save_path

    # ========================================================================
    # MASTER RUNNER
    # ========================================================================

    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run complete A/B test analysis.
        Returns
        -------
        Dict[str, Any]
            Complete A/B test results
        """
        print("\n🚀 Running complete A/B test analysis...\n")
        # Create test groups and simulate outcomes
        self.create_test_groups()
        self.simulate_outcomes()
        # Generate report
        report = self.generate_report()
        # Export results
        self.export_results()
        print("\n✅ A/B test analysis completed!")
        return report

