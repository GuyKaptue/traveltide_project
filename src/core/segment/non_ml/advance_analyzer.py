# core/segment/non_ml/advance_analyzer.py

import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import scipy.stats as stats # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
import os
import warnings
warnings.filterwarnings('ignore')

class AdvanceSegmentAnalyzer:
    """
    Complete professional segment analysis for customer behavioral data.
    Provides actionable insights for marketing team strategy.
    """
    
    def __init__(self, df, segment_col='assigned_group', segment_perk_col='assigned_perk', 
                 pca_data_path=None, pca_fig_path=None):
        self.df = df.copy()
        self.segment_col = segment_col
        self.segment_perk_col = segment_perk_col
        self.pca_data_path = pca_data_path
        self.pca_fig_path = pca_fig_path
        
        self._print_debug_info()
        self.setup_plot_style()
        
    
    def _print_debug_info(self):
        """Print comprehensive debug information"""
        print("🔍 DATASET DEBUG INFORMATION")
        print("=" * 80)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        if self.segment_col in self.df.columns:
            segment_counts = self.df[self.segment_col].value_counts()
            print(f"\n📊 Segment distribution:")  # noqa: F541
            for segment, count in segment_counts.items():
                pct = count / len(self.df) * 100
                print(f"   {segment}: {count:,} ({pct:.1f}%)")
        
        if self.segment_perk_col in self.df.columns:
            perk_counts = self.df[self.segment_perk_col].value_counts()
            print(f"\n🎁 Segment perk distribution:")  # noqa: F541
            for perk, count in perk_counts.items():
                pct = count / len(self.df) * 100
                print(f"   {perk}: {count:,} ({pct:.1f}%)")
        
        print(f"\n📈 Data Quality: Missing={self.df.isnull().sum().sum()}, Duplicates={self.df.duplicated().sum()}")
        print("=" * 80)
    
    def setup_plot_style(self):
        """Set professional plotting style"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11
        
    def _save_plot(self, filename, dpi=300):
        """Save plot to specified path"""
        if self.pca_fig_path:
            os.makedirs(self.pca_fig_path, exist_ok=True)
            full_path = os.path.join(self.pca_fig_path, filename)
            plt.savefig(full_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"💾 Plot saved: {full_path}")
    
    def _save_data(self, data, filename):
        """Save data to specified path"""
        if self.pca_data_path:
            os.makedirs(self.pca_data_path, exist_ok=True)
            full_path = os.path.join(self.pca_data_path, filename)
            if isinstance(data, pd.DataFrame):
                data.to_csv(full_path, index=True)
            elif isinstance(data, dict):
                pd.Series(data).to_csv(full_path)
            print(f"💾 Data saved: {full_path}")
    
    def descriptive_analysis(self):
        """Comprehensive descriptive analysis of segments"""
        print("\n" + "=" * 80)
        print("📊 DESCRIPTIVE SEGMENT ANALYSIS")
        print("=" * 80)
        
        segment_counts = self.df[self.segment_col].value_counts()
        perk_counts = self.df[self.segment_perk_col].value_counts()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Segment distribution pie chart
        colors1 = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
        axes[0,0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
                     colors=colors1, startangle=90)
        axes[0,0].set_title('Customer Segment Distribution', fontsize=14, fontweight='bold')
        
        # Perk distribution pie chart
        colors2 = plt.cm.Pastel1(np.linspace(0, 1, len(perk_counts)))
        axes[0,1].pie(perk_counts.values, labels=perk_counts.index, 
                     autopct='%1.1f%%', colors=colors2, startangle=90)
        axes[0,1].set_title('Perk Assignment Distribution', fontsize=14, fontweight='bold')
        
        # Demographics by segment
        demographic_vars = ['gender', 'married', 'has_children']
        available_demos = [v for v in demographic_vars if v in self.df.columns]
        
        for i, demo_var in enumerate(available_demos[:2]):
            ct = pd.crosstab(self.df[self.segment_col], self.df[demo_var], normalize='index') * 100
            ct.plot(kind='bar', ax=axes[1,i], stacked=True, edgecolor='black', width=0.7)
            axes[1,i].set_title(f'{demo_var.replace("_", " ").title()} by Segment', fontweight='bold')
            axes[1,i].set_ylabel('Percentage (%)')
            axes[1,i].legend(title=demo_var, bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1,i].set_xlabel('Segment')
            plt.setp(axes[1,i].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        self._save_plot('01_descriptive_analysis.png')
        plt.show()
        
        self._print_segment_profiles()
    
    def _print_segment_profiles(self):
        """Print detailed segment profiles"""
        print("\n📋 SEGMENT PROFILES:")
        print("-" * 80)
        
        key_metrics = ['total_spend', 'num_trips', 'num_sessions', 'cancelation_rate', 
                      'age', 'num_flights', 'num_hotels', 'avg_bags', 'avg_km_flown']
        available = [m for m in key_metrics if m in self.df.columns]
        
        segment_profiles = self.df.groupby(self.segment_col)[available].agg(['mean', 'median', 'std'])
        
        for segment in self.df[self.segment_col].unique():
            print(f"\n🎯 {segment}:")
            perk = self.df[self.df[self.segment_col]==segment][self.segment_perk_col].iloc[0]
            print(f"   Assigned Perk: {perk}")
            print(f"   Size: {len(self.df[self.df[self.segment_col]==segment]):,} customers")
            
            for metric in available:
                mean_val = segment_profiles.loc[segment, (metric, 'mean')]
                print(f"   Avg {metric}: {mean_val:.2f}")
    
    def behavioral_comparison(self):
        """Compare behavioral patterns across segments"""
        print("\n" + "=" * 80)
        print("📈 BEHAVIORAL COMPARISON ANALYSIS")
        print("=" * 80)
        
        behavioral_categories = {
            'Engagement': ['num_clicks', 'num_sessions', 'avg_session_clicks', 
                        'total_browsing_clicks', 'browsing_intensity'],
            'Spending': ['total_spend', 'avg_money_spent_flight', 'money_spent_hotel_total',
                        'avg_money_spent_hotel_trip', 'avg_money_spent_per_seat'],
            'Travel': ['num_trips', 'num_flights', 'num_hotels', 'num_destinations', 
                    'avg_km_flown', 'avg_trip_length'],
            'Efficiency': ['cancellation_rate', 'num_empty_sessions', 'conversion_rate', 'weekend_ratio']
        }

        
        available_categories = {cat: [m for m in metrics if m in self.df.columns] 
                               for cat, metrics in behavioral_categories.items()}
        available_categories = {k: v for k, v in available_categories.items() if v}
        
        n_cats = len(available_categories)
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (category, metrics) in enumerate(available_categories.items()):
            if idx < 4:
                segment_means = self.df.groupby(self.segment_col)[metrics].mean()
                
                # Normalize for comparison
                scaler = StandardScaler()
                normalized = pd.DataFrame(
                    scaler.fit_transform(segment_means),
                    index=segment_means.index,
                    columns=segment_means.columns
                )
                
                normalized.T.plot(kind='bar', ax=axes[idx], width=0.7)
                axes[idx].set_title(f'{category} Patterns (Standardized)', fontweight='bold', fontsize=12)
                axes[idx].set_ylabel('Standardized Score')
                axes[idx].legend(title='Segment', bbox_to_anchor=(1.05, 1))
                axes[idx].axhline(y=0, color='black', linestyle='--', alpha=0.3)
                plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        for idx in range(n_cats, 4):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        self._save_plot('02_behavioral_comparison.png')
        plt.show()
        
        self._statistical_tests()
    
    def _statistical_tests(self):
        """Statistical significance testing between segments"""
        print("\n🔬 STATISTICAL SIGNIFICANCE TESTS:")
        print("-" * 80)
        
        test_metrics = ['total_spend', 'num_trips', 'cancelation_rate', 'num_sessions']
        available = [m for m in test_metrics if m in self.df.columns]
        
        segments = self.df[self.segment_col].unique()
        results = []
        
        for metric in available:
            segment_groups = [self.df[self.df[self.segment_col]==seg][metric].dropna() 
                            for seg in segments]
            segment_groups = [g for g in segment_groups if len(g) > 0]
            
            if len(segment_groups) >= 2:
                if len(segment_groups) == 2:
                    stat, pval = stats.ttest_ind(segment_groups[0], segment_groups[1])
                    test_type = "T-test"
                else:
                    stat, pval = stats.f_oneway(*segment_groups)
                    test_type = "ANOVA"
                
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
                results.append({'Metric': metric, 'Test': test_type, 'Statistic': stat, 
                              'P-value': pval, 'Significance': sig})
                print(f"{metric:25} | {test_type:8} | stat={stat:7.2f} | p={pval:.4f} {sig}")
        
        results_df = pd.DataFrame(results)
        self._save_data(results_df, 'statistical_tests.csv')
    
    def spending_analysis(self):
        """Deep dive into spending patterns"""
        print("\n" + "=" * 80)
        print("💰 SPENDING PATTERN ANALYSIS")
        print("=" * 80)
        
        spending_cols = ['total_spend', 'base_fare_usd', 'hotel_price_per_room_night_usd',
                        'avg_money_spent_flight', 'avg_money_spent_hotel']
        available = [c for c in spending_cols if c in self.df.columns]
        
        if not available:
            print("❌ No spending metrics available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Box plots for spending distribution
        for idx, col in enumerate(available[:4]):
            self.df.boxplot(column=col, by=self.segment_col, ax=axes[idx])
            axes[idx].set_title(f'{col.replace("_", " ").title()}', fontweight='bold')
            axes[idx].set_xlabel('Segment')
            axes[idx].set_ylabel('USD ($)')
            plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Spending Distribution by Segment', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        self._save_plot('03_spending_analysis.png')
        plt.show()
        
        # Calculate spending metrics
        print("\n💵 SPENDING METRICS BY SEGMENT:")
        print("-" * 60)
        for col in available:
            print(f"\n{col.replace('_', ' ').title()}:")
            segment_stats = self.df.groupby(self.segment_col)[col].agg(['mean', 'median', 'sum'])
            print(segment_stats.to_string())
    
    def customer_journey_analysis(self):
        """Analyze customer journey and conversion patterns"""
        print("\n" + "=" * 80)
        print("🛤️  CUSTOMER JOURNEY ANALYSIS")
        print("=" * 80)
        
        journey_metrics = {
            'Booking': ['num_group_trips', 'num_pair_trips', 'num_business_trips',
                        'num_weekend_trips_agg', 'num_discount_trips_agg'],
            'Engagement': ['num_browsing_sessions', 'avg_browsing_duration', 'total_browsing_clicks'],
            'Conversion': ['num_trips', 'num_flights', 'num_hotels'],
            'Retention': ['cancellation_rate', 'num_canceled_trips']
        }

        
        available_journey = {}
        for cat, metrics in journey_metrics.items():
            avail = [m for m in metrics if m in self.df.columns]
            if avail:
                available_journey[cat] = avail
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (category, metrics) in enumerate(list(available_journey.items())[:4]):
            segment_means = self.df.groupby(self.segment_col)[metrics].mean()
            segment_means.plot(kind='bar', ax=axes[idx], width=0.7)
            axes[idx].set_title(f'{category} Metrics by Segment', fontweight='bold')
            axes[idx].set_ylabel('Average Value')
            axes[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        self._save_plot('04_customer_journey.png')
        plt.show()
        
        # Print conversion funnel
        self._conversion_funnel()
    
    def _conversion_funnel(self):
        """Calculate conversion funnel metrics"""
        print("\n🎯 CONVERSION FUNNEL BY SEGMENT:")
        print("-" * 60)
        
        for segment in self.df[self.segment_col].unique():
            seg_data = self.df[self.df[self.segment_col] == segment]
            print(f"\n{segment}:")
            
            if 'num_sessions' in self.df.columns and 'num_trips' in self.df.columns:
                session_to_trip = (seg_data['num_trips'].sum() / seg_data['num_sessions'].sum() * 100)
                print(f"  Sessions → Trips: {session_to_trip:.1f}%")
            
            if 'cancelation_rate' in self.df.columns:
                avg_cancel = seg_data['cancelation_rate'].mean() * 100
                print(f"  Cancellation Rate: {avg_cancel:.1f}%")
    
    def predictive_analysis(self):
        """Feature importance for segment prediction"""
        print("\n" + "=" * 80)
        print("🔮 PREDICTIVE FEATURE IMPORTANCE")
        print("=" * 80)
        
        # Select features for prediction
        feature_cols = [
            'num_clicks', 'num_sessions', 'num_trips', 'total_spend',
            'num_flights', 'num_hotels', 'cancellation_rate', 'age',
            'avg_session_duration', 'weekend_ratio', 'RFM_score',
            'bargain_hunter_index', 'sustainable_index', 'global_booking_share'
        ]

        
        available_features = [f for f in feature_cols if f in self.df.columns]
        
        if len(available_features) < 3:
            print("❌ Insufficient features for predictive analysis")
            return
        
        # Prepare data
        X = self.df[available_features].fillna(self.df[available_features].median())
        y = self.df[self.segment_col]
        
        # Train random forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X, y)
        
        # Feature importance
        importance_df = pd.DataFrame({
            'Feature': available_features,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
        
        # Color bars by importance
        colors = plt.cm.RdYlGn(importance_df['Importance'] / importance_df['Importance'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Feature Importance', fontweight='bold')
        ax.set_title('Key Drivers of Segment Classification', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        self._save_plot('05_feature_importance.png')
        plt.show()
        
        print("\n🔑 TOP 5 DISCRIMINATING FEATURES:")
        print(importance_df.head().to_string(index=False))
        
        self._save_data(importance_df, 'feature_importance.csv')
    
    def business_impact_analysis(self):
        """Calculate business impact and ROI potential"""
        print("\n" + "=" * 80)
        print("💼 BUSINESS IMPACT & MARKETING RECOMMENDATIONS")
        print("=" * 80)
        
        # Segment value analysis
        segment_summary = {}
        
        for segment in self.df[self.segment_col].unique():
            seg_data = self.df[self.df[self.segment_col] == segment]
            perk = seg_data[self.segment_perk_col].iloc[0]
            
            summary = {
                'Segment': segment,
                'Perk': perk,
                'Size': len(seg_data),
                'Pct_of_Base': len(seg_data) / len(self.df) * 100
            }
            
            if 'total_spend' in self.df.columns:
                summary['Avg_Spend'] = seg_data['total_spend'].mean()
                summary['Total_Revenue'] = seg_data['total_spend'].sum()
                summary['Revenue_Share_Pct'] = seg_data['total_spend'].sum() / self.df['total_spend'].sum() * 100
            
            if 'num_trips' in self.df.columns:
                summary['Avg_Trips'] = seg_data['num_trips'].mean()
            
            if 'cancelation_rate' in self.df.columns:
                summary['Avg_Cancel_Rate'] = seg_data['cancelation_rate'].mean()
            
            segment_summary[segment] = summary
        
        summary_df = pd.DataFrame(segment_summary).T
        
        print("\n📊 SEGMENT BUSINESS VALUE:")
        print(summary_df.to_string())
        
        self._save_data(summary_df, 'business_impact_summary.csv')
        
        # Marketing recommendations
        self._marketing_recommendations(summary_df)
    
    def _marketing_recommendations(self, summary_df):
        """Generate actionable marketing recommendations"""
        print("\n" + "=" * 80)
        print("🎯 MARKETING STRATEGY RECOMMENDATIONS")
        print("=" * 80)
        # Ensure numeric dtype for key columns
        for col in ['Avg_Spend', 'Size', 'Avg_Cancel_Rate']:
            if col in summary_df.columns:
                summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')
        
        if 'Avg_Spend' in summary_df.columns:
            high_value = summary_df.nlargest(1, 'Avg_Spend').index[0]
            print(f"\n💎 HIGH-VALUE SEGMENT: {high_value}")
            print(f"   → Perk: {summary_df.loc[high_value, 'Perk']}")
            print(f"   → Strategy: Premium retention campaigns, VIP experiences")  # noqa: F541
            print(f"   → Avg Spend: ${summary_df.loc[high_value, 'Avg_Spend']:.2f}")
        
        if 'Size' in summary_df.columns:
            largest = summary_df.nlargest(1, 'Size').index[0]
            print(f"\n👥 LARGEST SEGMENT: {largest}")
            print(f"   → Perk: {summary_df.loc[largest, 'Perk']}")
            print(f"   → Strategy: Mass market campaigns, loyalty programs")  # noqa: F541
            print(f"   → Size: {summary_df.loc[largest, 'Size']:,} customers")
        
        if 'Avg_Cancel_Rate' in summary_df.columns:
            at_risk = summary_df.nlargest(1, 'Avg_Cancel_Rate').index[0]
            print(f"\n⚠️  AT-RISK SEGMENT: {at_risk}")
            print(f"   → Perk: {summary_df.loc[at_risk, 'Perk']}")
            print(f"   → Strategy: Retention focus, flexible policies, satisfaction surveys")  # noqa: F541
            print(f"   → Cancel Rate: {summary_df.loc[at_risk, 'Avg_Cancel_Rate']*100:.1f}%")
    
    def perk_effectiveness_analysis(self):
        """Analyze perk assignment effectiveness"""
        print("\n" + "=" * 80)
        print("🎁 PERK EFFECTIVENESS ANALYSIS")
        print("=" * 80)
        
        # Cross-tabulation of segments and perks
        perk_segment_ct = pd.crosstab(
            self.df[self.segment_col], 
            self.df[self.segment_perk_col],
            margins=True
        )
        
        print("\n📋 SEGMENT-PERK ASSIGNMENT MATRIX:")
        print(perk_segment_ct)
        
        # Analyze metrics by perk type
        perk_metrics = ['total_spend', 'num_trips', 'cancelation_rate', 'num_sessions']
        available = [m for m in perk_metrics if m in self.df.columns]
        
        if available:
            fig, axes = plt.subplots(1, len(available), figsize=(5*len(available), 6))
            if len(available) == 1:
                axes = [axes]
            
            for idx, metric in enumerate(available):
                perk_means = self.df.groupby(self.segment_perk_col)[metric].mean().sort_values()
                perk_means.plot(kind='barh', ax=axes[idx], color='skyblue', edgecolor='black')
                axes[idx].set_title(f'Avg {metric.replace("_", " ").title()}\nby Perk Type', 
                                   fontweight='bold')
                axes[idx].set_xlabel('Average Value')
                
                # Add value labels
                for i, v in enumerate(perk_means):
                    axes[idx].text(v, i, f' {v:.1f}', va='center', fontweight='bold')
            
            plt.tight_layout()
            self._save_plot('06_perk_effectiveness.png')
            plt.show()
    
    def comprehensive_report(self):
        """Generate complete analysis report"""
        print("\n" + "🚀" * 40)
        print("COMPREHENSIVE CUSTOMER SEGMENTATION ANALYSIS")
        print("🚀" * 40)
        
        analyses = [
            ('Descriptive Analysis', self.descriptive_analysis),
            ('Behavioral Comparison', self.behavioral_comparison),
            ('Spending Analysis', self.spending_analysis),
            ('Customer Journey', self.customer_journey_analysis),
            ('Predictive Analysis', self.predictive_analysis),
            ('Perk Effectiveness', self.perk_effectiveness_analysis),
            ('Business Impact', self.business_impact_analysis)
        ]
        
        successful = 0
        for name, func in analyses:
            try:
                print(f"\n{'='*80}")
                print(f"▶️  Running: {name}")
                print('='*80)
                func()
                successful += 1
                print(f"✅ {name} completed successfully")
            except Exception as e:
                print(f"❌ {name} failed: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 80)
        print(f"🎉 ANALYSIS COMPLETE!")  # noqa: F541
        print(f"✅ {successful}/{len(analyses)} analyses completed successfully")
        print("=" * 80)
        
        if self.pca_fig_path:
            print(f"\n📁 All visualizations saved to: {self.pca_fig_path}")
        if self.pca_data_path:
            print(f"📁 All data outputs saved to: {self.pca_data_path}")

