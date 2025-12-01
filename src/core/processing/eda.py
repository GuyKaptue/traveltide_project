# src/core/processing/eda.py
"""
TravelTideEDA: Vollständige explorative Datenanalyse für TravelTide Rewards-Programm.
REFACTORED VERSION: Uses elena_cohort.sql and integrates with existing project structure.
"""
# type: ignore
import os
import sys
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from scipy import stats # type: ignore  # noqa: F401
from typing import Optional, Dict  # noqa: F401
from datetime import datetime  # noqa: F401
from IPython.display import display  # type: ignore

import geopandas as gpd  # pyright: ignore[reportMissingModuleSource]
import cartopy.crs as ccrs # type: ignore
import cartopy.feature as cfeature # type: ignore
from shapely.geometry import LineString # type: ignore

# Add core module to path
cwd = os.getcwd()
project_root = os.path.abspath(os.path.join(cwd, "..", ".."))
sys.path.insert(0, project_root)


from src.utils import get_path  # noqa: E402
from src.core.processing import DataLoader  # noqa: E402

class TravelTideEDA:
    """
    Comprehensive Exploratory Data Analysis (EDA) for TravelTide Rewards Program.
    Uses elena_cohort.sql for enriched session data with user, flight, and hotel information.
    """
    def __init__(self, verbosity: int = 3):
        """
        Initialize TravelTideEDA with specified verbosity level.
        
        Args:
            verbosity: 1 (Basic), 2 (Intermediate), 3 (Advanced)
        """
        self.verbosity = verbosity
        
        # Use core utility paths
        self.fig_dir = os.path.join(get_path('eda'), 'figures')
        self.data_dir = os.path.join(get_path('eda'), 'results')
        
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize DataLoader
        self.loader = DataLoader()
        
        # Set plot style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
        
        # Cache for loaded data
        self.enriched_sessions_df: Optional[pd.DataFrame] = None
        
        self._vprint(1, f"\n{'='*80}")
        self._vprint(1, f"TravelTide EDA initialized - Verbosity Level: {self.verbosity}")
        self._vprint(1, f"Mode: Using elena_cohort.sql")  # noqa: F541
        self._vprint(1, f"Output directories:")  # noqa: F541
        self._vprint(1, f"  Figures: {self.fig_dir}")
        self._vprint(1, f"  Data: {self.data_dir}")
        self._vprint(1, f"{'='*80}\n")

    def _vprint(self, level: int, *args, **kwargs):
        """Helper function for conditional printing based on verbosity level."""
        if self.verbosity >= level:
            print(*args, **kwargs)
    
    def _display(self, level: int, obj):
        """Helper function for conditional display of DataFrames."""
        if self.verbosity >= level:
            display(obj)

    def _save_plot(self, filename: str, show_plot: bool = True):
        """Save plot to file and optionally display it."""
        path = os.path.join(self.fig_dir, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        self._vprint(2, f"  ✅ Plot saved: {filename}")
        if show_plot and self.verbosity >= 2:
            plt.show()
        plt.close()

    def _save_data(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to CSV."""
        path = os.path.join(self.data_dir, filename)
        df.to_csv(path, index=False)
        self._vprint(2, f"  ✅ Data saved: {filename} ({len(df)} rows)")
        return path

    def close(self):
        """Closes the database connection."""
        if hasattr(self.loader, 'db') and self.loader.db:
            self.loader.db.close()
        self._vprint(1, "\n✅ Database connection closed.")

    # ==================== DATA LOADING ====================
    
    def _load_enriched_sessions(self, reload: bool = False) -> pd.DataFrame:
        """
        Load enriched sessions using elena_cohort.sql.
        
        Args:
            reload: If True, forces reload from SQL even if CSV exists
            
        Returns:
            DataFrame with enriched session data
        """
        # Resolve path to the raw CSV file
        sessions_cleaned_path = os.path.join(get_path('processed'), "sessions_cleaned.csv")
        cohort_path = os.path.join(get_path('raw'), "elena_cohort.csv")
        
        if not reload and self.enriched_sessions_df is not None:
            self._vprint(3, "  ♻️ Using cached enriched sessions data")
            return self.enriched_sessions_df
        
        if not os.path.exists(sessions_cleaned_path) or not os.path.exists(cohort_path):
            self._vprint(2, "\n📥 Loading enriched sessions from elena_cohort.sql...")

        # Check if the file exists
        if  os.path.exists(sessions_cleaned_path):
            self._vprint(2, "📄 Loaded sessions cleaned CSV")
            df = self.loader.load_table(
                data_type='processed',
                table_name='sessions_cleaned',
                show_table_display=(self.verbosity >= 3),
                is_session_base=False
            )
        elif os.path.exists(cohort_path):
            self._vprint(2, "📄 Loaded Elena cohort locally from CSV")
            df = self.loader.load_table(
                data_type='raw',
                table_name='elena_cohort',
                show_table_display=(self.verbosity >= 3),
                is_session_base=False
            )
        else:
            self._vprint(2, "🗄️ Loading Elena cohort from SQL fallback")
            df = self.loader.load_table(
                data_type='sql',
                table_name='elena_cohort',
                show_table_display=(self.verbosity >= 3),
                is_session_base=True
            )
        
        if df.empty:
            raise ValueError("❌ Failed to load enriched sessions data")
        
        self._vprint(2, f"✅ Loaded {len(df):,} enriched session records")
        self._vprint(2, f"   Unique users: {df['user_id'].nunique():,}")
        self._vprint(2, f"   Unique sessions: {df['session_id'].nunique():,}")
        
        # Cache the data
        self.enriched_sessions_df = df
        
        return df

    def _ensure_age_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure age_2025 column exists, calculate if needed."""
        if 'age_2025' not in df.columns:
            self._vprint(2, "  ⚠️ 'age_2025' column missing, calculating...")
            
            if 'birthdate' in df.columns:
                df['birthdate'] = pd.to_datetime(df['birthdate'], errors='coerce')
                df['age_2025'] = 2025 - df['birthdate'].dt.year
                self._vprint(2, "  ✅ 'age_2025' calculated from birthdate")
            else:
                self._vprint(2, "  ⚠️ Cannot calculate age - no birthdate column")
        
        return df

    # ==================== PHASE 1: STRUCTURE & DATA QUALITY ====================
    
    def analyze_enriched_sessions_structure(self):
        """Analyzes structure, types, and missing values for enriched sessions."""
        self._vprint(1, "\n🔹 PHASE 1: STRUKTURVERSTÄNDNIS & DATENQUALITÄT\n")
        
        self._vprint(2, f"\n{'─'*60}")
        self._vprint(2, "Analyzing: ENRICHED SESSIONS DATASET (elena_cohort)")
        self._vprint(2, f"{'─'*60}")
        
        # Load enriched sessions
        df = self._load_enriched_sessions()
        
        # Show structure
        self._vprint(2, f"\n📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        if self.verbosity >= 3:
            self._vprint(3, "\n📋 Column Data Types:")
            self.loader.show_dtypes(df)
            
            self._vprint(3, "\n🔍 Missing Values:")
            self.loader.show_nulls(df)
            
            self._vprint(3, "\n🔢 Unique Value Counts:")
            self.loader.show_unique_counts(df)
        elif self.verbosity >= 2:
            self._vprint(2, "\n🔍 Missing Values Summary:")
            self.loader.show_nulls(df)
        
        # Create structure summary
        structure_summary = {
            'dataset': 'elena_cohort',
            'rows': len(df),
            'columns': len(df.columns),
            'unique_users': df['user_id'].nunique(),
            'unique_sessions': df['session_id'].nunique(),
            'unique_trips': df['trip_id'].nunique() if 'trip_id' in df.columns else 0,
            'missing_values': df.isnull().sum().sum(),
            'missing_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        summary_df = pd.DataFrame([structure_summary]).T
        summary_df.columns = ['value']
        self._save_data(summary_df.reset_index(), 'enriched_sessions_structure_summary.csv')
        self._display(2, summary_df)
        
        # Check for potential outliers if verbosity is high
        if self.verbosity >= 3:
            self._vprint(3, "\n🔍 Outlier Detection Summary:")
            self.loader.generate_outlier_summary(df)
        
        return df

    #===================== Run SQL Query from File =====================

    def run_sql_query_from_file(self, filename: str, folder: str = "sql") -> pd.DataFrame:
        """
        Führt eine SQL-Abfrage aus, die in einer Datei gespeichert ist.

        Parameter:
        ----------
        filename : str
            Name der SQL-Datei (z. B. 'cancel_trips.sql').
        folder : str
            Unterordner im Projektverzeichnis, in dem sich die Datei befindet (Standard: 'sql').

        Rückgabe:
        ---------
        pd.DataFrame
            Ergebnis der SQL-Abfrage als DataFrame.
        """
        sql_path = os.path.join(get_path(folder), filename)

        if not os.path.exists(sql_path):
            raise FileNotFoundError(f"⚠️ SQL-Datei nicht gefunden: {sql_path}")

        with open(sql_path, "r", encoding="utf-8") as file:
            sql_query = file.read()

        self._vprint(1, f"\n🔹 Running SQL Query from File: {filename}\n")
        result_df = self.loader.load_custom_query(sql_query)

        return result_df

    # ==================== PHASE 2: DEMOGRAPHIC ANALYSIS ====================
    
    def analyze_user_demographics(self) -> pd.DataFrame:
        """Comprehensive analysis of user demographics from enriched sessions."""
        if self.enriched_sessions_df is None:
            self.enriched_sessions_df = self._load_enriched_sessions()
        
        self._vprint(1, "\n🔹 PHASE 2: DEMOGRAFISCHE ANALYSEN\n")
        
        # Get unique users (one row per user)
        user_cols = ['user_id', 'birthdate', 'gender', 'married', 'has_children', 
                    'home_country', 'home_city', 'sign_up_date']
        available_user_cols = [col for col in user_cols if col in self.enriched_sessions_df.columns]
        
        users_df = self.enriched_sessions_df[available_user_cols].drop_duplicates(subset=['user_id'])
        users_df = self._ensure_age_column(users_df)
        
        # Level 1: Basic
        if self.verbosity >= 1:
            self._vprint(1, "📊 Basic Demographics:\n")
            self._vprint(1, f"Total unique users: {len(users_df):,}\n")
            
            # Gender distribution
            if 'gender' in users_df.columns:
                self._vprint(1, "👥 Gender Distribution:")
                gender_counts = users_df['gender'].value_counts()
                for gender, count in gender_counts.items():
                    pct = (count / len(users_df)) * 100
                    self._vprint(1, f"   {gender}: {count:,} ({pct:.1f}%)")
            
            # Marital status
            if 'married' in users_df.columns:
                self._vprint(1, "\n💍 Marital Status:")
                married_counts = users_df['married'].value_counts()
                for status, count in married_counts.items():
                    pct = (count / len(users_df)) * 100
                    label = "Married" if status else "Not Married"
                    self._vprint(1, f"   {label}: {count:,} ({pct:.1f}%)")
            
            # Children status
            if 'has_children' in users_df.columns:
                self._vprint(1, "\n👶 Has Children:")
                children_counts = users_df['has_children'].value_counts()
                for status, count in children_counts.items():
                    pct = (count / len(users_df)) * 100
                    label = "Has Children" if status else "No Children"
                    self._vprint(1, f"   {label}: {count:,} ({pct:.1f}%)")
        
        # Level 2: Intermediate - Cross-tabulations
        if self.verbosity >= 2:
            self._vprint(2, "\n📈 Detailed Cross-Tabulation:\n")
            
            required_cols = ['gender', 'married', 'has_children']
            if all(col in users_df.columns for col in required_cols):
                # Gender x Married x Children
                cross_tab = pd.crosstab(
                    [users_df['gender'], users_df['married']],
                    users_df['has_children'],
                    normalize='all'
                ) * 100
                
                self._display(2, cross_tab.round(2))
                self._save_data(cross_tab.reset_index(), 'demographics_crosstab.csv')
                
                # Create demographic summary statistics
                agg_dict = {'user_id': 'count'}
                if 'age_2025' in users_df.columns:
                    agg_dict['age_2025'] = ['mean', 'median', 'std']
                
                demo_summary = users_df.groupby(['gender', 'married', 'has_children']).agg(agg_dict).round(2)
                
                if 'age_2025' in agg_dict:
                    demo_summary.columns = ['count', 'age_mean', 'age_median', 'age_std']
                else:
                    demo_summary.columns = ['count']
                
                demo_summary = demo_summary.reset_index()
                self._save_data(demo_summary, 'demographics_summary.csv')
                self._display(2, demo_summary)
            
            # Age statistics
            if 'age_2025' in users_df.columns:
                self._vprint(2, "\n📅 Age Statistics:")
                age_stats = users_df['age_2025'].describe()
                self._display(2, age_stats.to_frame('Age (2025)'))
            
            # Combined demographic summary
            self._vprint(2, "\n📋 Combined Demographic Summary Table:")
            summary_df = self.demographic_summary(users_df)
            self._save_data(summary_df.reset_index(), 'demographics_combined_summary.csv')
            self._display(2, summary_df)
        
        # Level 3: Advanced Visualizations
        if self.verbosity >= 3:
            self._vprint(3, "\n🎨 Creating Advanced Visualizations...\n")
            
            if 'age_2025' in users_df.columns:
                self.plot_demographics_heatmap(users_df)
                self.plot_demographics_violin(users_df)
                self.plot_age_distribution(users_df)
            
            if 'home_country' in users_df.columns:
                self.plot_geographic_distribution(users_df)
            
            self.plot_demographics_dashboard(users_df)
            self.plot_demographic_summary(summary_df, save_path=os.path.join(self.fig_dir, 'demographics_summary_plot.png'))
        
        return users_df
    
    def demographic_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create demographic summary table."""
        gender_counts = df['gender'].value_counts()
        married_counts = df[df['married'] == True]['gender'].value_counts()  # noqa: E712
        children_counts = df[df['has_children'] == True]['gender'].value_counts()  # noqa: E712

        summary_df = pd.DataFrame({
            'Gender': gender_counts,
            'Married': married_counts.reindex(gender_counts.index, fill_value=0),
            'Has_Children': children_counts.reindex(gender_counts.index, fill_value=0)
        })

        return summary_df

    def plot_demographic_summary(self, summary_df: pd.DataFrame, save_path: str = None):
        """Plot demographic summary bar chart."""
        labels = summary_df.index
        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width, summary_df['Gender'], width, label='Gesamtanzahl', color='#4C72B0')
        bars2 = ax.bar(x, summary_df['Married'], width, label='Verheiratet', color='#55A868')
        bars3 = ax.bar(x + width, summary_df['Has_Children'], width, label='Mit Kindern', color='#C44E52')

        ax.set_xlabel('Geschlecht')
        ax.set_ylabel('Anzahl Personen')
        ax.set_title('Demografischer Überblick nach Geschlecht')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self._vprint(2, f"✅ Grafik erfolgreich gespeichert unter: {save_path}")
        else:
            self._save_plot('demographics_summary_plot.png')

    # ==================== PHASE 3: SESSION BEHAVIOR ANALYSIS ====================
    
    def analyze_session_behavior(self):
        """Analyze session behavior patterns from enriched dataset."""
        if self.enriched_sessions_df is None:
            self.enriched_sessions_df = self._load_enriched_sessions()
        
        self._vprint(1, "\n🔹 PHASE 3: SESSION-VERHALTENSANALYSE\n")
        
        sessions_df = self.enriched_sessions_df.copy()
        
        # Basic statistics
        self._vprint(2, "🖱️ Session Statistics:\n")
        
        session_stats = {
            'total_sessions': len(sessions_df),
            'unique_users': sessions_df['user_id'].nunique(),
            'avg_page_clicks': sessions_df['page_clicks'].mean() if 'page_clicks' in sessions_df.columns else 0,
            'median_page_clicks': sessions_df['page_clicks'].median() if 'page_clicks' in sessions_df.columns else 0,
            'max_page_clicks': sessions_df['page_clicks'].max() if 'page_clicks' in sessions_df.columns else 0,
            'avg_sessions_per_user': len(sessions_df) / sessions_df['user_id'].nunique(),
            'flight_discount_usage_pct': (sessions_df['flight_discount'].sum() / len(sessions_df) * 100) if 'flight_discount' in sessions_df.columns else 0,
            'hotel_discount_usage_pct': (sessions_df['hotel_discount'].sum() / len(sessions_df) * 100) if 'hotel_discount' in sessions_df.columns else 0,
            'flight_booking_rate': (sessions_df['flight_booked'].sum() / len(sessions_df) * 100) if 'flight_booked' in sessions_df.columns else 0,
            'hotel_booking_rate': (sessions_df['hotel_booked'].sum() / len(sessions_df) * 100) if 'hotel_booked' in sessions_df.columns else 0
        }
        
        session_stats_df = pd.DataFrame([session_stats]).T
        session_stats_df.columns = ['value']
        self._display(2, session_stats_df.round(2))
        self._save_data(session_stats_df.reset_index(), 'session_statistics.csv')
        
        # User-level session aggregation
        if self.verbosity >= 2:
            agg_dict = {
                'session_id': 'count',
                'page_clicks': 'sum'
            }
            
            if 'flight_booked' in sessions_df.columns:
                agg_dict['flight_booked'] = 'sum'
            if 'hotel_booked' in sessions_df.columns:
                agg_dict['hotel_booked'] = 'sum'
            
            user_session_agg = sessions_df.groupby('user_id').agg(agg_dict)
            
            col_names = ['total_sessions', 'total_clicks']
            if 'flight_booked' in agg_dict:
                col_names.append('total_flights')
            if 'hotel_booked' in agg_dict:
                col_names.append('total_hotels')
            
            user_session_agg.columns = col_names
            
            self._vprint(2, "\n📊 User Session Aggregation Summary:")
            self._display(2, user_session_agg.describe())
            self._save_data(user_session_agg.reset_index(), 'user_session_aggregation.csv')
        
        # Visualizations
        if self.verbosity >= 3:
            self._vprint(3, "\n🎨 Creating Session Visualizations...\n")
            self.plot_session_analysis_dashboard(sessions_df)
        
        return sessions_df

    # ==================== PHASE 4: FLIGHT AND HOTEL BEHAVIOR ANALYSIS ====================
    
    def analyze_bookings_comprehensive(self):
        """Comprehensive analysis of flight and hotel bookings."""
        if self.enriched_sessions_df is None:
            self.enriched_sessions_df = self._load_enriched_sessions()
        
        self._vprint(1, "\n🔹 PHASE 4: BUCHUNGSANALYSE\n")
        
        df = self.enriched_sessions_df.copy()
        
        # Filter to actual bookings
        flight_bookings = df[df['flight_booked'] == True].copy() if 'flight_booked' in df.columns else pd.DataFrame()  # noqa: E712
        hotel_bookings = df[df['hotel_booked'] == True].copy() if 'hotel_booked' in df.columns else pd.DataFrame()  # noqa: E712
        
        # Flight Analysis
        if len(flight_bookings) > 0:
            self._vprint(2, "✈️ Flight Booking Statistics:\n")
            
            flight_stats = {
                'total_bookings': len(flight_bookings),
                'unique_users': flight_bookings['user_id'].nunique(),
                'avg_fare': flight_bookings['base_fare_usd'].mean() if 'base_fare_usd' in flight_bookings.columns else 0,
                'median_fare': flight_bookings['base_fare_usd'].median() if 'base_fare_usd' in flight_bookings.columns else 0,
                'total_revenue': flight_bookings['base_fare_usd'].sum() if 'base_fare_usd' in flight_bookings.columns else 0,
                'avg_discount': flight_bookings['flight_discount_amount'].mean() if 'flight_discount_amount' in flight_bookings.columns else 0,
                'cancellation_rate': (flight_bookings['cancellation'].sum() / len(flight_bookings) * 100) if 'cancellation' in flight_bookings.columns else 0,
                'avg_seats': flight_bookings['seats'].mean() if 'seats' in flight_bookings.columns else 0,
                'return_flight_pct': (flight_bookings['return_flight_booked'].sum() / len(flight_bookings) * 100) if 'return_flight_booked' in flight_bookings.columns else 0
            }
            
            flight_stats_df = pd.DataFrame([flight_stats]).T
            flight_stats_df.columns = ['value']
            self._display(2, flight_stats_df.round(2))
            self._save_data(flight_stats_df.reset_index(), 'flight_booking_statistics.csv')
            
            # Top destinations
            if 'destination_airport' in flight_bookings.columns and self.verbosity >= 2:
                top_dest = flight_bookings['destination_airport'].value_counts().head(10)
                self._vprint(2, "\n🏆 Top 10 Flight Destinations:")
                self._display(2, top_dest.to_frame('bookings'))
                self._save_data(top_dest.reset_index(), 'top_flight_destinations.csv')
        
        # Hotel Analysis
        if len(hotel_bookings) > 0:
            self._vprint(2, "\n🏨 Hotel Booking Statistics:\n")
            
            hotel_stats = {
                'total_bookings': len(hotel_bookings),
                'unique_users': hotel_bookings['user_id'].nunique(),
                'avg_price_per_night': hotel_bookings['hotel_price_per_room_night_usd'].mean() if 'hotel_price_per_room_night_usd' in hotel_bookings.columns else 0,
                'median_price_per_night': hotel_bookings['hotel_price_per_room_night_usd'].median() if 'hotel_price_per_room_night_usd' in hotel_bookings.columns else 0,
                'avg_nights': hotel_bookings['nights'].mean() if 'nights' in hotel_bookings.columns else 0,
                'avg_rooms': hotel_bookings['rooms'].mean() if 'rooms' in hotel_bookings.columns else 0,
                'total_revenue': hotel_bookings['hotel_price_per_room_night_usd'].sum() if 'hotel_price_per_room_night_usd' in hotel_bookings.columns else 0,
                'avg_discount': hotel_bookings['hotel_discount_amount'].mean() if 'hotel_discount_amount' in hotel_bookings.columns else 0,
                'cancellation_rate': (hotel_bookings['cancellation'].sum() / len(hotel_bookings) * 100) if 'cancellation' in hotel_bookings.columns else 0
            }
            
            hotel_stats_df = pd.DataFrame([hotel_stats]).T
            hotel_stats_df.columns = ['value']
            self._display(2, hotel_stats_df.round(2))
            self._save_data(hotel_stats_df.reset_index(), 'hotel_booking_statistics.csv')
        
        # Visualizations
        if self.verbosity >= 3 and (len(flight_bookings) > 0 or len(hotel_bookings) > 0):
            self._vprint(3, "\n🎨 Creating Booking Visualizations...\n")
            if len(flight_bookings) > 0:
                self.plot_flight_analysis_dashboard(flight_bookings)
            if len(hotel_bookings) > 0:
                self.plot_hotel_analysis_dashboard(hotel_bookings)

    # ==================== PHASE 5: COHORT ANALYSIS ====================
    
    def get_elena_cohort_summary(self) -> pd.DataFrame:
        """Generate comprehensive cohort summary from enriched sessions."""
        self._vprint(1, "\n🔹 PHASE 5: KOHORTENANALYSE & SEGMENTIERUNGSBASIS\n")
        
        try:
            if self.enriched_sessions_df is None:
                self.enriched_sessions_df = self._load_enriched_sessions()
            
            df = self.enriched_sessions_df.copy()
            df = self._ensure_age_column(df)
            
            self._vprint(2, "🔄 Building cohort metrics from enriched sessions...\n")
            
            # Build aggregation dictionary dynamically based on available columns
            agg_dict = {
                'session_id': 'count',
                'page_clicks': 'sum' if 'page_clicks' in df.columns else 'size',
            }
            
            # Add optional columns if they exist
            optional_metrics = {
                'flight_discount': 'sum',
                'hotel_discount': 'sum',
                'flight_booked': 'sum',
                'hotel_booked': 'sum',
                'base_fare_usd': lambda x: x.sum() if x.notna().any() else 0,
                'hotel_price_per_room_night_usd': lambda x: x.sum() if x.notna().any() else 0,
                'flight_discount_amount': lambda x: x.sum() if x.notna().any() else 0,
                'hotel_discount_amount': lambda x: x.sum() if x.notna().any() else 0,
            }
            
            for col, func in optional_metrics.items():
                if col in df.columns:
                    agg_dict[col] = func
            
            # Add time-based aggregations
            if 'session_start' in df.columns:
                agg_dict['session_start'] = ['min', 'max']
            
            # Add user attribute aggregations (first non-null value)
            user_attrs = ['birthdate', 'gender', 'married', 'has_children', 
                         'home_country', 'sign_up_date', 'age_2025']
            for attr in user_attrs:
                if attr in df.columns:
                    agg_dict[attr] = 'first'
            
            # Perform aggregation
            cohort_df = df.groupby('user_id').agg(agg_dict)
            
            # Flatten column names
            new_columns = []
            for col in cohort_df.columns:
                if isinstance(col, tuple):
                    if col[1] == '':
                        new_columns.append(col[0])
                    elif col[1] == 'min':
                        new_columns.append('first_session')
                    elif col[1] == 'max':
                        new_columns.append('last_session')
                    elif col[1] == 'first':
                        new_columns.append(col[0])
                    else:
                        new_columns.append(f'{col[0]}_{col[1]}')
                else:
                    new_columns.append(col)
            
            cohort_df.columns = new_columns
            cohort_df = cohort_df.reset_index()
            
            # Calculate conversion rates (safely handle division by zero)
            if 'total_flight_bookings' in cohort_df.columns and 'total_sessions' in cohort_df.columns:
                cohort_df['flight_conversion_rate'] = np.where(
                    cohort_df['total_sessions'] > 0,
                    (cohort_df['total_flight_bookings'] / cohort_df['total_sessions']) * 100,
                    0
                )
            
            if 'total_hotel_bookings' in cohort_df.columns and 'total_sessions' in cohort_df.columns:
                cohort_df['hotel_conversion_rate'] = np.where(
                    cohort_df['total_sessions'] > 0,
                    (cohort_df['total_hotel_bookings'] / cohort_df['total_sessions']) * 100,
                    0
                )
            
            # Calculate average spend per booking
            if 'total_base_fare_spend' in cohort_df.columns and 'total_flight_bookings' in cohort_df.columns:
                cohort_df['avg_flight_spend'] = np.where(
                    cohort_df['total_flight_bookings'] > 0,
                    cohort_df['total_base_fare_spend'] / cohort_df['total_flight_bookings'],
                    0
                )
            
            if 'total_hotel_price_per_room_night_spend' in cohort_df.columns and 'total_hotel_bookings' in cohort_df.columns:
                cohort_df['avg_hotel_spend'] = np.where(
                    cohort_df['total_hotel_bookings'] > 0,
                    cohort_df['total_hotel_price_per_room_night_spend'] / cohort_df['total_hotel_bookings'],
                    0
                )
            
            self._vprint(2, f"✅ Cohort created: {len(cohort_df)} users\n")
            
            # Display summary statistics
            if self.verbosity >= 2:
                metrics_to_show = [col for col in cohort_df.columns 
                                  if any(keyword in col.lower() for keyword in 
                                        ['total', 'conversion', 'avg']) 
                                  and col != 'user_id']
                
                if metrics_to_show:
                    summary_stats = cohort_df[metrics_to_show].describe().T
                    self._display(2, summary_stats)
            
            # Save cohort data
            self._save_data(cohort_df, 'elena_cohort_complete.csv')
            self._save_data(cohort_df, 'elena_cohort_aggregated.csv')
            
            return cohort_df
            
        except Exception as e:
            self._vprint(1, f"❌ Error creating cohort: {e}")
            import traceback
            if self.verbosity >= 3:
                self._vprint(3, traceback.format_exc())
            return pd.DataFrame()

    def _create_cohort_summary_report(self, cohort_df: pd.DataFrame):
        """
        Create comprehensive cohort summary report with key metrics.
        
        Args:
            cohort_df: Aggregated cohort DataFrame
        """
        self._vprint(2, "\n📊 Creating Cohort Summary Report...\n")
        
        try:
            report = {
                'Total Users': len(cohort_df),
                'Avg Sessions per User': cohort_df['total_sessions'].mean() if 'total_sessions' in cohort_df.columns else 0,
                'Median Sessions per User': cohort_df['total_sessions'].median() if 'total_sessions' in cohort_df.columns else 0,
                'Total Page Clicks': cohort_df['page_clicks'].sum() if 'page_clicks' in cohort_df.columns else 0,
                'Avg Page Clicks per User': cohort_df['page_clicks'].mean() if 'page_clicks' in cohort_df.columns else 0,
            }
            
            # Add booking metrics if available
            if 'total_flight_bookings' in cohort_df.columns:
                report['Total Flight Bookings'] = cohort_df['total_flight_bookings'].sum()
                report['Users with Flight Bookings'] = (cohort_df['total_flight_bookings'] > 0).sum()
                report['Flight Booking Rate %'] = (report['Users with Flight Bookings'] / len(cohort_df) * 100)
            
            if 'total_hotel_bookings' in cohort_df.columns:
                report['Total Hotel Bookings'] = cohort_df['total_hotel_bookings'].sum()
                report['Users with Hotel Bookings'] = (cohort_df['total_hotel_bookings'] > 0).sum()
                report['Hotel Booking Rate %'] = (report['Users with Hotel Bookings'] / len(cohort_df) * 100)
            
            # Revenue metrics
            if 'total_base_fare_spend' in cohort_df.columns:
                report['Total Flight Revenue'] = cohort_df['total_base_fare_spend'].sum()
                report['Avg Flight Revenue per User'] = cohort_df['total_base_fare_spend'].mean()
            
            if 'total_hotel_price_per_room_night_spend' in cohort_df.columns:
                report['Total Hotel Revenue'] = cohort_df['total_hotel_price_per_room_night_spend'].sum()
                report['Avg Hotel Revenue per User'] = cohort_df['total_hotel_price_per_room_night_spend'].mean()
            
            # Discount metrics
            if 'total_flight_discounts' in cohort_df.columns:
                report['Total Flight Discounts'] = cohort_df['total_flight_discounts'].sum()
            
            if 'total_hotel_discounts' in cohort_df.columns:
                report['Total Hotel Discounts'] = cohort_df['total_hotel_discounts'].sum()
            
            # Demographics
            if 'age_2025' in cohort_df.columns:
                report['Avg Age'] = cohort_df['age_2025'].mean()
                report['Median Age'] = cohort_df['age_2025'].median()
            
            if 'gender' in cohort_df.columns:
                report['Male Count'] = (cohort_df['gender'] == 'M').sum()
                report['Female Count'] = (cohort_df['gender'] == 'F').sum()
            
            if 'married' in cohort_df.columns:
                report['Married Count'] = cohort_df['married'].sum()
            
            if 'has_children' in cohort_df.columns:
                report['Has Children Count'] = cohort_df['has_children'].sum()
            
            # Convert to DataFrame
            report_df = pd.DataFrame.from_dict(report, orient='index', columns=['Value'])
            report_df.index.name = 'Metric'
            
            # Display and save
            self._display(2, report_df)
            self._save_data(report_df.reset_index(), 'cohort_summary_report.csv')
            
            # Create visualization
            if self.verbosity >= 3:
                self._plot_cohort_summary_dashboard(cohort_df, report)
            
            self._vprint(2, "✅ Cohort summary report created\n")
            
        except Exception as e:
            self._vprint(2, f"⚠️ Error creating cohort summary report: {e}")

    def _plot_cohort_summary_dashboard(self, cohort_df: pd.DataFrame, report: Dict):
        """Create visual dashboard for cohort summary."""
        try:
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Session distribution
            if 'total_sessions' in cohort_df.columns:
                ax1 = fig.add_subplot(gs[0, :])
                session_data = cohort_df['total_sessions'].clip(upper=20)
                sns.histplot(session_data, bins=20, kde=True, ax=ax1, color='steelblue')
                ax1.set_title('Sessions per User Distribution (capped at 20)', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Number of Sessions')
                ax1.set_ylabel('User Count')
            
            # 2. Flight bookings pie
            if 'total_flight_bookings' in cohort_df.columns:
                ax2 = fig.add_subplot(gs[1, 0])
                flight_users = (cohort_df['total_flight_bookings'] > 0).sum()
                no_flight_users = len(cohort_df) - flight_users
                ax2.pie([flight_users, no_flight_users], 
                       labels=['Booked Flights', 'No Flights'],
                       autopct='%1.1f%%',
                       colors=['#2ecc71', '#e74c3c'])
                ax2.set_title('Flight Booking Rate', fontweight='bold')
            
            # 3. Hotel bookings pie
            if 'total_hotel_bookings' in cohort_df.columns:
                ax3 = fig.add_subplot(gs[1, 1])
                hotel_users = (cohort_df['total_hotel_bookings'] > 0).sum()
                no_hotel_users = len(cohort_df) - hotel_users
                ax3.pie([hotel_users, no_hotel_users],
                       labels=['Booked Hotels', 'No Hotels'],
                       autopct='%1.1f%%',
                       colors=['#3498db', '#95a5a6'])
                ax3.set_title('Hotel Booking Rate', fontweight='bold')
            
            # 4. Age distribution
            if 'age_2025' in cohort_df.columns:
                ax4 = fig.add_subplot(gs[1, 2])
                sns.histplot(cohort_df['age_2025'].dropna(), bins=30, kde=True, ax=ax4, color='coral')
                ax4.set_title('Age Distribution', fontweight='bold')
                ax4.set_xlabel('Age')
                ax4.set_ylabel('Count')
            
            # 5. Revenue distribution (Flight)
            if 'total_base_fare_spend' in cohort_df.columns:
                ax5 = fig.add_subplot(gs[2, 0])
                revenue_data = cohort_df[cohort_df['total_base_fare_spend'] > 0]['total_base_fare_spend']
                if len(revenue_data) > 0:
                    sns.histplot(revenue_data.clip(upper=revenue_data.quantile(0.95)), 
                               bins=30, kde=True, ax=ax5, color='green')
                    ax5.set_title('Flight Revenue Distribution', fontweight='bold')
                    ax5.set_xlabel('Total Spend (USD)')
            
            # 6. Revenue distribution (Hotel)
            if 'total_hotel_price_per_room_night_spend' in cohort_df.columns:
                ax6 = fig.add_subplot(gs[2, 1])
                hotel_revenue = cohort_df[cohort_df['total_hotel_price_per_room_night_spend'] > 0]['total_hotel_price_per_room_night_spend']
                if len(hotel_revenue) > 0:
                    sns.histplot(hotel_revenue.clip(upper=hotel_revenue.quantile(0.95)),
                               bins=30, kde=True, ax=ax6, color='purple')
                    ax6.set_title('Hotel Revenue Distribution', fontweight='bold')
                    ax6.set_xlabel('Total Spend (USD)')
            
            # 7. Page clicks distribution
            if 'page_clicks' in cohort_df.columns:
                ax7 = fig.add_subplot(gs[2, 2])
                clicks_data = cohort_df['page_clicks'].clip(upper=100)
                sns.histplot(clicks_data, bins=30, kde=True, ax=ax7, color='orange')
                ax7.set_title('Total Page Clicks Distribution', fontweight='bold')
                ax7.set_xlabel('Page Clicks')
            
            plt.suptitle('Cohort Summary Dashboard', fontsize=16, fontweight='bold', y=0.995)
            self._save_plot('cohort_summary_dashboard.png')
            
        except Exception as e:
            self._vprint(3, f"⚠️ Error creating cohort dashboard: {e}")
            plt.close()

    def _generate_final_summary_report(self, start_time: datetime):
        """
        Generate final comprehensive summary report for the entire EDA.
        
        Args:
            start_time: Timestamp when EDA started
        """
        self._vprint(2, "\n📋 Generating Final Summary Report...\n")
        
        try:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            summary = {
                'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Execution Time (seconds)': round(duration, 2),
                'Verbosity Level': self.verbosity,
                'Output Directory - Figures': self.fig_dir,
                'Output Directory - Data': self.data_dir,
            }
            
            # Add dataset info if available
            if self.enriched_sessions_df is not None:
                df = self.enriched_sessions_df
                summary.update({
                    'Total Sessions': len(df),
                    'Unique Users': df['user_id'].nunique(),
                    'Unique Sessions': df['session_id'].nunique(),
                    'Date Range Start': df['session_start'].min() if 'session_start' in df.columns else 'N/A',
                    'Date Range End': df['session_start'].max() if 'session_start' in df.columns else 'N/A',
                })
            
            # Count generated files
            try:
                fig_files = len([f for f in os.listdir(self.fig_dir) if f.endswith('.png')])
                data_files = len([f for f in os.listdir(self.data_dir) if f.endswith('.csv')])
                summary['Generated Figures'] = fig_files
                summary['Generated Data Files'] = data_files
            except:  # noqa: E722
                pass
            
            # Convert to DataFrame
            summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=['Value'])
            summary_df.index.name = 'Metric'
            
            # Display and save
            self._display(1, summary_df)
            self._save_data(summary_df.reset_index(), 'final_eda_summary.csv')
            
            # Create execution summary text file
            summary_text_path = os.path.join(self.data_dir, 'eda_execution_summary.txt')
            with open(summary_text_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("TRAVELTIDE EDA - EXECUTION SUMMARY\n")
                f.write("="*80 + "\n\n")
                f.write(f"Execution Date: {summary['Analysis Date']}\n")
                f.write(f"Total Duration: {summary['Execution Time (seconds)']} seconds\n")
                f.write(f"Verbosity Level: {summary['Verbosity Level']}\n\n")
                
                f.write("-"*80 + "\n")
                f.write("PHASES COMPLETED:\n")
                f.write("-"*80 + "\n")
                f.write("✅ Phase 1: Structure & Data Quality Analysis\n")
                f.write("✅ Phase 2: Demographic Analysis\n")
                f.write("✅ Phase 3: Session Behavior Analysis\n")
                f.write("✅ Phase 4: Booking Analysis\n")
                f.write("✅ Phase 5: Flight Path Analysis\n")
                f.write("✅ Phase 6: Cohort Analysis\n\n")
                
                f.write("-"*80 + "\n")
                f.write("OUTPUT LOCATIONS:\n")
                f.write("-"*80 + "\n")
                f.write(f"Figures: {self.fig_dir}\n")
                f.write(f"Data Files: {self.data_dir}\n\n")
                
                if 'Generated Figures' in summary:
                    f.write(f"Total Figures Generated: {summary['Generated Figures']}\n")
                if 'Generated Data Files' in summary:
                    f.write(f"Total Data Files Generated: {summary['Generated Data Files']}\n\n")
                
                f.write("="*80 + "\n")
                f.write("NEXT STEPS:\n")
                f.write("="*80 + "\n")
                f.write("1. Review generated figures in the figures directory\n")
                f.write("2. Analyze cohort data: elena_cohort_complete.csv\n")
                f.write("3. Proceed to K-Means segmentation using prepared cohort data\n")
                f.write("4. Develop personalized perks based on segment characteristics\n\n")
                
            self._vprint(2, f"✅ Final summary saved to: {summary_text_path}\n")
            
        except Exception as e:
            self._vprint(2, f"⚠️ Error generating final summary: {e}")

    # ==================== FLIGHT PATH ANALYSIS ====================
    
    def create_flight_paths(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Creates a GeoDataFrame of flight paths from home to destination airports.
        
        Args:
            df (pd.DataFrame): DataFrame containing airport coordinates (already filtered).
            
        Returns:
            gpd.GeoDataFrame: GeoDataFrame with LineString geometries.
        """
        required_cols = [
            "home_airport_lat", "home_airport_lon",
            "destination_airport_lat", "destination_airport_lon"
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Filter only valid coordinates
        valid_df = df.dropna(subset=required_cols)
        
        if len(valid_df) == 0:
            self._vprint(2, "⚠️ No valid flight coordinates found")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        paths = []
        for _, row in valid_df.iterrows():
            line = LineString([
                (row["home_airport_lon"], row["home_airport_lat"]),
                (row["destination_airport_lon"], row["destination_airport_lat"])
            ])
            paths.append(line)

        self._vprint(2, f"✅ Created {len(paths)} flight paths")
        return gpd.GeoDataFrame(geometry=paths, crs="EPSG:4326")

    def plot_flight_map(
        self,
        df: pd.DataFrame,
        gdf_paths: gpd.GeoDataFrame,
        title: str = "Mapping the World's Flight Paths",
        figsize: tuple = (18, 10),
        save_path: str = None
    ) -> None:
        """
        Plots flight paths and airport locations on a world map.
        
        Args:
            df (pd.DataFrame): DataFrame with airport coordinates (already filtered).
            gdf_paths (gpd.GeoDataFrame): GeoDataFrame of LineStrings.
            title (str): Title of the map.
            figsize (tuple): Size of the figure.
            save_path (str, optional): If provided, saves the figure to this path.
        """
        self._vprint(2, "🗺️ Creating flight map visualization...")

        if len(gdf_paths) == 0:
            self._vprint(2, "⚠️ No flight paths to display")
            return

        fig = plt.figure(figsize=figsize)  # noqa: F841
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Add base map features
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.set_global()

        # Plot flight paths
        gdf_paths.plot(
            ax=ax,
            transform=ccrs.Geodetic(),
            color='crimson',
            linewidth=0.7,
            alpha=0.5
        )

        # Plot airport locations
        ax.scatter(
            df["home_airport_lon"],
            df["home_airport_lat"],
            color='navy',
            s=10,
            transform=ccrs.PlateCarree(),
            label="Origin Airports",
            alpha=0.7
        )
        ax.scatter(
            df["destination_airport_lon"],
            df["destination_airport_lat"],
            color='darkorange',
            s=10,
            transform=ccrs.PlateCarree(),
            label="Destination Airports",
            alpha=0.7
        )

        # Title and legend
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc="lower left", frameon=True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            self._vprint(2, f"📍 Map saved to: {save_path}")

        self._save_plot('flight_paths_map.png', show_plot=(self.verbosity >= 2))

    def plot_top_destinations(
        self,
        df: pd.DataFrame,
        column: str = "destination",
        top_n: int = 10,
        filename: str = "top_destinations.png",
        title: str = "Top Destination Cities",
        xlabel: str = "Number of Bookings",
        ylabel: str = "Destination"
    ) -> None:
        """
        Plots a horizontal bar chart of the top N destination values.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Column to analyze (default: 'destination').
            top_n (int): Number of top entries to display.
            filename (str): Filename for the saved image.
            title (str): Chart title.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
        """
        self._vprint(2, f"📊 Creating top {top_n} {column} destinations plot...")

        if column not in df.columns:
            self._vprint(2, f"⚠️ Column '{column}' not found in DataFrame")
            return

        # Prepare data
        top_values = df[column].value_counts().head(top_n)

        if len(top_values) == 0:
            self._vprint(2, f"⚠️ No data found for column '{column}'")
            return

        # Create plot
        plt.figure(figsize=(10, 6))
        sns.barplot(
            y=top_values.index,
            x=top_values.values,
            palette="viridis"
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()

        self._save_plot(filename, show_plot=(self.verbosity >= 2))

    def plot_and_save_user_booking_distribution(
        self,
        df: pd.DataFrame,
        filename: str = "user_booking_distribution.png"
    ) -> None:
        """
        Creates and saves a horizontal layout of pie charts showing:
        - Top 10 home countries
        - Top 10 home cities
        - Booking type distribution
        
        Args:
            df (pd.DataFrame): Input DataFrame with booking and location info.
            filename (str): Filename for the saved image.
        """
        self._vprint(2, "📊 Creating user booking distribution plot...")

        # Check required columns
        required_cols = ['flight_booked', 'hotel_booked']
        if not all(col in df.columns for col in required_cols):
            self._vprint(2, f"⚠️ Missing required columns: {required_cols}")
            return

        # Count top locations
        home_country_counts = df['home_country'].value_counts().head(10) if 'home_country' in df.columns else pd.Series()
        home_city_counts = df['home_city'].value_counts().head(10) if 'home_city' in df.columns else pd.Series()

        # Classify booking type
        def classify_booking(row):
            if row['flight_booked'] and row['hotel_booked']:
                return "Flight + Hotel"
            elif row['flight_booked']:
                return "Flight Only"
            elif row['hotel_booked']:
                return "Hotel Only"
            else:
                return "No Booking"

        df['booking_type'] = df.apply(classify_booking, axis=1)
        booking_counts = df['booking_type'].value_counts()

        # Create figure layout
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))

        # Pie 1: Home Country
        if len(home_country_counts) > 0:
            axes[0].pie(
                home_country_counts,
                labels=home_country_counts.index,
                autopct='%1.1f%%',
                startangle=140,
                colors=sns.color_palette("pastel", len(home_country_counts))
            )
            axes[0].set_title("🌍 User Distribution by Home Country")
        else:
            axes[0].text(0.5, 0.5, 'No Data', ha='center', va='center')

        # Pie 2: Home City
        if len(home_city_counts) > 0:
            axes[1].pie(
                home_city_counts,
                labels=home_city_counts.index,
                autopct='%1.1f%%',
                startangle=140,
                colors=sns.color_palette("Set2", len(home_city_counts))
            )
            axes[1].set_title("🏙️ User Distribution by Home City")
        else:
            axes[1].text(0.5, 0.5, 'No Data', ha='center', va='center')

        # Pie 3: Booking Type
        axes[2].pie(
            booking_counts,
            labels=booking_counts.index,
            autopct='%1.1f%%',
            startangle=140,
            colors=sns.color_palette("coolwarm", len(booking_counts))
        )
        axes[2].set_title("✈️ Booking Type Distribution")

        # Final polish
        plt.tight_layout(pad=3)

        self._save_plot(filename, show_plot=(self.verbosity >= 2))

    def analyze_flight_paths(self):
        """Analyzes and visualizes flight paths from enriched sessions."""
        self._vprint(1, "\n🔹 PHASE 5.5: FLUGPFAD-ANALYSE\n")

        df = self._load_enriched_sessions()

        # Filter for valid flight data
        flight_cols = [
            "home_airport_lat", "home_airport_lon",
            "destination_airport_lat", "destination_airport_lon"
        ]
        
        # Check if columns exist
        missing_cols = [col for col in flight_cols if col not in df.columns]
        if missing_cols:
            self._vprint(2, f"⚠️ Missing required columns: {missing_cols}")
            self._vprint(2, "Skipping flight path analysis...")
            return

        valid_flights = df.dropna(subset=flight_cols)

        if len(valid_flights) == 0:
            self._vprint(2, "⚠️ Keine gültigen Flugdaten gefunden.")
            return

        self._vprint(2, f"🛫 {len(valid_flights):,} gültige Flugpfade gefunden.")

        # Create flight paths - NOW WITH PARAMETER
        gdf_paths = self.create_flight_paths(valid_flights)

        # Plot flight map - pass both DataFrames
        self.plot_flight_map(valid_flights, gdf_paths)

        # Plot top destinations
        if 'destination_airport' in valid_flights.columns:
            self.plot_top_destinations(
                valid_flights, 
                column='destination_airport', 
                top_n=15,
                filename='top_flight_destinations_map.png',
                title="Top 15 Destination Airports"
            )

        # Plot user booking distribution
        self.plot_and_save_user_booking_distribution(valid_flights)
        
        self._vprint(2, "✅ Flight path analysis completed\n")

    # ==================== VISUALIZATION METHODS (CONTINUED) ====================
    
    def plot_demographics_heatmap(self, users_df: pd.DataFrame):
        """Plot demographics heatmap showing average age."""
        required_cols = ['age_2025', 'gender', 'married', 'has_children']
        if not all(col in users_df.columns for col in required_cols):
            self._vprint(3, f"  ⚠️ Skipping heatmap - missing required columns")  # noqa: F541
            return
            
        try:
            cross_tab = pd.crosstab(
                [users_df['gender'], users_df['married']],
                users_df['has_children'],
                values=users_df['age_2025'],
                aggfunc='mean'
            )
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(cross_tab, annot=True, fmt='.1f', cmap='YlGnBu', 
                       cbar_kws={'label': 'Average Age'})
            plt.title('Average Age by Gender, Marital Status, and Children')
            plt.xlabel('Has Children')
            plt.ylabel('Gender / Married')
            self._save_plot('demographics_heatmap.png')
        except Exception as e:
            self._vprint(3, f"  ⚠️ Error creating heatmap: {e}")

    def plot_demographics_violin(self, users_df: pd.DataFrame):
        """Plot violin plots for age distribution."""
        required_cols = ['age_2025', 'gender', 'married']
        if not all(col in users_df.columns for col in required_cols):
            self._vprint(3, f"  ⚠️ Skipping violin plot - missing required columns")  # noqa: F541
            return
            
        try:
            plt.figure(figsize=(12, 6))
            sns.violinplot(x='gender', y='age_2025', hue='married', 
                          data=users_df, split=True, palette='Set2')
            plt.title('Age Distribution by Gender and Marital Status')
            plt.xlabel('Gender')
            plt.ylabel('Age (2025)')
            plt.legend(title='Married', labels=['Not Married', 'Married'])
            self._save_plot('demographics_violin.png')
        except Exception as e:
            self._vprint(3, f"  ⚠️ Error creating violin plot: {e}")

    def plot_age_distribution(self, users_df: pd.DataFrame):
        """Plot detailed age distribution."""
        if 'age_2025' not in users_df.columns:
            self._vprint(3, "  ⚠️ Skipping age distribution - age_2025 not available")
            return
            
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Overall distribution
            sns.histplot(users_df['age_2025'], bins=30, kde=True, ax=axes[0, 0])
            axes[0, 0].set_title('Overall Age Distribution')
            axes[0, 0].set_xlabel('Age')
            axes[0, 0].set_ylabel('Count')
            
            # By gender
            if 'gender' in users_df.columns:
                for gender in users_df['gender'].unique():
                    if pd.notna(gender):
                        data = users_df[users_df['gender'] == gender]['age_2025']
                        sns.kdeplot(data, label=gender, ax=axes[0, 1])
                axes[0, 1].set_title('Age Distribution by Gender')
                axes[0, 1].set_xlabel('Age')
                axes[0, 1].legend()
            
            # By marital status
            if 'married' in users_df.columns:
                sns.boxplot(data=users_df, x='married', y='age_2025', ax=axes[1, 0])
                axes[1, 0].set_title('Age by Marital Status')
                axes[1, 0].set_xticklabels(['Not Married', 'Married'])
                axes[1, 0].set_ylabel('Age')
            
            # By children status
            if 'has_children' in users_df.columns:
                sns.boxplot(data=users_df, x='has_children', y='age_2025', ax=axes[1, 1])
                axes[1, 1].set_title('Age by Children Status')
                axes[1, 1].set_xticklabels(['No Children', 'Has Children'])
                axes[1, 1].set_ylabel('Age')
            
            plt.suptitle('Comprehensive Age Analysis', fontsize=16, y=0.995)
            self._save_plot('age_distribution_complete.png')
        except Exception as e:
            self._vprint(3, f"  ⚠️ Error creating age distribution: {e}")

    def plot_geographic_distribution(self, users_df: pd.DataFrame):
        """Plot geographic distribution of users."""
        if 'home_country' not in users_df.columns:
            self._vprint(3, "  ⚠️ 'home_country' column not found")
            return
            
        country_counts = users_df['home_country'].value_counts().nlargest(20)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart
        sns.barplot(x=country_counts.values, y=country_counts.index, 
                   palette='viridis', ax=axes[0])
        axes[0].set_title('Top 20 Countries by User Count')
        axes[0].set_xlabel('Number of Users')
        axes[0].set_ylabel('Country')
        
        # Pie chart for top 10
        top10 = country_counts.nlargest(10)
        others = country_counts[10:].sum()
        pie_data = pd.concat([top10, pd.Series({'Others': others})])
        axes[1].pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%')
        axes[1].set_title('Top 10 Countries Distribution')
        
        self._save_plot('geographic_distribution.png')

    def plot_demographics_dashboard(self, users_df: pd.DataFrame):
        """Create comprehensive demographics dashboard."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        try:
            # 1. Gender distribution
            if 'gender' in users_df.columns:
                ax1 = fig.add_subplot(gs[0, 0])
                gender_counts = users_df['gender'].value_counts()
                ax1.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
                ax1.set_title('Gender Distribution')
            
            # 2. Marital status
            if 'married' in users_df.columns:
                ax2 = fig.add_subplot(gs[0, 1])
                married_counts = users_df['married'].value_counts()
                ax2.bar(['Not Married', 'Married'], married_counts.values, color='skyblue')
                ax2.set_title('Marital Status')
                ax2.set_ylabel('Count')
            
            # 3. Children status
            if 'has_children' in users_df.columns:
                ax3 = fig.add_subplot(gs[0, 2])
                children_counts = users_df['has_children'].value_counts()
                ax3.bar(['No Children', 'Has Children'], children_counts.values, color='coral')
                ax3.set_title('Has Children')
                ax3.set_ylabel('Count')
            
            # 4. Age distribution
            if 'age_2025' in users_df.columns:
                ax4 = fig.add_subplot(gs[1, :])
                sns.histplot(users_df['age_2025'], bins=30, kde=True, ax=ax4)
                ax4.set_title('Age Distribution')
                ax4.set_xlabel('Age')
            
            # 5. Gender x Married
            if 'gender' in users_df.columns and 'married' in users_df.columns:
                ax5 = fig.add_subplot(gs[2, 0])
                cross_data = pd.crosstab(users_df['gender'], users_df['married'])
                cross_data.plot(kind='bar', ax=ax5)
                ax5.set_title('Gender by Marital Status')
                ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
                ax5.legend(['Not Married', 'Married'])
            
            # 6. Gender x Children
            if 'gender' in users_df.columns and 'has_children' in users_df.columns:
                ax6 = fig.add_subplot(gs[2, 1])
                cross_data = pd.crosstab(users_df['gender'], users_df['has_children'])
                cross_data.plot(kind='bar', ax=ax6)
                ax6.set_title('Gender by Children Status')
                ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
                ax6.legend(['No Children', 'Has Children'])
            
            # 7. Married x Children
            if 'married' in users_df.columns and 'has_children' in users_df.columns:
                ax7 = fig.add_subplot(gs[2, 2])
                cross_data = pd.crosstab(users_df['married'], users_df['has_children'])
                cross_data.plot(kind='bar', ax=ax7)
                ax7.set_title('Marital Status by Children')
                ax7.set_xticklabels(['Not Married', 'Married'], rotation=0)
                ax7.legend(['No Children', 'Has Children'])
            
            plt.suptitle('Comprehensive Demographics Dashboard', fontsize=16, y=0.995)
            self._save_plot('demographics_dashboard.png')
            
        except Exception as e:
            self._vprint(3, f"  ⚠️ Error creating demographics dashboard: {e}")
            plt.close()

    def plot_session_analysis_dashboard(self, sessions_df: pd.DataFrame):
        """Create comprehensive session analysis dashboard."""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        try:
            # 1. Page clicks distribution
            if 'page_clicks' in sessions_df.columns:
                ax1 = fig.add_subplot(gs[0, :])
                click_data = sessions_df['page_clicks'].clip(upper=50)
                sns.histplot(click_data, bins=50, kde=True, ax=ax1)
                ax1.set_title('Page Clicks Distribution (capped at 50)')
                ax1.set_xlabel('Page Clicks')
            
            # 2. Flight discount usage
            if 'flight_discount' in sessions_df.columns:
                ax2 = fig.add_subplot(gs[1, 0])
                flight_discount = sessions_df['flight_discount'].value_counts()
                ax2.pie(flight_discount.values, labels=['No Discount', 'With Discount'], 
                       autopct='%1.1f%%', colors=['lightblue', 'orange'])
                ax2.set_title('Flight Discount Usage')
            
            # 3. Hotel discount usage
            if 'hotel_discount' in sessions_df.columns:
                ax3 = fig.add_subplot(gs[1, 1])
                hotel_discount = sessions_df['hotel_discount'].value_counts()
                ax3.pie(hotel_discount.values, labels=['No Discount', 'With Discount'], 
                       autopct='%1.1f%%', colors=['lightgreen', 'purple'])
                ax3.set_title('Hotel Discount Usage')
            
            # 4. Sessions over time
            if 'session_start' in sessions_df.columns:
                ax4 = fig.add_subplot(gs[1, 2])
                sessions_df['session_date'] = pd.to_datetime(sessions_df['session_start']).dt.date
                daily_sessions = sessions_df['session_date'].value_counts().sort_index()
                daily_sessions.plot(ax=ax4, color='steelblue')
                ax4.set_title('Daily Session Count')
                ax4.set_xlabel('Date')
                ax4.set_ylabel('Sessions')
                plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
            
            plt.suptitle('Comprehensive Session Analysis Dashboard', fontsize=16, y=0.995)
            self._save_plot('session_analysis_dashboard.png')
            
        except Exception as e:
            self._vprint(3, f"  ⚠️ Error creating session dashboard: {e}")
            plt.close()

    def plot_flight_analysis_dashboard(self, flights_df: pd.DataFrame):
        """Create comprehensive flight analysis dashboard."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        try:
            # 1. Fare distribution
            if 'base_fare_usd' in flights_df.columns:
                ax1 = fig.add_subplot(gs[0, :])
                sns.histplot(flights_df['base_fare_usd'], bins=50, kde=True, ax=ax1)
                ax1.set_title('Flight Fare Distribution')
                ax1.set_xlabel('Base Fare (USD)')
            
            # 2. Top destinations
            if 'destination_airport' in flights_df.columns:
                ax2 = fig.add_subplot(gs[1, 0])
                top_dest = flights_df['destination_airport'].value_counts().head(10)
                top_dest.plot(kind='barh', ax=ax2, color='coral')
                ax2.set_title('Top 10 Destinations')
                ax2.set_xlabel('Number of Flights')
            
            # 3. Top origins
            if 'origin_airport' in flights_df.columns:
                ax3 = fig.add_subplot(gs[1, 1])
                top_origin = flights_df['origin_airport'].value_counts().head(10)
                top_origin.plot(kind='barh', ax=ax3, color='green')
                ax3.set_title('Top 10 Origins')
                ax3.set_xlabel('Number of Flights')
            
            # 4. Cancellation rate
            if 'cancellation' in flights_df.columns:
                ax4 = fig.add_subplot(gs[1, 2])
                cancel_data = flights_df['cancellation'].value_counts()
                ax4.pie(cancel_data.values, labels=['Not Cancelled', 'Cancelled'], 
                       autopct='%1.1f%%', colors=['green', 'red'])
                ax4.set_title('Cancellation Rate')
            
            # 5. Seats distribution
            if 'seats' in flights_df.columns:
                ax5 = fig.add_subplot(gs[2, 0])
                seat_counts = flights_df['seats'].value_counts().sort_index().head(10)
                seat_counts.plot(kind='bar', ax=ax5, color='skyblue')
                ax5.set_title('Seats per Booking')
                ax5.set_xlabel('Number of Seats')
                ax5.set_ylabel('Count')
            
            # 6. Return flight ratio
            if 'return_flight_booked' in flights_df.columns:
                ax6 = fig.add_subplot(gs[2, 1])
                return_counts = flights_df['return_flight_booked'].value_counts()
                ax6.pie(return_counts.values, labels=['One-way', 'Round-trip'], 
                       autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
                ax6.set_title('Flight Type Distribution')
            
            # 7. Discount analysis
            if 'flight_discount_amount' in flights_df.columns:
                ax7 = fig.add_subplot(gs[2, 2])
                discount_data = flights_df[flights_df['flight_discount_amount'] > 0]['flight_discount_amount']
                if len(discount_data) > 0:
                    sns.histplot(discount_data, bins=30, kde=True, ax=ax7)
                    ax7.set_title('Discount Amount Distribution')
                    ax7.set_xlabel('Discount (USD)')
            
            plt.suptitle('Comprehensive Flight Analysis Dashboard', fontsize=16, y=0.995)
            self._save_plot('flight_analysis_dashboard.png')
            
        except Exception as e:
            self._vprint(3, f"  ⚠️ Error creating flight dashboard: {e}")
            plt.close()

    def plot_hotel_analysis_dashboard(self, hotels_df: pd.DataFrame):
        """Create comprehensive hotel analysis dashboard."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        try:
            # 1. Price distribution
            if 'hotel_price_per_room_night_usd' in hotels_df.columns:
                ax1 = fig.add_subplot(gs[0, :])
                sns.histplot(hotels_df['hotel_price_per_room_night_usd'], bins=50, kde=True, ax=ax1)
                ax1.set_title('Hotel Price per Room Night Distribution')
                ax1.set_xlabel('Price per Room Night (USD)')
            
            # 2. Nights distribution
            if 'nights' in hotels_df.columns:
                ax2 = fig.add_subplot(gs[1, 0])
                night_counts = hotels_df['nights'].value_counts().sort_index().head(15)
                night_counts.plot(kind='bar', ax=ax2, color='skyblue')
                ax2.set_title('Booking by Number of Nights')
                ax2.set_xlabel('Nights')
                ax2.set_ylabel('Count')
            
            # 3. Rooms distribution
            if 'rooms' in hotels_df.columns:
                ax3 = fig.add_subplot(gs[1, 1])
                room_counts = hotels_df['rooms'].value_counts().sort_index().head(10)
                room_counts.plot(kind='bar', ax=ax3, color='coral')
                ax3.set_title('Booking by Number of Rooms')
                ax3.set_xlabel('Rooms')
                ax3.set_ylabel('Count')
            
            # 4. Cancellation rate
            if 'cancellation' in hotels_df.columns:
                ax4 = fig.add_subplot(gs[1, 2])
                cancel_data = hotels_df['cancellation'].value_counts()
                ax4.pie(cancel_data.values, labels=['Not Cancelled', 'Cancelled'], 
                       autopct='%1.1f%%', colors=['green', 'red'])
                ax4.set_title('Cancellation Rate')
            
            # 5. Average price by nights
            if 'nights' in hotels_df.columns and 'hotel_price_per_room_night_usd' in hotels_df.columns:
                ax5 = fig.add_subplot(gs[2, 0])
                price_by_nights = hotels_df.groupby('nights')['hotel_price_per_room_night_usd'].mean().head(15)
                price_by_nights.plot(kind='line', marker='o', ax=ax5)
                ax5.set_title('Average Price by Nights')
                ax5.set_xlabel('Nights')
                ax5.set_ylabel('Avg Price (USD)')
            
            # 6. Average price by rooms
            if 'rooms' in hotels_df.columns and 'hotel_price_per_room_night_usd' in hotels_df.columns:
                ax6 = fig.add_subplot(gs[2, 1])
                price_by_rooms = hotels_df.groupby('rooms')['hotel_price_per_room_night_usd'].mean().head(10)
                price_by_rooms.plot(kind='line', marker='s', ax=ax6, color='coral')
                ax6.set_title('Average Price by Rooms')
                ax6.set_xlabel('Rooms')
                ax6.set_ylabel('Avg Price (USD)')
            
            # 7. Discount analysis
            if 'hotel_discount_amount' in hotels_df.columns:
                ax7 = fig.add_subplot(gs[2, 2])
                discount_data = hotels_df[hotels_df['hotel_discount_amount'] > 0]['hotel_discount_amount']
                if len(discount_data) > 0:
                    sns.histplot(discount_data, bins=30, kde=True, ax=ax7)
                    ax7.set_title('Discount Amount Distribution')
                    ax7.set_xlabel('Discount (USD)')
            
            plt.suptitle('Comprehensive Hotel Analysis Dashboard', fontsize=16, y=0.995)
            self._save_plot('hotel_analysis_dashboard.png')
            
        except Exception as e:
            self._vprint(3, f"  ⚠️ Error creating hotel dashboard: {e}")
            plt.close()
           

  
    # ==================== MAIN WORKFLOW ====================
    
    def run_full_eda(self):
        """Execute complete exploratory data analysis workflow."""
        start_time = datetime.now()

        self._vprint(1, "\n" + "="*80)
        self._vprint(1, f"VOLLSTÄNDIGE EXPLORATIVE DATENANALYSE (EDA)")  # noqa: F541
        self._vprint(1, f"TravelTide Rewards Programm")  # noqa: F541
        self._vprint(1, f"Verbosity Level: {self.verbosity}")
        self._vprint(1, f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._vprint(1, "="*80)

        try:
            # PHASE 1: Structure & Quality
            self.analyze_enriched_sessions_structure()

            # PHASE 2: Demographics
            self.analyze_user_demographics()

            # PHASE 3: Session Behavior
            self.analyze_session_behavior()

            # PHASE 4: Bookings
            self.analyze_bookings_comprehensive()

            # PHASE 5: Flight Paths
            self.analyze_flight_paths()

            # PHASE 6: Cohort Analysis
            cohort_df = self.get_elena_cohort_summary()

            if not cohort_df.empty:
                # Create summary report
                self._create_cohort_summary_report(cohort_df)

            # Generate final summary
            self._generate_final_summary_report(start_time)

            self._vprint(1, "\n" + "="*80)
            self._vprint(1, "🎉 EDA ERFOLGREICH ABGESCHLOSSEN")
            self._vprint(1, "="*80)
            self._vprint(1, f"\n📁 All results saved to:")  # noqa: F541
            self._vprint(1, f"   Figures: {self.fig_dir}")
            self._vprint(1, f"   Data: {self.data_dir}")
            self._vprint(1, "\n✨ Die Daten sind nun für die K-Means Segmentierung bereit.")
            self._vprint(1, "="*80 + "\n")

        except Exception as e:
            self._vprint(1, f"\n❌ FEHLER im EDA-Workflow: {e}")
            if self.verbosity >= 3:
                import traceback
                self._vprint(3, "\n" + "="*80)
                self._vprint(3, "FULL ERROR TRACEBACK:")
                self._vprint(3, "="*80)
                self._vprint(3, traceback.format_exc())
                self._vprint(3, "="*80)

        finally:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            self._vprint(1, f"\n⏱️ Total execution time: {duration:.2f} seconds")
            self.close()

