# scr/core/processing/load_data.py

# type: ignore
import os
import sys
from typing import List

import pandas as pd  # type: ignore
from IPython.display import display  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns

# Add core module to path
cwd = os.getcwd()
project_root = os.path.abspath(os.path.join(cwd, "..", ".."))
sys.path.insert(0, project_root)

from src.db import Database  # noqa: E402
from src.utils import (  # noqa: E402
    raw_data_path, 
    processed_data_path, 
    feature_processed_path,
    segmentation_processed_path,
    sql_path, 
    pca_processed_path,
    non_machine_learning_path,
    ml_model_path,
    get_path
    )


class DataLoader:
    def __init__(self, db: Database = None):
        """
        Initializes the DataLoader with an optional Database instance.
        """
        self.db = db or Database()

    def _get_path(self, data_type: str, table_name: str) -> tuple[str, str]:
        """
        Resolves the file path for a given data type and table name.
        """
        if data_type == "raw":
            path = os.path.join(raw_data_path, f"{table_name}.csv")
        elif data_type == "processed":
            path = os.path.join(processed_data_path, f"{table_name}.csv")
        elif data_type == "sql":
            path = os.path.join(sql_path, f"{table_name}.sql")
        elif data_type == "feature":
            path = os.path.join(feature_processed_path, f"{table_name}.csv")
        elif data_type == "segment":
            path = os.path.join(segmentation_processed_path, f"{table_name}.csv")
        elif data_type == "pca":
            path = os.path.join(pca_processed_path, f"{table_name}.csv")
        elif data_type == "non_ml":
            path = os.path.join(non_machine_learning_path, f"{table_name}.csv")
        elif data_type == "model":
            path = os.path.join(ml_model_path, f"{table_name}.csv")
        elif data_type == "kmeans":
            path = os.path.join(get_path('kmeans'), f"{table_name}.csv")
        elif data_type == "dbscan":
            path = os.path.join(get_path('dbscan'), f"{table_name}.csv")
        else:
            raise ValueError(f"❌ Ungültiger Datentyp: '{data_type}'. Erlaubt sind 'raw', 'processed', 'sql'.")
        return path, data_type

    def load_table(self, data_type: str, table_name: str, show_table_display: bool = False, is_session_base:bool=True) -> pd.DataFrame:
        """
        Loads a table from SQL, CSV, or directly from the database.

        Args:
            data_type (str): One of 'raw', 'processed', or 'sql'.
            table_name (str): Name of the table or file.
            show_table_display (bool): Whether to display a sample of the DataFrame.

        Returns:
            pd.DataFrame: Loaded data.
        """
        file_path, resolved_type = self._get_path(data_type, table_name)
        
        if resolved_type == "sql" and os.path.exists(file_path):
            print(f"📄 Lade Tabelle '{table_name}' aus SQL-Datei: {file_path}")
            df = self.db.execute_sql_file(file_path)
            print(f"✅ SQL-Abfrage erfolgreich. Zeilen: {len(df)}")

            if is_session_base:
                new_csv_path = os.path.join(raw_data_path, f"{table_name}.csv")
            else:
                new_csv_path = os.path.join(processed_data_path, f"{table_name}.csv")
            df.to_csv(new_csv_path, index=False)
            print(f"💾 Gespeichert unter: {new_csv_path}")

        elif resolved_type in ["raw", "processed", "feature", "segment", "pca", "non_ml", "model", "dbscan", "kmeans"] and os.path.exists(file_path):
            print(f"📁 Lade Tabelle '{table_name}' aus CSV: {file_path}")
            df = pd.read_csv(file_path)
            print(f"✅ CSV geladen. Zeilen: {len(df)}")

        else:
            print(f"🌐 Lade Tabelle '{table_name}' direkt aus der Datenbank...")
            print(f"path to file was: {file_path}")
            df = self.db.execute_query(f"SELECT * FROM {table_name};")
            print(f"✅ Datenbankabfrage erfolgreich. Zeilen: {len(df)}")

            if not df.empty and resolved_type in ["raw", "processed"]:
                df.to_csv(file_path, index=False)
                print(f"💾 Gespeichert unter: {file_path}")
            elif df.empty:
                print(f"⚠️ Keine Daten gefunden für Tabelle '{table_name}'")

        if not df.empty and show_table_display:
            display(df.sample(min(100, len(df))))
        return df

    def load_custom_query(self, query: str) -> pd.DataFrame:
        """
        Executes a custom SQL query and returns the result.

        Args:
            query (str): SQL query string.

        Returns:
            pd.DataFrame: Query result.
        """
        print(" Führe benutzerdefinierte SQL-Abfrage aus...")
        df = self.db.execute_query(query)
        if not df.empty:
            display(df.sample(min(100, len(df))))
        else:
            print("⚠️ Keine Ergebnisse für diese Abfrage.")
        return df
    
    def show_nulls(self, df: pd.DataFrame) -> None:
        """
        Displays null counts and percentages per column.
        """
        print(" Missing Values Summary:")
        nulls = pd.DataFrame({
            "null_count": df.isnull().sum(),
            "null_percent": (df.isnull().sum() / len(df)) * 100
        }).sort_values(by="null_count", ascending=False)
        display(nulls.round(2))
    
    def plot_missing_values(self, df: pd.DataFrame, save_path:str=None) -> None:
        """
        Plot missing value percentages per column.
        """
        # Calculate missing percentages
        nulls = pd.DataFrame({
            "missing_count": df.isnull().sum(),
            "missing_percent": (df.isnull().sum() / len(df)) * 100
        }).sort_values(by="null_count", ascending=False)
        
        nulls = nulls[nulls["missing_percent"] > 0]

        print("=== Missing Values Heatmap ===")
        plt.figure(figsize=(10, max(6, len(nulls) * 0.3)))
        sns.heatmap(nulls.T, cmap="Reds", annot=True, fmt=".1f", cbar=True)
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)
        plt.title("Feature Missingness (%)", fontsize=14)
        
      
        save_path = save_path or os.path.join(get_path("loader"), "missing_values.png")
        plt.savefig(save_path, dpi=300)
           
        plt.show()

    def show_dtypes(self, df: pd.DataFrame) -> None:
        """
        Displays data types of each column.
        """
        print("🔤 Data Types:")
        dtypes = pd.DataFrame(df.dtypes, columns=["dtype"])
        display(dtypes)

    def format_dates(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Converts specified columns to datetime format and prints status.
        """
        print(" Formatting Date Columns:")
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                print(f"✅ Converted: {col}")
            else:
                print(f"⚠️ Column not found: {col}")
        return df

    def show_summary_stats(self, df: pd.DataFrame) -> None:
        """
        Displays basic summary statistics.
        """
        print(" Summary Statistics:")
        display(df.describe().transpose().round(2))

    def show_sample(self, df: pd.DataFrame, n: int = 5, transpose: bool = False) -> None:
        """
        Displays a sample of the DataFrame.

        Parameters:
        - df: DataFrame to sample.
        - n: Number of rows to display.
        - transpose: If True, transposes the sample output.
        """
        sample = df.sample(min(n, len(df)))
        if transpose:
            display(sample.T)
        else:
            display(sample)

    def show_unique_counts(self, df: pd.DataFrame) -> None:
        """
        Displays number of unique values per column.
        """
        print(" Unique Value Counts:")
        unique_counts = pd.DataFrame({
            "unique_count": df.nunique()
        }).sort_values(by="unique_count", ascending=False)
        display(unique_counts)
    
    def generate_outlier_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a table with min, 25%, 75%, max, and custom outlier notes for selected columns.
        """
        columns_to_check = {
            "page_clicks": "✅ Very high max — likely outliers above 100 clicks",
            "flight_discount_amount": "✅ Values > 0.35 may be outliers",
            "hotel_discount_amount": "✅ Values > 0.3 may be outliers",
            "base_fare_usd": "🚨 Extreme outlier — fares above $1500 likely anomalous",
            "checked_bags": "✅ More than 3 bags is unusual",
            "seats": "✅ More than 3 seats per booking is rare",
            "nights": "🚨 Negative nights = data error; >20 nights = outlier",
            "rooms": "✅ More than 2 rooms may be outliers",
            "hotel_price_per_room_night_usd": "✅ Prices > $500 are likely outliers",
        }

        summary = []

        for col, note in columns_to_check.items():
            if col in df.columns:
                stats = df[col].describe(percentiles=[0.25, 0.75])
                summary.append({
                    "Column": col,
                    "Min": round(stats["min"], 2),
                    "25%": round(stats["25%"], 2),
                    "75%": round(stats["75%"], 2),
                    "Max": round(stats["max"], 2),
                    "Outlier Notes": note
                })

        summary_df = pd.DataFrame(summary)
        display(summary_df)
        return summary_df
    
    def plot_missingness_combined(self, df: pd.DataFrame) -> None:
        """
        Plot missingness overview (bar chart by category) and detailed heatmap of features in one diagram.
        Optionally save the figure to a file path.
        """

        # --- Calculate missing counts and percentages ---
        nulls = pd.DataFrame({
            "missing_count": df.isnull().sum(),
            "missing_percent": (df.isnull().sum() / len(df)) * 100
        }).sort_values(by="missing_count", ascending=False)

        # Keep only columns with missing values
        nulls = nulls[nulls["missing_percent"] > 0]

        # --- Categorize features by missingness thresholds ---
        def categorize(p):
            if p >= 60:
                return "High (≥60%)"
            elif p >= 8:
                return "Moderate (8–14%)"
            elif p > 0:
                return "Low (<1%)"
            else:
                return "None (0%)"

        nulls["Category"] = nulls["missing_percent"].apply(categorize)

        # Average missingness per category
        cat_df = nulls.groupby("Category")["missing_percent"].mean().reset_index()

        # --- Plot side by side ---
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # Bar chart by category
        sns.barplot(
            data=cat_df, 
            x="Category", 
            y="missing_percent", 
            ax=axes[0], 
            hue="missing_percent",
            palette="viridis")
        axes[0].set_title("Average Missingness by Category", fontsize=14)
        axes[0].set_ylabel("Missing %")
        axes[0].set_xlabel("")
        axes[0].tick_params(axis="x", rotation=30)

        # Heatmap of individual features
        sns.heatmap(
            nulls[["missing_percent"]], 
            cmap="Reds", 
            annot=True, 
            fmt=".1f", 
            ax=axes[1], 
            cbar=True)
        axes[1].set_title("Feature-level Missingness (%)", fontsize=14)
        axes[1].set_ylabel("")
        axes[1].set_xlabel("Feature")
        axes[1].tick_params(axis="x", rotation=90)

        plt.tight_layout()

        # Save if path provided
        loader_base_path = os.path.join(get_path("loader"))
        os.makedirs(loader_base_path, exist_ok=True)
        save_path = os.path.join(loader_base_path, "missing_values_avg.png")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Missingness diagram saved to {save_path}")

        plt.show()

        
    
 

    def generate_session_eda(self, df: pd.DataFrame) -> None:
        """
        Generate and save exploratory visualizations for TravelTide session data.

        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame containing TravelTide session data.

        Saves:
        ------
        - session_distributions.png : Distribution plots of key numeric variables
        - booking_discounts.png     : Count plots of booking and discount statuses
        - relationships.png         : Scatter and box plots showing fare relationships

        Notes:
        ------
        - Figures are saved in `reports/eda/figures` via `get_path("eda")`
        - Missing directories are created automatically
        - Plots are displayed interactively and saved as PNGs
        """

        # --- Setup ---
        eda_base_path = os.path.join(get_path("loader"))
        os.makedirs(eda_base_path, exist_ok=True)

        sns.set_style("whitegrid")
        plt.style.use("ggplot")

        print("\n📊 Generating EDA Visualizations for TravelTide Session Data")

        # ========== 1. Distributions ==========
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Distribution of Session Metrics', fontsize=16)

        dist_features = [
            ('page_clicks', 'skyblue', 'Page Clicks'),
            ('nights', 'lightcoral', 'Nights Booked'),
            ('base_fare_usd', 'lightgreen', 'Base Fare (USD)'),
            ('hotel_price_per_room_night_usd', 'orchid', 'Hotel Price per Room Night (USD)')
        ]

        for ax, (col, color, title) in zip(axes.flat, dist_features):
            sns.histplot(df[col], bins=20, kde=True, ax=ax, color=color)
            ax.set_title(f'Distribution of {title}')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        dist_path = os.path.join(eda_base_path, "session_distributions.png")
        plt.savefig(dist_path, dpi=300)
        plt.show()
        print(f"✅ Saved: {dist_path}")

        # ========== 2. Booking & Discount Statuses ==========
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Booking and Discount Statuses', fontsize=16)

        booked_df = df[['flight_booked', 'hotel_booked']].melt(var_name='Booking Type', value_name='Booked')
        sns.countplot(x='Booking Type', hue='Booked', data=booked_df, ax=axes[0])
        axes[0].set_title('Flight vs. Hotel Booked')

        discount_df = df[['flight_discount', 'hotel_discount']].melt(var_name='Discount Type', value_name='Applied')
        sns.countplot(x='Discount Type', hue='Applied', data=discount_df, ax=axes[1])
        axes[1].set_title('Flight vs. Hotel Discount Applied')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        cat_path = os.path.join(eda_base_path, "booking_discounts.png")
        plt.savefig(cat_path, dpi=300)
        plt.show()
        print(f"✅ Saved: {cat_path}")

        # ========== 3. Relationships ==========
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Relationships in Session Data', fontsize=16)

        sns.scatterplot(x='page_clicks', y='base_fare_usd', data=df, ax=axes[0], color='steelblue', alpha=0.7)
        axes[0].set_title('Page Clicks vs. Base Fare (USD)')

        # Fixing FutureWarning: assign x to hue and disable legend
        sns.boxplot(y='base_fare_usd', x='flight_booked', hue='flight_booked',
                    data=df, ax=axes[1], showfliers=False, palette='Set2', legend=False)
        axes[1].set_title('Base Fare by Flight Booked (Outliers Removed)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        rel_path = os.path.join(eda_base_path, "relationships.png")
        plt.savefig(rel_path, dpi=300)
        plt.show()
        print(f"✅ Saved: {rel_path}")

        print("\n🎉 All TravelTide EDA visualizations successfully generated and saved!")

