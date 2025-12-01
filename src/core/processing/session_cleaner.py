# src/core/processing/outlier_handler.py

import os
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from IPython.display import display  # type: ignore
from scipy import stats  # type: ignore

from src.utils import get_path  # type: ignore

class SessionCleaner:
    """Class to clean session data and handle outliers using various statistical methods."""

    def __init__(self, verbosity=1, path_type="cleaner"):
        self.verbosity = verbosity
        self.output_dir = os.path.join(get_path(path_type))
        self._vprint(1, f"SessionCleaner initialized with verbosity={verbosity}")
        self._vprint(2, f"Output directory resolved to: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)

    def _vprint(self, level, message):
        if self.verbosity >= level:
            print(message)

    def _save_plot(self, filename):
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300)
        self._vprint(2, f"Plot saved to {path}")

    def _get_filtered_numeric_columns(self, df: pd.DataFrame):
        """Returns numeric columns excluding those with 'id', 'lat', or 'lon' in their names."""
        exclude_keywords = ['id', 'lat', 'lon']
        numeric_cols = df.select_dtypes(include=np.number).columns
        return [
            col for col in numeric_cols
            if not any(keyword in col.lower() for keyword in exclude_keywords)
        ]

    def remove_canceled_trips(self, sessions_df: pd.DataFrame, cancel_trip_ids_df: pd.DataFrame, filename="sessions_not_canceled_trips.csv") -> pd.DataFrame:
        self._vprint(1, "\n🧹 Filtering out canceled trips from session data")
        self._vprint(2, f"Gesamtzahl stornierter Reisen: {cancel_trip_ids_df.shape[0]:,}")
        
        canceled_trip_ids  = cancel_trip_ids_df["trip_id"].tolist()
        df_canceled_trips = sessions_df[sessions_df['trip_id'].isin(canceled_trip_ids)]
        df_canceled_trips = df_canceled_trips.drop_duplicates(subset=['trip_id'])
        print(f"We have in Total {df_canceled_trips.shape[0]} canceld Trips after our Cohor Filtering")

        df_cleaned = sessions_df.dropna(subset=["trip_id"])
        self._vprint(2, f"Entfernte Zeilen ohne trip_id: {sessions_df.shape[0] - df_cleaned.shape[0]:,}")

        df_cleaned = df_cleaned[~df_cleaned["trip_id"].isin(canceled_trip_ids)]
        self._vprint(2, f"✅ Nicht stornierte Reisen übrig: {df_cleaned.shape[0]:,}")
        
        # A single trip (trip_id) can be associated with multiple sessions (session_id), 
        # which results in incorrect double counting when directly aggregating trip-related metrics 
        # like revenue or booking volume. To guarantee accurate and unbiased analytical results, 
        # it is crucial to always use COUNT(DISTINCT trip_id) when counting bookings.
        df_cleaned = df_cleaned.drop_duplicates(subset=['trip_id'])

        save_path = os.path.join(get_path("processed"), filename)
        df_cleaned.to_csv(save_path, index=False)
        self._vprint(1, f"Bereinigte Sessions gespeichert unter: {save_path}")

        return df_cleaned

    def detect_outlier_columns(self, df: pd.DataFrame, method='iqr', threshold=1.5, 
                           extreme_threshold=3.0, min_outlier_percentage=1.0,
                           min_coefficient_of_variation=0.3, skip_low_variance=True):
        """
        Detect columns with extreme outliers, filtering out low-variance columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        method : str
            'iqr' or 'zscore'
        threshold : float
            Standard threshold for outlier detection (1.5 for IQR, 3 for z-score)
        extreme_threshold : float
            Threshold for EXTREME outliers (3.0 for IQR, 4-5 for z-score)
        min_outlier_percentage : float
            Minimum percentage of extreme outliers to flag column (default: 1%)
        min_coefficient_of_variation : float
            Minimum CV (std/mean) to consider column (default: 0.3 = 30%)
            Skips columns where values are too uniform
        skip_low_variance : bool
            Whether to skip low-variance columns like seats/rooms (default: True)
        
        Returns:
        --------
        tuple : (outlier_info dict, list of extreme outlier columns)
        """
        outlier_info = {}
        filtered_cols = self._get_filtered_numeric_columns(df)
        skipped_cols = []

        for col in filtered_cols:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue

            # Check coefficient of variation to skip low-variance columns
            if skip_low_variance:
                mean_val = col_data.mean()
                std_val = col_data.std()
                cv = std_val / abs(mean_val) if mean_val != 0 else 0
                
                if cv < min_coefficient_of_variation:
                    skipped_cols.append(col)
                    self._vprint(2, f"⏭️  Skipping '{col}' - low variance (CV={cv:.3f}, std={std_val:.2f})")
                    continue

            if method == 'iqr':
                Q1, Q3 = np.percentile(col_data, [25, 75])
                IQR = Q3 - Q1
                
                # Skip if IQR is effectively zero (all values nearly identical)
                if IQR < 1e-6:
                    skipped_cols.append(col)
                    self._vprint(2, f"⏭️  Skipping '{col}' - IQR=0 (all values identical)")
                    continue
                
                # Extreme outlier bounds
                extreme_lower = Q1 - extreme_threshold * IQR
                extreme_upper = Q3 + extreme_threshold * IQR
                extreme_mask = (col_data < extreme_lower) | (col_data > extreme_upper)
                extreme_count = extreme_mask.sum()
                
                # Normal outliers for reference
                normal_lower = Q1 - threshold * IQR
                normal_upper = Q3 + threshold * IQR
                normal_mask = (col_data < normal_lower) | (col_data > normal_upper)
                normal_count = normal_mask.sum()
                
                # Calculate max deviation ratio
                max_deviation = max(
                    abs(col_data.min() - Q1) / IQR,
                    abs(col_data.max() - Q3) / IQR
                )

            elif method == 'zscore':
                mean_val = col_data.mean()
                std_val = col_data.std()
                
                # Skip if std is effectively zero
                if std_val < 1e-6:
                    skipped_cols.append(col)
                    self._vprint(2, f"⏭️  Skipping '{col}' - std≈0 (all values identical)")
                    continue
                
                z = np.abs((col_data - mean_val) / std_val)
                
                # Extreme outliers
                extreme_count = (z > extreme_threshold).sum()
                
                # Normal outliers for reference
                normal_count = (z > threshold).sum()
                
                # Max deviation
                max_deviation = z.max()

            else:
                raise ValueError("Method must be 'iqr' or 'zscore'.")

            # Calculate percentages
            extreme_percentage = (extreme_count / len(col_data)) * 100
            
            # Flag column if:
            # 1. Has enough extreme outliers (percentage criterion), OR
            # 2. Has very extreme max deviation (even if rare)
            flag_column = (
                extreme_percentage >= min_outlier_percentage or
                (extreme_count > 0 and max_deviation >= extreme_threshold * 1.5)
            )
            
            if flag_column:
                outlier_info[col] = {
                    'extreme_outlier_count': int(extreme_count),
                    'extreme_percentage': round(extreme_percentage, 2),
                    'normal_outlier_count': int(normal_count),
                    'normal_percentage': round((normal_count / len(col_data)) * 100, 2),
                    'max_deviation': round(max_deviation, 2),
                    'total_values': len(col_data),
                    'cv': round(col_data.std() / abs(col_data.mean()) if col_data.mean() != 0 else 0, 3)
                }

        extreme_outlier_columns = list(outlier_info.keys())
        
        self._vprint(1, f"\n🎯 Detected {len(extreme_outlier_columns)} columns with EXTREME outliers")
        self._vprint(2, f"Criteria: {method.upper()} extreme_threshold={extreme_threshold}, min {min_outlier_percentage}% outliers")
        if skipped_cols:
            self._vprint(2, f"📊 Skipped {len(skipped_cols)} low-variance columns: {skipped_cols}")
        
        return outlier_info, extreme_outlier_columns

    def handle_outliers(self, df, column, method='iqr', action='remove', clip_value=None, z_thresh=3):
        self._vprint(1, f"\nHandling outliers in column: '{column}'")
        self._vprint(2, f"Method: {method}, Action: {action}, Threshold: {z_thresh}")

        df = df.copy()

        if column == 'nights':
            self._vprint(2, "⚠️ Replacing 'nights' values ≤ 0 with 1")
            df[column] = df[column].apply(lambda x: 1 if x <= 0 else x)
            return df

        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - z_thresh * IQR
            upper_bound = Q3 + z_thresh * IQR
            mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            mask = z_scores < z_thresh
            mask = pd.Series(mask, index=df[column].dropna().index)
            mask = df.index.isin(mask[mask].index)

        else:
            raise ValueError("❌ Method must be 'iqr' or 'zscore'")

        if action == 'remove':
            df_cleaned = df[mask]
        elif action == 'clip':
            if clip_value is None:
                clip_value = (lower_bound, upper_bound)
                self._vprint(2, f"Auto-calculated clip bounds: {clip_value}")
            df[column] = df[column].clip(lower=clip_value[0], upper=clip_value[1])
            df_cleaned = df
        elif action == 'transform':
            df[column] = np.log1p(df[column])
            df_cleaned = df
        else:
            raise ValueError("❌ Action must be 'remove', 'clip', or 'transform'")

        self._vprint(2, f"✅ Outlier handling completed for '{column}'")
        return df_cleaned

    def plot_outlier_comparison(self, df, column, method='iqr', action='clip', clip_value=None, z_thresh=3, figsize=(12, 10)):
        self._vprint(1, f"\nGenerating outlier comparison plots for '{column}'")
        df_cleaned = self.handle_outliers(df, column, method=method, action=action, clip_value=clip_value, z_thresh=z_thresh)

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        sns.boxplot(x=df[column], ax=axes[0, 0], color='lightcoral')
        axes[0, 0].set_title(f'{column} Boxplot (Original)')
        axes[0, 0].set_xlabel(column)

        sns.histplot(df[column].dropna(), kde=True, ax=axes[0, 1], color='darkred')
        axes[0, 1].set_title(f'{column} Histogram + KDE (Original)')
        axes[0, 1].set_xlabel(column)

        sns.boxplot(x=df_cleaned[column], ax=axes[1, 0], color='lightgreen')
        axes[1, 0].set_title(f'{column} Boxplot (Cleaned)')
        axes[1, 0].set_xlabel(column)

        sns.histplot(df_cleaned[column].dropna(), kde=True, ax=axes[1, 1], color='darkgreen')
        axes[1, 1].set_title(f'{column} Histogram + KDE (Cleaned)')
        axes[1, 1].set_xlabel(column)

        plt.tight_layout()
        self._save_plot(f'{column}_outlier_comparison_4panel.png')
        plt.show()
        self._vprint(2, f"✅ 4-panel comparison plotted for '{column}'")
        return df_cleaned

    def describe_columns(self, df, columns=None):
        if columns is None:
            columns = self._get_filtered_numeric_columns(df)

        self._vprint(1, f"\nGenerating descriptive statistics for columns: {columns}")
        summary_stats = df[columns].describe().T
        self._vprint(2, "Summary statistics:")
        display(summary_stats)
        return summary_stats

