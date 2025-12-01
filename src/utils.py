# src/utils.py
# type: ignore

import os
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
import plotly.io as pio  # type: ignore
import plotly.graph_objects as go  # type: ignore
import joblib  # type: ignore
import yaml  # type: ignore
from typing import Any, Callable, Dict, List


# ============================================================
# 📁 DIRECTORY MANAGEMENT
# ============================================================

# Absolute path to this file
current_file = os.path.abspath(__file__)

# Project root = 2 levels above (src/core/utils.py → src → project)
project_root = os.path.dirname(os.path.dirname(current_file))

# Base src directory (src/)
base_path = os.path.dirname(project_root)

# --- Project-level paths ---
data_path = os.path.join(project_root, "data")
reports_path = os.path.join(project_root, "reports")
config_path = os.path.join(project_root, "config")

# --- Data directories ---
csv_path = os.path.join(data_path, "csv")
sql_path = os.path.join(data_path, "sql")

raw_data_path = os.path.join(csv_path, "raw")
processed_data_path = os.path.join(csv_path, "processed")

# processed subfolders
feature_processed_path = os.path.join(processed_data_path, "features")
segmentation_processed_path = os.path.join(processed_data_path, "segment")
pca_processed_path = os.path.join(segmentation_processed_path, "pca")
non_machine_learning_path= os.path.join(segmentation_processed_path, "non_ml")
ml_model_path  = os.path.join(segmentation_processed_path, "ml_model")
kmeans_path =os.path.join(ml_model_path, 'kmeans')
dbscan_path =os.path.join(ml_model_path, 'dbacan')
ab_test_path = os.path.join(segmentation_processed_path, 'ab_test')
classifier_path = os.path.join(segmentation_processed_path, 'classifier')

# --- Reports ---
eda_path = os.path.join(reports_path, "eda")
load_data_path = os.path.join(reports_path, "loader")
cleaner_path =os.path.join(reports_path, "cleaner")
feature_path = os.path.join(reports_path, "features")
segment_reports_path = os.path.join(reports_path, "segment")

# ML model output dir
model_path = os.path.join(segment_reports_path, "model")
non_ml_path = os.path.join(segment_reports_path, "non_ml")




# ============================================================
# ⚙️ CONFIG UTILITIES
# ============================================================

def load_config(config_file: str = None) -> dict:
    """
    Load the main YAML config from the config/ directory.
    """
    default_path = os.path.join(config_path, "config.yaml")
    final_path = config_file or default_path

    try:
        with open(final_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"❌ Failed to load config '{final_path}': {e}")
        return {}


def load_yaml(path: str) -> Dict[str, Any]:
    """General YAML loader with validation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ YAML file not found: {path}")

    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"❌ YAML file empty: {path}")

        print(f"✅ Loaded YAML: {path}")
        return config

    except yaml.YAMLError as e:
        raise ValueError(f"❌ Invalid YAML in {path}: {e}")


# ============================================================
# 💾 MODEL PERSISTENCE
# ============================================================

def save_model(model: Any, model_name: str, algorithm: str = "generic") -> str:
    """
    Save a trained model inside processed/models/<algorithm>/
    """
    model_dir = os.path.join(processed_data_path, "models", algorithm)
    os.makedirs(model_dir, exist_ok=True)

    file_path = os.path.join(model_dir, f"{model_name}.pkl")

    try:
        joblib.dump(model, file_path)
        print(f"✅ Model saved: {file_path}")
        return file_path
    except Exception as e:
        print(f"❌ Failed to save model '{model_name}': {e}")
        return ""


def load_model(model_name: str, algorithm: str = "generic") -> Any:
    """
    Load a model previously saved by save_model().
    """
    file_path = os.path.join(processed_data_path, "models", algorithm, f"{model_name}.pkl")

    if not os.path.exists(file_path):
        print(f"❌ Model not found: {file_path}")
        return None

    try:
        model = joblib.load(file_path)
        print(f"✅ Model loaded: {file_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model '{model_name}': {e}")
        return None


# ============================================================
#  REPORTING UTILITIES
# ============================================================

def export_results_to_excel(results: Dict[str, pd.DataFrame], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, "segmentation_results.xlsx")

    with pd.ExcelWriter(excel_path) as writer:
        for sheet, df in results.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)

    print(f"📄 Excel report saved: {excel_path}")


def export_all_plots(
    plot_functions: List[Callable],
    figure_path: str,
    format: str = "pdf",
    html_scatter_data: dict = None,
    html_filename: str = "report.html",
    pdf_filename: str = "report.pdf"
) -> None:
    os.makedirs(figure_path, exist_ok=True)

    # PDF Export
    if format == "pdf":
        pdf_path = os.path.join(figure_path, pdf_filename)
        with PdfPages(pdf_path) as pdf:
            for plot_func in plot_functions:
                plot_func()
                fig = plt.gcf()
                pdf.savefig(fig)
                plt.close(fig)

        print(f"📄 PDF report saved: {pdf_path}")
        return

    # HTML Export
    if format == "html" and html_scatter_data:
        fig = go.Figure(
            data=go.Scatter(
                x=html_scatter_data.get("x"),
                y=html_scatter_data.get("y"),
                mode="markers",
                marker=dict(color=html_scatter_data.get("color")),
                text=html_scatter_data.get("hover")
            )
        )
        fig.update_layout(title=html_scatter_data.get("title", "Cluster Scatter"))
        html_path = os.path.join(figure_path, html_filename)
        pio.write_html(fig, file=html_path, auto_open=False)
        print(f"🌐 HTML report saved: {html_path}")
        return

    print("⚠️ Invalid export format or missing HTML input.")


# ============================================================
# 🧮 DATAFRAME UTILITIES
# ============================================================

def to_datetime(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def group_summary(df: pd.DataFrame, group_cols: List[str], metrics: Dict[str, str],
                  sort_by: str = None, top_n: int = None) -> pd.DataFrame:
    result = df.groupby(group_cols).agg(metrics).reset_index()

    if sort_by in result.columns:
        result = result.sort_values(sort_by, ascending=False)

    if top_n:
        result = result.head(top_n)

    return result.round(2)


def calculate_duration(df: pd.DataFrame, start_col: str, end_col: str,
                       new_col: str = "duration_days") -> pd.DataFrame:
    df = to_datetime(df, [start_col, end_col])
    df[new_col] = (df[end_col] - df[start_col]).dt.days
    return df


# ============================================================
# 🔍 PATH RESOLVER
# ============================================================

def get_path(path_type: str) -> str:
    """
    Convenient path resolver with automatic directory creation.

    Returns any project directory path based on a keyword.
    """

    paths = {
        # Project root structure
        "project": project_root,
        "base": base_path,
        
        # Config
        "config":config_path,

        # Data-level folders
        "data": data_path,
        "csv": csv_path,
        "sql": sql_path,
        "raw": raw_data_path,
        "processed": processed_data_path,

        # Processed subfolders
        "features_processed": feature_processed_path,
        "segmentation_processed": segmentation_processed_path,
        "pca_processed": pca_processed_path,
        "non_ml_csv": non_machine_learning_path,
        "ml_model": ml_model_path,
        "ab_test": ab_test_path,
        "classifier":classifier_path,
        

        # Reports top-level
        "reports": reports_path,
        "eda": eda_path,
        "loader":load_data_path,
        "cleaner":cleaner_path,
        "feature_reports": feature_path,
        "segment_reports": segment_reports_path,

        # Model directory (in reports/segment/models)
        "models": model_path,
        "non_ml": non_ml_path,
        'kmeans': kmeans_path,
        'dbscan': dbscan_path
    }

    if path_type not in paths:
        raise ValueError(
            f"❌ Unknown path type '{path_type}'. Allowed values: {list(paths.keys())}"
        )

    resolved = os.path.abspath(paths[path_type])
    os.makedirs(resolved, exist_ok=True)
    return resolved

