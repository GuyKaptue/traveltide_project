# src/__init__.py
"""
TravelTide Package
"""
__version__ = "0.1.0"
__author__ = "Guy Kaptue"

from .db import Database
from .utils import (
    # Directory paths
    project_root,
    base_path,
    config_path,
    raw_data_path,
    processed_data_path,
    feature_processed_path,
    segmentation_processed_path,
    sql_path,
    reports_path,
    pca_processed_path,
    get_path,

    # Config
    load_config,
    load_yaml,

    # Model persistence
    save_model,
    load_model,

    # Reporting
    export_results_to_excel,
    export_all_plots,

    # DataFrame utilities
    to_datetime,
    group_summary,
    calculate_duration,
)

from .core import (
    # Preparing data
    DataLoader,
    SessionCleaner,
    TravelTideEDA,

    # Feature Metrics
    UserBehaviorMetrics,
    UserAdvancedMetrics,
    UserFeaturePipeline,

    # Non‑ML segmentation
    MetricsComputer,
    ThresholdManager,
    PerkAssigner,
    DataManager,
    SegmentationAnalyzer,
    SegmentationVisualizer,
    NonMachineLearningSegment,
    AdvanceSegmentAnalyzer,

    # ML segmentation
    MLClustering,
    ClusteringOrchestrator,
    KmeansClustering, 
    FeatureEngineer,
    KMeansEngine,
    DBSCANEngine,
    MLPerkAssigner,
    MetricsCalculator,
    ClusterVisualizer,
    DataExporter,
    
    # Kmeans vs. Manual
    SegmentationComparator,
    
    #Tests
    PerkAssignmentTest,
    ABTestFramework,
    ClassificationModelTest,
)



__all__ = [
    # Database
    "Database",

    # Paths
    "project_root",
    "base_path",
    "config_path",
    "raw_data_path",
    "processed_data_path",
    "feature_processed_path",
    "segmentation_processed_path",
    "sql_path",
    "reports_path",
    "pca_processed_path",
    "get_path",

    # Config
    "load_config",
    "load_yaml",

    # Model persistence
    "save_model",
    "load_model",

    # Reporting
    "export_results_to_excel",
    "export_all_plots",

    # DataFrame utilities
    "to_datetime",
    "group_summary",
    "calculate_duration",

    # Preparing Data
    "DataLoader",
    "SessionCleaner",
    "TravelTideEDA",

    # Feature Metrics
    "UserBehaviorMetrics",
    "UserAdvancedMetrics",
    "UserFeaturePipeline",

    # Non‑ML segmentation
    "MetricsComputer",
    "ThresholdManager",
    "PerkAssigner",
    "DataManager",
    "SegmentationAnalyzer",
    "SegmentationVisualizer",
    "NonMachineLearningSegment",
    "AdvanceSegmentAnalyzer",

    # ML segmentation
    "MLClustering",
    "ClusteringOrchestrator",
    "KmeansClustering",
    "FeatureEngineer",
    "KMeansEngine",
    "DBSCANEngine",
    "MLPerkAssigner",
    "MetricsCalculator",
    "ClusterVisualizer",
    "DataExporter",
    
    # Kmeans vs. Manual
    'SegmentationComparator',
    
    # Tests
    'PerkAssignmentTest',
    'ABTestFramework',
    'ClassificationModelTest',
]
