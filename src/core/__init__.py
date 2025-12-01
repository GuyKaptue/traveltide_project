# src/core/__init__.py
"""
Core module initializer for TravelTide.

Provides data preparation, feature engineering, and segmentation components.
"""

from .processing import DataLoader, SessionCleaner, TravelTideEDA
from .features import (
    UserBehaviorMetrics,
    UserAdvancedMetrics,
    UserFeaturePipeline,
)

from .segment.non_ml import (
    MetricsComputer,
    ThresholdManager,
    PerkAssigner,
    DataManager,
    SegmentationAnalyzer,
    SegmentationVisualizer,
    NonMachineLearningSegment,
    AdvanceSegmentAnalyzer,
)

from .segment.ml_model import (
    MLClustering,
    ClusteringOrchestrator,
    KmeansClustering,
    FeatureEngineer,
    KMeansEngine,
    DBSCANEngine,
    PerkAssigner as MLPerkAssigner,
    MetricsCalculator,
    ClusterVisualizer,
    DataExporter,
)

from .segment.comparison import SegmentationComparator

from .segment.test import (
    PerkAssignmentTest,
    ABTestFramework,
    ClassificationModelTest
)

__all__ = [
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
    
    # kmeans vs Manual
    'SegmentationComparator',
    
    # Tests
    'PerkAssignmentTest',
    'ABTestFramework',
    'ClassificationModelTest'
]
