# core/segment/__init__.py
"""
Core segmentation package initializer.

Exposes both non‑ML (rule‑based) and ML (clustering‑based) segmentation
components for external use.
"""

# ---------------------------------------------------------------------------
# Non‑ML segmentation components
# ---------------------------------------------------------------------------
from .non_ml import (
    MetricsComputer,
    ThresholdManager,
    PerkAssigner,
    DataManager,
    SegmentationAnalyzer,
    SegmentationVisualizer,
    NonMachineLearningSegment,
    AdvanceSegmentAnalyzer,
)

# ---------------------------------------------------------------------------
# ML segmentation components
# ---------------------------------------------------------------------------
from .ml_model import (
    # Main interfaces
    MLClustering,
    ClusteringOrchestrator,
    KmeansClustering,

    # Core components
    FeatureEngineer,
    KMeansEngine,
    DBSCANEngine,
    PerkAssigner as MLPerkAssigner,
    MetricsCalculator,
    ClusterVisualizer,
    DataExporter,
)

from .comparison import SegmentationComparator
from .test import (
    PerkAssignmentTest,
    ABTestFramework,
    ClassificationModelTest
)

__all__ = [
    # Non‑ML methods
    "MetricsComputer",
    "ThresholdManager",
    "PerkAssigner",
    "DataManager",
    "SegmentationAnalyzer",
    "SegmentationVisualizer",
    "NonMachineLearningSegment",
    "AdvanceSegmentAnalyzer",

    # ML methods
    "MLClustering",
    "ClusteringOrchestrator",
    'KmeansClustering',
    
    
    "FeatureEngineer",
    "KMeansEngine",
    "DBSCANEngine",
    "MLPerkAssigner",
    "MetricsCalculator",
    "ClusterVisualizer",
    "DataExporter",
    
    # Comparings kmeans vs Manual
    'SegmentationComparator',
    
    # Test
    'PerkAssignmentTest',
    'ABTestFramework',
    'ClassificationModelTest'
]
