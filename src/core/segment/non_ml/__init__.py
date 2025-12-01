# core/segment/non_ml/__init__.py

"""
Non-ML Customer Segmentation Module
====================================

Rule-based customer segmentation using behavioral thresholds
and tier-based perk assignment logic.

Main Components:
----------------
- NonMachineLearningSegment: Main orchestrator class
- MetricsComputer: Behavioral metric computation
- ThresholdManager: Quantile-based threshold calculation
- PerkAssigner: Tier-based perk assignment
- DataManager: Data export operations
- SegmentationAnalyzer: Analysis and debugging tools
- SegmentationVisualizer: Visualization generation
- AdvanceSegmentAnalyzer: Advance Analysis

Usage:
------
    from core.segment.non_ml import NonMachineLearningSegment
    
    segmenter = NonMachineLearningSegment(users_df)
    segmented_df, distribution, tree_fig = segmenter.run()

Individual Operations:
----------------------
    # Step by step
    segmenter.compute_intermediate_metrics()
    segmenter.compute_thresholds()
    distribution = segmenter.assign_perks()
    segmenter.save_customer_segmentation()
    analysis = segmenter.analyze_perk_assignments()
    segmenter.create_visualizations()
"""

from .metrics import MetricsComputer
from .threshold_manager import ThresholdManager
from .perk_assigner import PerkAssigner
from .data_manager import DataManager
from .analyzer import SegmentationAnalyzer
from .visualizer import SegmentationVisualizer
from .non_machine_learning_segment import NonMachineLearningSegment
from .advance_analyzer import AdvanceSegmentAnalyzer

__all__ = [
    'MetricsComputer',
    'ThresholdManager',
    'PerkAssigner',
    'DataManager',
    'SegmentationAnalyzer',
    'SegmentationVisualizer',
    'NonMachineLearningSegment',
    'AdvanceSegmentAnalyzer'
]

