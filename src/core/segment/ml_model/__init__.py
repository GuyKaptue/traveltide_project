# core/segment/ml_model/__init__.py

"""
ML Customer Segmentation Module
================================

Professional machine learning clustering for customer segmentation
using K-Means and DBSCAN algorithms with automatic perk assignment.

Main Components:
----------------
- ClusteringOrchestrator: Main pipeline coordinator
- MLClustering: Simplified all-in-one interface (RECOMMENDED)
- FeatureEngineer: Feature creation and selection
- KMeansEngine: K-Means implementation
- DBSCANEngine: DBSCAN implementation
- PerkAssigner: Perk and segment name assignment
- MetricsCalculator: Clustering quality metrics
- ClusterVisualizer: Visualization generation
- DataExporter: Data export operations

Quick Start:
------------
    from core.segment.ml_model import MLClustering
    
    # Simple usage
    ml = MLClustering(config_path='config/ml_config.yaml')
    results = ml.run_both(df, n_clusters=5)
    
    # Access results
    kmeans_df = results['kmeans']['df']
    dbscan_df = results['dbscan']['df']

Advanced Usage:
---------------
    from core.segment.ml_model import ClusteringOrchestrator
    
    # Full modular system
    orchestrator = ClusteringOrchestrator(
        config_path='config/ml_config.yaml',
        run_name='experiment_v1'
    )
    
    # Run comparison
    results = orchestrator.run_comparison(df)

Individual Components:
----------------------
    from core.segment.ml_model import (
        FeatureEngineer,
        KMeansEngine,
        DBSCANEngine,
        PerkAssigner
    )
    
    # Use components independently
    config = load_yaml('config/ml_config.yaml')
    
    feature_eng = FeatureEngineer(config)
    df_eng = feature_eng.engineer_features(df)
    
    kmeans = KMeansEngine(config)
    results = kmeans.fit_and_assign(X_scaled, df_eng)
"""

# Import main classes for easy access
from .simple_ml_clustering import MLClustering
from .clustering_orchestrator import ClusteringOrchestrator
from .feature_engineer import FeatureEngineer
from .kmeans_engine import KMeansEngine
from .dbscan_engine import DBSCANEngine
from .perk_assigner import PerkAssigner
from .metrics_calculator import MetricsCalculator
from .visualizer import ClusterVisualizer
from .data_exporter import DataExporter
from .kmean_cluster import KmeansClustering

# Define public API
__all__ = [
    # Main interfaces
    'MLClustering',                 
    'ClusteringOrchestrator', 
    'KmeansClustering',      
    
    # Core components
    'FeatureEngineer',
    'KMeansEngine',
    'DBSCANEngine',
    'PerkAssigner',
    'MetricsCalculator',
    'ClusterVisualizer',
    'DataExporter',
]

# Version info
__version__ = '1.0.0'
__author__ = 'Guy Kaptue'
__description__ = 'ML Customer Segmentation with K-Means and DBSCAN'

# Module metadata
__clustering_algorithms__ = ['kmeans', 'dbscan']
__supported_metrics__ = [
    'silhouette_score',
    'davies_bouldin_score',
    'calinski_harabasz_score',
    'cluster_balance',
    'cluster_stability',
    'business_alignment'
]

# Configuration validation
def validate_config(config: dict) -> bool:
    """
    Validate configuration dictionary.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
        
    Returns
    -------
    bool
        True if valid, False otherwise
    """
    required_sections = ['segmentation']  # noqa: F841
    required_subsections = [
        'threshold_definitions',
        'all_perks',
        'group_names',
        'clustering'
    ]
    
    # Check main section
    if 'segmentation' not in config:
        print("❌ Missing 'segmentation' section in config")
        return False
    
    seg_config = config['segmentation']
    
    # Check subsections
    for subsection in required_subsections:
        if subsection not in seg_config:
            print(f"❌ Missing '{subsection}' in segmentation config")
            return False
    
    # Validate perks and names match
    n_perks = len(seg_config.get('all_perks', []))
    n_names = len(seg_config.get('group_names', []))
    
    if n_perks != n_names:
        print(f"⚠️ Warning: {n_perks} perks but {n_names} group names")
        print("   They should match for proper assignment")
    
    print("✅ Configuration is valid")
    return True


def print_module_info():
    """Print module information and usage examples."""
    info = f"""
╔════════════════════════════════════════════════════════════════╗
║              ML CUSTOMER SEGMENTATION MODULE                   ║
║                      Version {__version__}                            ║
╚════════════════════════════════════════════════════════════════╝

📦 Package: core.segment.ml_model
👤 Author: {__author__}
📝 Description: {__description__}

🎯 Supported Algorithms:
   • K-Means - Fixed cluster count, balanced segments
   • DBSCAN - Density-based, automatic cluster detection

📊 Quality Metrics:
   • Silhouette Score (cluster separation)
   • Davies-Bouldin Index (cluster quality)
   • Calinski-Harabasz Score (variance ratio)
   • Business alignment metrics

🚀 Quick Start:
   >>> from core.segment.ml_model import MLClustering
   >>> ml = MLClustering('config/ml_config.yaml')
   >>> results = ml.run_both(your_dataframe)

📚 Documentation:
   See README.md for complete usage guide

💡 Need help?
   • Check config/ml_config.yaml for settings
   • Review example_usage.py for patterns
   • Validate config with validate_config(config)
"""
    print(info)


# Print info when module is imported (optional, comment out if too verbose)
print_module_info()