"""
Clustering
==========
Semantic and distribution-aware node clustering with feature scaling.

Usage::

    from core.clustering import ClusterManager, SemanticClusterer, FeatureScaler
"""

from core.clustering.cluster_manager import ClusterManager
from core.clustering.semantic_clusterer import SemanticClusterer
from core.clustering.feature_scaler import FeatureScaler

__all__ = [
    "ClusterManager",
    "SemanticClusterer",
    "FeatureScaler",
]
