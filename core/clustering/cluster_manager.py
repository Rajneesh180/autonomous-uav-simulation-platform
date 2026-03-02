from typing import List, Optional, Tuple
import numpy as np
from core.models.node_model import Node
from config.config import Config
from core.clustering.semantic_clusterer import SemanticClusterer


class ClusterManager:
    """
    Manages operational clustering triggers and delegates mathematical
    latent space dimensionality reduction to the SemanticClusterer.

    Supports quality-based reclustering: if the silhouette score from the
    last clustering falls below ``Config.SILHOUETTE_RECLUSTER_THRESH``,
    ``should_recluster`` returns True even when the node count is stable.
    """

    def __init__(self):
        self.last_node_count = 0
        self.recluster_count = 0

        conf = {
            "scaling_method": Config.SCALING_METHOD,
            "reduction_target": Config.REDUCTION_DIMS,
            "cluster_algo": Config.CLUSTER_ALGO_MODE,
        }
        self.semantic_engine = SemanticClusterer(config=conf)
        self.current_centroids = np.zeros((1, 3))
        self.current_labels = np.array([])

    # ------------------------------------------------------------------
    # Reclustering trigger
    # ------------------------------------------------------------------

    def should_recluster(
        self, current_node_count: int, threshold: int = 5,
        current_silhouette: Optional[float] = None,
    ) -> bool:
        """
        Returns True when reclustering is warranted.

        Triggers:
          1. Node count changed by ≥ *threshold* since last clustering.
          2. Cluster quality (silhouette) dropped below the configured
             ``SILHOUETTE_RECLUSTER_THRESH``.
        """
        if abs(current_node_count - self.last_node_count) >= threshold:
            return True

        sil = current_silhouette if current_silhouette is not None else self.last_silhouette
        if sil is not None and sil < Config.SILHOUETTE_RECLUSTER_THRESH:
            return True

        return False

    # ------------------------------------------------------------------
    # Clustering execution
    # ------------------------------------------------------------------

    def perform_clustering(
        self, nodes: List[Node], current_time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute semantic clustering and cache centroids / labels."""
        self.last_node_count = len(nodes)
        self.recluster_count += 1

        self.current_centroids, self.current_labels = self.semantic_engine.cluster(
            nodes, current_time, n_clusters=Config.CLUSTER_COUNT,
        )
        return self.current_centroids, self.current_labels

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def last_silhouette(self) -> Optional[float]:
        """Silhouette score from the most recent clustering (may be None)."""
        return self.semantic_engine.last_silhouette

    def get_recluster_count(self) -> int:
        return self.recluster_count
