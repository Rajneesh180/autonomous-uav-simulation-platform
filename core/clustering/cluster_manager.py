from typing import List, Tuple
import numpy as np
from core.models.node_model import Node
from config.config import Config
from core.clustering.semantic_clusterer import SemanticClusterer

class ClusterManager:
    """
    Manages operational clustering triggers and delegates mathematical 
    latent space dimensionality reduction to the SemanticClusterer.
    """
    def __init__(self):
        self.last_node_count = 0
        self.recluster_count = 0
        
        # Load Phase-4 Config
        conf = {
            "scaling_method": Config.SCALING_METHOD,
            "reduction_target": Config.REDUCTION_DIMS,
            "cluster_algo": Config.CLUSTER_ALGO_MODE
        }
        self.semantic_engine = SemanticClusterer(config=conf)
        self.current_centroids = np.zeros((1, 3))
        self.current_labels = np.array([])

    def should_recluster(self, current_node_count: int, threshold: int = 5) -> bool:
        if abs(current_node_count - self.last_node_count) >= threshold:
            return True
        return False

    def perform_clustering(self, nodes: List[Node], current_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes semantic or geometric clustering based on config toggles.
        """
        self.last_node_count = len(nodes)
        self.recluster_count += 1
        
        if Config.ENABLE_SEMANTIC_CLUSTERING:
            self.current_centroids, self.current_labels = self.semantic_engine.cluster(
                nodes, current_time, n_clusters=Config.CLUSTER_COUNT
            )
        else:
            # Fallback for geometric only, handled generically through SemanticClusterer 
            # with disabled scaling if specified.
            self.current_centroids, self.current_labels = self.semantic_engine.cluster(
                nodes, current_time, n_clusters=Config.CLUSTER_COUNT
            )
            
        return self.current_centroids, self.current_labels

    def get_recluster_count(self) -> int:
        return self.recluster_count
