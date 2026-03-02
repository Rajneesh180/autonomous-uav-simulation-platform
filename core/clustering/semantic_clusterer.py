import numpy as np
from typing import List, Tuple, Dict, Any

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.cluster import DBSCAN, KMeans
except ImportError:
    PCA = None

from core.models.node_model import Node
from core.clustering.feature_scaler import FeatureScaler


class SemanticClusterer:
    """
    Phase 4: Semantic Intelligence Layer.
    Translates raw Node spatial properties into high-dimensional semantic spaces,
    reduces dimensions via PCA/t-SNE, and applies hybrid weighted clustering (DBSCAN/KMeans).
    """

    def __init__(self, config: Dict[str, Any]):
        self.scaling_method = config.get("scaling_method", "minmax")
        self.reduction_target = config.get("reduction_target", 3)
        self.cluster_algo = config.get("cluster_algo", "kmeans")
        self.feature_scaler = FeatureScaler(method=self.scaling_method)
        
        self.pca_model = None
        if PCA:
            self.pca_model = PCA(n_components=self.reduction_target)

    def extract_and_scale(self, nodes: List[Node], current_time: float) -> np.ndarray:
        if not nodes:
            return np.array([])

        raw_features = np.array([n.get_feature_vector() for n in nodes])
        
        # Structure: [x, y, z, priority, risk, signal, deadline, buffer, reliability, aoi]
        # We handle Time Decay for the deadline (index 6) separately
        deadlines = raw_features[:, 6]
        urgency_weights = self.feature_scaler.apply_time_decay(deadlines, current_time)
        
        # Scale remaining features
        scaled = self.feature_scaler.fit_transform(raw_features)
        
        # Override the deadline column with calculated, scaled urgency
        scaled[:, 6] = urgency_weights
        
        # Phase 3.9: Age of Information (AoI) Calculus
        # Dynamically boost the Priority (Index 3) based on AoI Staleness (Index 9)
        # Configurable weight (Config.AOI_URGENCY_WEIGHT) controls how aggressively
        # stale-data nodes override spatial proximity in PCA/DBSCAN clustering.
        if scaled.shape[1] > 9:
            from config.config import Config
            scaled[:, 3] += (scaled[:, 9] * Config.AOI_URGENCY_WEIGHT)
            
        return scaled

    def reduce_dimensions(self, scaled_features: np.ndarray) -> np.ndarray:
        if self.pca_model and len(scaled_features) > self.reduction_target:
            try:
                return self.pca_model.fit_transform(scaled_features)
            except Exception as e:
                print(f"[Warning] PCA Dimension Reduction failed: {e}. Falling back to raw.")
                # Fallback if too few samples
                return scaled_features
        return scaled_features

    def cluster(self, nodes: List[Node], current_time: float, n_clusters: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes the Semantic Clustering pipeline over the node lists.
        Returns (Centroids, Labels).
        """
        if not nodes or PCA is None:
            # Fallback for empty or missing sklearn
            return np.zeros((1, 3)), np.zeros(len(nodes))

        # 1. Pipeline: Semantic Normalization
        scaled = self.extract_and_scale(nodes, current_time)

        # 2. Pipeline: Latent Dimension Reduction
        latent_space = self.reduce_dimensions(scaled)

        # 3. Pipeline: Clustering Assignment
        if self.cluster_algo == "dbscan":
            from config.config import Config
            model = DBSCAN(eps=Config.DBSCAN_EPS, min_samples=Config.DBSCAN_MIN_SAMPLES)
            labels = model.fit_predict(latent_space)
            
            # DBSCAN does not provide standard centroids perfectly. 
            # We approximate them by averaging points in each cluster.
            unique_labels = set(labels)
            centroids = []
            for k in unique_labels:
                if k == -1: # Noise
                    continue
                class_member_mask = (labels == k)
                # Compute centroid back in real spatial space for UAV routing
                spatial_coords = np.array([[n.x, n.y, n.z] for i, n in enumerate(nodes) if class_member_mask[i]])
                centroids.append(spatial_coords.mean(axis=0))
            
            # Format
            centroids = np.array(centroids) if centroids else np.zeros((1, 3))
            
        else: # Default K-Means
            n_clusters = min(n_clusters, len(nodes))
            model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
            labels = model.fit_predict(latent_space)
            
            # Map latent centroids back to spatial space proxy by finding nearest real nodes
            centroids = []
            spatial_coords = np.array([[n.x, n.y, n.z] for n in nodes])
            for k in range(n_clusters):
                mask = (labels == k)
                if np.any(mask):
                    centroids.append(spatial_coords[mask].mean(axis=0))
                else:
                    centroids.append([0.0, 0.0, 0.0])
            centroids = np.array(centroids)

        return centroids, labels

