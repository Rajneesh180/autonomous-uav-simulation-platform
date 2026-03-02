import numpy as np
from typing import List, Tuple, Dict, Any, Optional

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
except ImportError:
    PCA = None
    silhouette_score = None

from core.models.node_model import Node
from core.clustering.feature_scaler import FeatureScaler


class SemanticClusterer:
    """
    Phase 4: Semantic Intelligence Layer.

    Translates raw Node spatial properties into high-dimensional semantic spaces,
    reduces dimensions via PCA, and applies clustering (KMeans / DBSCAN / GMM / auto).

    When ``cluster_algo == "auto"``  the clusterer evaluates KMeans, DBSCAN, and GMM
    on the current feature matrix and selects the algorithm with the highest
    silhouette score — making algorithm choice data-driven rather than hardcoded.
    """

    def __init__(self, config: Dict[str, Any]):
        self.scaling_method = config.get("scaling_method", "minmax")
        self.reduction_target = config.get("reduction_target", 3)
        self.cluster_algo = config.get("cluster_algo", "kmeans")
        self.feature_scaler = FeatureScaler(method=self.scaling_method)

        self.pca_model = None
        if PCA:
            self.pca_model = PCA(n_components=self.reduction_target)

        # Track last clustering quality for reclustering decisions
        self.last_silhouette: Optional[float] = None

    # ------------------------------------------------------------------
    # Feature pipeline
    # ------------------------------------------------------------------

    def extract_and_scale(self, nodes: List[Node], current_time: float) -> np.ndarray:
        if not nodes:
            return np.array([])

        raw_features = np.array([n.get_feature_vector() for n in nodes])

        # Structure: [x, y, z, priority, risk, signal, deadline, buffer, reliability, aoi]
        deadlines = raw_features[:, 6]
        urgency_weights = self.feature_scaler.apply_time_decay(deadlines, current_time)

        scaled = self.feature_scaler.fit_transform(raw_features)
        scaled[:, 6] = urgency_weights

        # AoI-driven priority boost (Phase 3.9)
        if scaled.shape[1] > 9:
            from config.config import Config
            scaled[:, 3] += scaled[:, 9] * Config.AOI_URGENCY_WEIGHT

        return scaled

    def reduce_dimensions(self, scaled_features: np.ndarray) -> np.ndarray:
        if self.pca_model and len(scaled_features) > self.reduction_target:
            try:
                return self.pca_model.fit_transform(scaled_features)
            except Exception as e:
                print(f"[Warning] PCA Dimension Reduction failed: {e}. Falling back to raw.")
                return scaled_features
        return scaled_features

    # ------------------------------------------------------------------
    # Adaptive K selection (silhouette-based)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_best_k(X: np.ndarray, k_min: int = 2, k_max: int = 10) -> int:
        """Sweep k ∈ [k_min, k_max] and return the k with highest silhouette score."""
        k_max = min(k_max, max(2, len(X) // 3))
        if k_max < k_min:
            return k_min

        best_k, best_score = k_min, -1.0
        for k in range(k_min, k_max + 1):
            try:
                labels = KMeans(n_clusters=k, n_init=5, random_state=42).fit_predict(X)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_k, best_score = k, score
            except Exception:
                continue
        return best_k

    # ------------------------------------------------------------------
    # Individual algorithm runners
    # ------------------------------------------------------------------

    @staticmethod
    def _run_kmeans(X: np.ndarray, nodes: List[Node], n_clusters: int
                    ) -> Tuple[np.ndarray, np.ndarray]:
        n_clusters = min(n_clusters, len(nodes))
        model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42)
        labels = model.fit_predict(X)
        centroids = SemanticClusterer._spatial_centroids(nodes, labels, n_clusters)
        return centroids, labels

    @staticmethod
    def _run_dbscan(X: np.ndarray, nodes: List[Node]
                    ) -> Tuple[np.ndarray, np.ndarray]:
        from config.config import Config
        model = DBSCAN(eps=Config.DBSCAN_EPS, min_samples=Config.DBSCAN_MIN_SAMPLES)
        labels = model.fit_predict(X)
        centroids = SemanticClusterer._spatial_centroids_dbscan(nodes, labels)
        return centroids, labels

    @staticmethod
    def _run_gmm(X: np.ndarray, nodes: List[Node], n_clusters: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
        n_clusters = min(n_clusters, len(nodes))
        gmm = GaussianMixture(n_components=n_clusters, covariance_type="full",
                               random_state=42, max_iter=200)
        labels = gmm.fit_predict(X)
        centroids = SemanticClusterer._spatial_centroids(nodes, labels, n_clusters)
        return centroids, labels

    # ------------------------------------------------------------------
    # Auto-select: try all three, keep highest silhouette
    # ------------------------------------------------------------------

    def _auto_select(self, X: np.ndarray, nodes: List[Node], n_clusters: int
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate KMeans, DBSCAN, GMM and pick the one with best silhouette."""
        candidates: list = []

        # KMeans
        try:
            c, l = self._run_kmeans(X, nodes, n_clusters)
            if len(set(l)) >= 2:
                s = silhouette_score(X, l)
                candidates.append(("kmeans", c, l, s))
        except Exception:
            pass

        # DBSCAN
        try:
            c, l = self._run_dbscan(X, nodes)
            valid = set(l) - {-1}
            if len(valid) >= 2:
                mask = l != -1
                s = silhouette_score(X[mask], l[mask])
                candidates.append(("dbscan", c, l, s))
        except Exception:
            pass

        # GMM
        try:
            c, l = self._run_gmm(X, nodes, n_clusters)
            if len(set(l)) >= 2:
                s = silhouette_score(X, l)
                candidates.append(("gmm", c, l, s))
        except Exception:
            pass

        if not candidates:
            # Fallback to KMeans with no silhouette guard
            return self._run_kmeans(X, nodes, n_clusters)

        best = max(candidates, key=lambda t: t[3])
        return best[1], best[2]

    # ------------------------------------------------------------------
    # Centroid helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spatial_centroids(nodes: List[Node], labels: np.ndarray, n_clusters: int
                           ) -> np.ndarray:
        spatial = np.array([[n.x, n.y, n.z] for n in nodes])
        centroids = []
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                centroids.append(spatial[mask].mean(axis=0))
            else:
                centroids.append([0.0, 0.0, 0.0])
        return np.array(centroids)

    @staticmethod
    def _spatial_centroids_dbscan(nodes: List[Node], labels: np.ndarray
                                  ) -> np.ndarray:
        unique = sorted(set(labels) - {-1})
        if not unique:
            return np.zeros((1, 3))
        spatial = np.array([[n.x, n.y, n.z] for n in nodes])
        centroids = []
        for k in unique:
            mask = labels == k
            centroids.append(spatial[mask].mean(axis=0))
        return np.array(centroids)

    # ------------------------------------------------------------------
    # Main clustering entry point
    # ------------------------------------------------------------------

    def cluster(self, nodes: List[Node], current_time: float,
                n_clusters: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full semantic clustering pipeline.

        Returns (centroids, labels).
        """
        if not nodes or PCA is None:
            return np.zeros((1, 3)), np.zeros(len(nodes))

        from config.config import Config

        # 1. Feature extraction + scaling
        scaled = self.extract_and_scale(nodes, current_time)

        # 2. Dimensionality reduction
        latent = self.reduce_dimensions(scaled)

        # 3. Adaptive K selection (for kmeans / gmm / auto)
        if self.cluster_algo in ("kmeans", "gmm", "auto") and silhouette_score is not None:
            best_k = self._find_best_k(latent, Config.AUTO_K_MIN, Config.AUTO_K_MAX)
        else:
            best_k = n_clusters

        # 4. Clustering
        if self.cluster_algo == "auto":
            centroids, labels = self._auto_select(latent, nodes, best_k)
        elif self.cluster_algo == "gmm":
            centroids, labels = self._run_gmm(latent, nodes, best_k)
        elif self.cluster_algo == "dbscan":
            centroids, labels = self._run_dbscan(latent, nodes)
        else:  # kmeans (default)
            centroids, labels = self._run_kmeans(latent, nodes, best_k)

        # 5. Track silhouette for quality-based reclustering
        self.last_silhouette = None
        try:
            unique_labels = set(labels)
            non_noise = unique_labels - {-1}
            if len(non_noise) >= 2 and silhouette_score is not None:
                if -1 in unique_labels:
                    mask = labels != -1
                    self.last_silhouette = float(silhouette_score(latent[mask], labels[mask]))
                else:
                    self.last_silhouette = float(silhouette_score(latent, labels))
        except Exception:
            pass

        return centroids, labels

