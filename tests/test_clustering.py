"""
Tests for the clustering pipeline:
  - core.clustering.semantic_clusterer.SemanticClusterer
  - core.clustering.cluster_manager.ClusterManager
  - core.clustering.feature_scaler.FeatureScaler
Validates feature extraction, PCA reduction, KMeans/DBSCAN/GMM/auto assignment,
adaptive-K selection, silhouette quality tracking, and cluster manager reclustering.
"""

import numpy as np
import pytest

from core.clustering.semantic_clusterer import SemanticClusterer
from core.clustering.cluster_manager import ClusterManager
from core.clustering.feature_scaler import FeatureScaler
from config.config import Config


# ------------------------------------------------------------------
# FeatureScaler tests
# ------------------------------------------------------------------

class TestFeatureScaler:

    def test_minmax_bounds(self, sample_nodes):
        """MinMax scaling should produce values in [0, 1]."""
        scaler = FeatureScaler(method="minmax")
        raw = np.array([n.get_feature_vector() for n in sample_nodes])
        scaled = scaler.fit_transform(raw)
        assert scaled.min() >= -1e-6 and scaled.max() <= 1.0 + 1e-6

    def test_zscore_zero_mean(self, sample_nodes):
        """Z-score scaling should centre variable features near zero."""
        scaler = FeatureScaler(method="zscore")
        raw = np.array([n.get_feature_vector() for n in sample_nodes])
        scaled = scaler.fit_transform(raw)
        means = np.abs(scaled.mean(axis=0))
        # Constant-variance columns stay unchanged; only check variable ones
        stds = raw.std(axis=0)
        variable = stds > 1e-6
        assert np.all(means[variable] < 0.5), f"Z-score means too large: {means[variable]}"

    def test_time_decay(self):
        """Time decay should produce higher urgency for closer deadlines."""
        scaler = FeatureScaler(method="minmax")
        deadlines = np.array([10.0, 50.0, 100.0, 500.0])
        weights = scaler.apply_time_decay(deadlines, current_time=40.0)
        # Deadline 10 is already passed (high urgency), 500 is distant (low)
        assert weights[0] >= weights[-1]


# ------------------------------------------------------------------
# SemanticClusterer tests
# ------------------------------------------------------------------

class TestSemanticClusterer:

    @pytest.fixture
    def kmeans_clusterer(self):
        return SemanticClusterer({
            "scaling_method": "minmax",
            "reduction_target": 3,
            "cluster_algo": "kmeans",
        })

    @pytest.fixture
    def dbscan_clusterer(self):
        return SemanticClusterer({
            "scaling_method": "minmax",
            "reduction_target": 3,
            "cluster_algo": "dbscan",
        })

    @pytest.fixture
    def gmm_clusterer(self):
        return SemanticClusterer({
            "scaling_method": "minmax",
            "reduction_target": 3,
            "cluster_algo": "gmm",
        })

    @pytest.fixture
    def auto_clusterer(self):
        return SemanticClusterer({
            "scaling_method": "minmax",
            "reduction_target": 3,
            "cluster_algo": "auto",
        })

    def test_kmeans_returns_correct_shapes(self, kmeans_clusterer, sample_nodes):
        """KMeans should return centroids and labels of expected shape."""
        centroids, labels = kmeans_clusterer.cluster(sample_nodes, current_time=0, n_clusters=3)
        assert centroids.shape[1] == 3  # x, y, z
        assert len(labels) == len(sample_nodes)

    def test_dbscan_returns_labels(self, dbscan_clusterer, sample_nodes):
        """DBSCAN should assign a label to every node (noise = -1 is valid)."""
        _, labels = dbscan_clusterer.cluster(sample_nodes, current_time=0)
        assert len(labels) == len(sample_nodes)

    def test_gmm_returns_labels(self, gmm_clusterer, sample_nodes):
        """GMM should assign a label to every node."""
        centroids, labels = gmm_clusterer.cluster(sample_nodes, current_time=0, n_clusters=3)
        assert len(labels) == len(sample_nodes)
        assert centroids.shape[1] == 3

    def test_auto_select_returns_labels(self, auto_clusterer, sample_nodes):
        """Auto-select should pick the best algorithm and return valid output."""
        centroids, labels = auto_clusterer.cluster(sample_nodes, current_time=0, n_clusters=3)
        assert len(labels) == len(sample_nodes)
        assert centroids.shape[1] == 3

    def test_silhouette_tracked(self, kmeans_clusterer, sample_nodes):
        """After clustering, last_silhouette should be populated (or None for tiny data)."""
        kmeans_clusterer.cluster(sample_nodes, current_time=0, n_clusters=3)
        sil = kmeans_clusterer.last_silhouette
        # With enough nodes and distinct clusters, silhouette should be a float
        assert sil is None or (-1.0 <= sil <= 1.0)

    def test_adaptive_k_returns_valid(self, sample_nodes):
        """_find_best_k should return an integer in [k_min, k_max]."""
        clusterer = SemanticClusterer({
            "scaling_method": "minmax", "reduction_target": 3, "cluster_algo": "kmeans"
        })
        scaled = clusterer.extract_and_scale(sample_nodes, current_time=0)
        reduced = clusterer.reduce_dimensions(scaled)
        best_k = SemanticClusterer._find_best_k(reduced, k_min=2, k_max=6)
        assert 2 <= best_k <= 6

    def test_empty_nodes(self, kmeans_clusterer):
        """Empty node list should not crash."""
        centroids, labels = kmeans_clusterer.cluster([], current_time=0)
        assert len(labels) == 0

    def test_extract_and_scale_dimensions(self, kmeans_clusterer, sample_nodes):
        """Scaled features should have 10 columns (feature vector dimension)."""
        scaled = kmeans_clusterer.extract_and_scale(sample_nodes, current_time=0)
        assert scaled.shape == (len(sample_nodes), 10)

    def test_pca_reduces_dimensions(self, kmeans_clusterer, sample_nodes):
        """PCA should reduce from 10-D to reduction_target (3) dimensions."""
        scaled = kmeans_clusterer.extract_and_scale(sample_nodes, current_time=0)
        reduced = kmeans_clusterer.reduce_dimensions(scaled)
        assert reduced.shape[1] == 3


# ------------------------------------------------------------------
# ClusterManager tests
# ------------------------------------------------------------------

class TestClusterManager:

    def test_perform_clustering_returns_centroids_and_labels(self, sample_nodes):
        """ClusterManager should return centroids and labels arrays."""
        manager = ClusterManager()
        centroids, labels = manager.perform_clustering(sample_nodes, current_time=0.0)
        assert centroids.shape[1] == 3  # spatial centroids
        assert len(labels) == len(sample_nodes)

    def test_should_recluster_triggers_on_count_change(self, sample_nodes):
        """Recluster flag should trigger when node count changes beyond threshold."""
        manager = ClusterManager()
        manager.last_node_count = 10
        assert manager.should_recluster(20, threshold=5) is True
        assert manager.should_recluster(12, threshold=5) is False

    def test_should_recluster_triggers_on_low_silhouette(self, sample_nodes):
        """Recluster should trigger when silhouette is below threshold."""
        manager = ClusterManager()
        manager.last_node_count = 10
        # Node count hasn't changed, but silhouette is poor
        assert manager.should_recluster(10, threshold=5, current_silhouette=0.1) is True
        # Good silhouette + stable count → no recluster
        assert manager.should_recluster(10, threshold=5, current_silhouette=0.8) is False

    def test_last_silhouette_accessible(self, sample_nodes):
        """last_silhouette property should work after perform_clustering."""
        manager = ClusterManager()
        manager.perform_clustering(sample_nodes, current_time=0.0)
        sil = manager.last_silhouette
        assert sil is None or isinstance(sil, float)

    def test_recluster_count_increments(self, sample_nodes):
        """Each perform_clustering call should increment recluster_count."""
        manager = ClusterManager()
        assert manager.get_recluster_count() == 0
        manager.perform_clustering(sample_nodes, current_time=0.0)
        assert manager.get_recluster_count() == 1
        manager.perform_clustering(sample_nodes, current_time=1.0)
        assert manager.get_recluster_count() == 2
