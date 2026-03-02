"""
Tests for core.models.node_model.Node
Validates dataclass fields, position tuple, feature vector, and buffer dynamics.
"""

import pytest
from core.models.node_model import Node
from config.config import Config


class TestNodeDataclass:

    def test_position_tuple(self, make_node):
        """position() should return (x, y, z)."""
        node = make_node(x=10.0, y=20.0, z=5.0)
        assert node.position() == (10.0, 20.0, 5.0)

    def test_feature_vector_length(self, make_node):
        """get_feature_vector() should return exactly 10 elements."""
        node = make_node()
        vec = node.get_feature_vector()
        assert len(vec) == 10

    def test_feature_vector_contains_position(self, make_node):
        """First three features should be x, y, z."""
        node = make_node(x=42.0, y=99.0, z=3.0)
        vec = node.get_feature_vector()
        assert vec[0] == 42.0
        assert vec[1] == 99.0
        assert vec[2] == 3.0

    def test_default_aoi_timer_zero(self, make_node):
        """AoI timer should start at 0."""
        node = make_node()
        assert node.aoi_timer == 0.0

    def test_buffer_capacity_positive(self, make_node):
        """Buffer capacity must be positive."""
        node = make_node(buffer_capacity=50.0)
        assert node.buffer_capacity > 0

    def test_node_ids_unique(self, sample_nodes):
        """All nodes in a sample set should have unique IDs."""
        ids = [n.id for n in sample_nodes]
        assert len(ids) == len(set(ids))
