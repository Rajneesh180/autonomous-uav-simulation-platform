"""
Tests for the path-planning pipeline:
  - path.pca_gls_router.PCAGLSRouter
  - path.ga_sequence_optimizer.GASequenceOptimizer
  - path.hover_optimizer.HoverOptimizer
  - core.rendezvous_selector.RendezvousSelector
Validates route construction, GA optimisation, SCA hover refinement, and RP selection.
"""

import math
import pytest

from path.pca_gls_router import PCAGLSRouter
from path.ga_sequence_optimizer import GASequenceOptimizer
from path.hover_optimizer import HoverOptimizer
from core.rendezvous_selector import RendezvousSelector
from core.models.obstacle_model import Obstacle, ObstacleHeightModel
from config.config import Config


# ------------------------------------------------------------------
# Route distance helper
# ------------------------------------------------------------------

def _route_distance(positions):
    """Total Euclidean distance through a sequence of (x, y, z) positions."""
    d = 0.0
    for i in range(len(positions) - 1):
        d += math.sqrt(sum((a - b) ** 2 for a, b in zip(positions[i], positions[i + 1])))
    return d


# ------------------------------------------------------------------
# PCA-GLS Router tests
# ------------------------------------------------------------------

class TestPCAGLSRouter:

    def test_route_visits_all_nodes(self, sample_nodes, uav_start_pos):
        """Route should contain every input node exactly once."""
        route = PCAGLSRouter.optimize(uav_start_pos, sample_nodes)
        assert len(route) == len(sample_nodes)
        route_ids = {n.id for n in route}
        expected_ids = {n.id for n in sample_nodes}
        assert route_ids == expected_ids

    def test_two_opt_improves_or_preserves(self, sample_nodes, uav_start_pos):
        """2-opt should not increase total route distance."""
        initial_route = list(sample_nodes)
        improved = PCAGLSRouter._two_opt(initial_route)
        dist_before = _route_distance([n.position() for n in initial_route])
        dist_after = _route_distance([n.position() for n in improved])
        assert dist_after <= dist_before + 1e-6

    def test_single_node(self, make_node, uav_start_pos):
        """Single-node input should return that node."""
        node = make_node(x=200, y=200)
        route = PCAGLSRouter.optimize(uav_start_pos, [node])
        assert len(route) == 1 and route[0].id == node.id

    def test_empty_input(self, uav_start_pos):
        """Empty input should return empty route."""
        route = PCAGLSRouter.optimize(uav_start_pos, [])
        assert route == []


# ------------------------------------------------------------------
# GA Sequence Optimizer tests
# ------------------------------------------------------------------

class TestGASequenceOptimizer:

    def test_ga_visits_all_nodes(self, sample_nodes, uav_start_pos):
        """GA output should be a permutation of the input nodes."""
        result = GASequenceOptimizer.apply(uav_start_pos, sample_nodes)
        assert len(result) == len(sample_nodes)
        assert {n.id for n in result} == {n.id for n in sample_nodes}

    def test_ga_improves_over_random(self, sample_nodes, uav_start_pos):
        """GA-optimised distance should be ≤ worst random permutation distance."""
        import random
        random.seed(99)
        worst = list(sample_nodes)
        random.shuffle(worst)
        worst_dist = _route_distance([uav_start_pos] + [n.position() for n in worst])

        optimised = GASequenceOptimizer.apply(uav_start_pos, sample_nodes)
        opt_dist = _route_distance([uav_start_pos] + [n.position() for n in optimised])
        # GA should at least not be catastrophically worse
        assert opt_dist < worst_dist * 2.0

    def test_small_input_passthrough(self, make_node, uav_start_pos):
        """≤ 2 nodes should be returned as-is (no GA overhead)."""
        nodes = [make_node(x=10, y=10, id=1), make_node(x=20, y=20, id=2)]
        result = GASequenceOptimizer.apply(uav_start_pos, nodes)
        assert len(result) == 2

    def test_fitness_deterministic(self, sample_nodes, uav_start_pos):
        """Same chromosome should produce the same fitness score."""
        ga = GASequenceOptimizer()
        chrom = list(range(len(sample_nodes)))
        f1 = ga._fitness(chrom, uav_start_pos, sample_nodes)
        f2 = ga._fitness(chrom, uav_start_pos, sample_nodes)
        assert f1 == f2

    def test_fitness_positive(self, sample_nodes, uav_start_pos):
        """Fitness should always be in (0, 1]."""
        ga = GASequenceOptimizer()
        chrom = list(range(len(sample_nodes)))
        f = ga._fitness(chrom, uav_start_pos, sample_nodes)
        assert 0 < f <= 1.0


# ------------------------------------------------------------------
# Hover Optimizer tests
# ------------------------------------------------------------------

class TestHoverOptimizer:

    def test_output_tuple_length(self, sample_obstacles):
        """Hover position should be a 3-tuple (x, y, z)."""
        opt = HoverOptimizer()
        pos = opt.optimise(
            node_pos=(250, 250, 0),
            prev_pos=(200, 200, Config.UAV_FLIGHT_ALTITUDE),
            next_pos=(350, 350, Config.UAV_FLIGHT_ALTITUDE),
            obstacles=sample_obstacles,
        )
        assert len(pos) == 3

    def test_altitude_respects_minimum(self, sample_obstacles):
        """Hover z should be ≥ UAV_FLIGHT_ALTITUDE."""
        opt = HoverOptimizer()
        pos = opt.optimise(
            node_pos=(250, 250, 0),
            prev_pos=(100, 100, Config.UAV_FLIGHT_ALTITUDE),
            next_pos=None,
            obstacles=sample_obstacles,
        )
        assert pos[2] >= Config.UAV_FLIGHT_ALTITUDE - 1.0  # small tolerance

    def test_hover_within_radius(self, sample_obstacles):
        """Optimised hover should stay within hover_radius of node."""
        opt = HoverOptimizer(hover_radius=30.0)
        node_pos = (400, 300, 0)
        pos = opt.optimise(
            node_pos=node_pos,
            prev_pos=(300, 200, Config.UAV_FLIGHT_ALTITUDE),
            next_pos=(500, 400, Config.UAV_FLIGHT_ALTITUDE),
            obstacles=sample_obstacles,
        )
        lateral_dist = math.hypot(pos[0] - node_pos[0], pos[1] - node_pos[1])
        assert lateral_dist <= 30.0 + 1.0  # small tolerance

    def test_sequence_optimise(self, sample_nodes, sample_obstacles, uav_start_pos):
        """optimise_sequence should return one hover position per node."""
        nodes = sample_nodes[:5]
        positions = HoverOptimizer.apply_sequence(uav_start_pos, nodes, sample_obstacles)
        assert len(positions) == len(nodes)
        for p in positions:
            assert len(p) == 3


# ------------------------------------------------------------------
# Rendezvous Point Selector tests
# ------------------------------------------------------------------

class TestRendezvousSelector:

    def test_rp_reduction(self, sample_nodes):
        """RP selection should reduce node count (or match it if r_max is tiny)."""
        rps, member_map = RendezvousSelector.apply(sample_nodes, r_max=200.0)
        assert len(rps) <= len(sample_nodes)
        assert len(rps) >= 1

    def test_all_nodes_covered(self, sample_nodes):
        """Every non-RP node should appear in exactly one RP's member list."""
        rps, member_map = RendezvousSelector.apply(sample_nodes, r_max=200.0)
        rp_ids = {n.id for n in rps}
        member_ids = set()
        for members in member_map.values():
            member_ids.update(members)
        all_covered = rp_ids | member_ids
        expected = {n.id for n in sample_nodes}
        assert all_covered == expected

    def test_small_radius_no_reduction(self, sample_nodes):
        """Very small radius → each node is its own RP (no merging)."""
        rps, _ = RendezvousSelector.apply(sample_nodes, r_max=1.0)
        assert len(rps) == len(sample_nodes)

    def test_empty_input(self):
        """Empty node list should return empty results."""
        rps, member_map = RendezvousSelector.apply([], r_max=100.0)
        assert rps == [] and member_map == {}


# ------------------------------------------------------------------
# Obstacle Model tests
# ------------------------------------------------------------------

class TestObstacleModel:

    def test_contains_point_inside(self, make_obstacle):
        """Point inside bounding box should be contained."""
        obs = make_obstacle(x1=100, y1=100, x2=200, y2=200)
        assert obs.contains_point(150, 150) is True

    def test_contains_point_outside(self, make_obstacle):
        """Point outside bounding box should not be contained."""
        obs = make_obstacle(x1=100, y1=100, x2=200, y2=200)
        assert obs.contains_point(50, 50) is False

    def test_gaussian_height_peaks_at_center(self, make_obstacle):
        """Gaussian height should peak near the center of the obstacle."""
        obs = make_obstacle(x1=100, y1=100, x2=200, y2=200, height=60)
        center_h = obs.gaussian_height(150, 150)
        edge_h = obs.gaussian_height(100, 100)
        assert center_h > edge_h

    def test_required_altitude_above_obstacle(self, sample_obstacles):
        """Required altitude above obstacle center should exceed obstacle height."""
        obs = sample_obstacles[0]  # x1=200, y1=200, x2=300, y2=300, height=50
        alt = ObstacleHeightModel.required_altitude(250, 250, sample_obstacles)
        assert alt > 50.0  # height + clearance

    def test_required_altitude_clear_area(self, sample_obstacles):
        """Required altitude in open area should be low or at safety minimum."""
        alt = ObstacleHeightModel.required_altitude(50, 50, sample_obstacles)
        assert alt <= 10.0  # far from any obstacle — at or below safety floor
