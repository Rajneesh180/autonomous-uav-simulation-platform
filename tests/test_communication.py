"""
Tests for core.comms.communication.CommunicationEngine
Validates channel modelling, LoS probability, path loss, Shannon rate, and sensing trials.
"""

import math
import pytest

from core.comms.communication import CommunicationEngine
from config.config import Config


class TestElevationAngle:
    """Validate elevation angle computation."""

    def test_directly_above(self):
        """UAV directly above node → 90° elevation."""
        angle = CommunicationEngine.elevation_angle(
            (100, 100, 0), (100, 100, Config.UAV_FLIGHT_ALTITUDE)
        )
        assert abs(angle - math.pi / 2) < 1e-6

    def test_far_away_low_angle(self):
        """UAV far horizontally → small elevation angle."""
        angle = CommunicationEngine.elevation_angle(
            (0, 0, 0), (1000, 0, Config.UAV_FLIGHT_ALTITUDE)
        )
        assert 0 < angle < math.pi / 4, "Far horizontal should yield low elevation"

    def test_angle_always_positive(self):
        """Elevation angle should always be non-negative."""
        for dx in [10, 50, 200, 500]:
            angle = CommunicationEngine.elevation_angle(
                (0, 0, 0), (dx, 0, Config.UAV_FLIGHT_ALTITUDE)
            )
            assert angle >= 0


class TestProbLoS:
    """Validate LoS probability sigmoid model."""

    def test_high_elevation_high_prob(self):
        """Near-vertical → high LoS probability."""
        p = CommunicationEngine.prob_los(math.radians(85), True)
        assert p > 0.8

    def test_low_elevation_lower_prob(self):
        """Low elevation → reduced LoS probability."""
        p_high = CommunicationEngine.prob_los(math.radians(80), True)
        p_low = CommunicationEngine.prob_los(math.radians(10), True)
        assert p_high >= p_low

    def test_nlos_always_lower(self):
        """Obstructed path should yield lower LoS probability than clear path."""
        angle = math.radians(45)
        p_clear = CommunicationEngine.prob_los(angle, True)
        p_blocked = CommunicationEngine.prob_los(angle, False)
        assert p_blocked <= p_clear


class TestPathLoss:
    """Validate FSPL blended path loss model."""

    def test_loss_increases_with_distance(self):
        """Path loss should increase with distance."""
        l_near = CommunicationEngine.path_loss(50, 0.9)
        l_far = CommunicationEngine.path_loss(500, 0.9)
        assert l_far > l_near

    def test_higher_plos_lower_loss(self):
        """Better LoS → lower expected loss (LoS exponent < NLoS exponent)."""
        l_good = CommunicationEngine.path_loss(100, 0.95)
        l_bad = CommunicationEngine.path_loss(100, 0.1)
        assert l_good < l_bad


class TestAchievableDataRate:
    """Validate Shannon capacity computation."""

    def test_close_uav_high_rate(self):
        """UAV close to node should yield a positive data rate."""
        rate = CommunicationEngine.achievable_data_rate(
            (100, 100, 0), (100, 100, Config.UAV_FLIGHT_ALTITUDE)
        )
        assert rate > 0, "Rate should be positive when UAV is close"

    def test_rate_decreases_with_distance(self):
        """Rate should decrease as UAV moves farther away."""
        close = CommunicationEngine.achievable_data_rate(
            (100, 100, 0), (100, 100, Config.UAV_FLIGHT_ALTITUDE)
        )
        far = CommunicationEngine.achievable_data_rate(
            (100, 100, 0), (800, 100, Config.UAV_FLIGHT_ALTITUDE)
        )
        assert close > far


class TestSensingTrials:
    """Validate multi-trial sensing model (Zheng & Liu, IEEE TVT 2025)."""

    def test_close_distance_single_trial(self):
        """Very short distance → single sensing trial suffices."""
        n = CommunicationEngine.required_sensing_trials(1.0)
        assert n == 1

    def test_longer_distance_more_trials(self):
        """Larger distance requires more trials to reach cumulative threshold."""
        n_close = CommunicationEngine.required_sensing_trials(10.0)
        n_far = CommunicationEngine.required_sensing_trials(100.0)
        assert n_far >= n_close

    def test_minimum_hover_time_positive(self):
        """Hover time must be strictly positive for any reachable distance."""
        t = CommunicationEngine.minimum_hover_time(50.0)
        assert t > 0


class TestNodeTXEnergy:
    """Validate first-order radio energy model."""

    def test_zero_bits_zero_energy(self):
        assert CommunicationEngine.compute_node_tx_energy(0, 100) == 0.0

    def test_energy_increases_with_distance(self):
        e_close = CommunicationEngine.compute_node_tx_energy(1e6, 10)
        e_far = CommunicationEngine.compute_node_tx_energy(1e6, 100)
        assert e_far > e_close, "TX energy should increase with d^2"

    def test_energy_increases_with_bits(self):
        e_small = CommunicationEngine.compute_node_tx_energy(1e3, 50)
        e_big = CommunicationEngine.compute_node_tx_energy(1e6, 50)
        assert e_big > e_small
