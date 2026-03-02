"""
Tests for core.models.energy_model.EnergyModel
Validates rotary-wing propulsion physics, energy accounting, and safety checks.
"""

import math
import pytest

from core.models.energy_model import EnergyModel
from config.config import Config


class TestPropulsionPower:
    """Validate the rotary-wing propulsion power curve P_p(v)."""

    def test_hover_power_at_zero_speed(self):
        """P(0) = P_0 + P_i — profile + induced power, zero parasite drag."""
        power = EnergyModel.propulsion_power(0.0)
        expected = Config.PROFILE_POWER_HOVER + Config.INDUCED_POWER_HOVER
        assert abs(power - expected) < 0.01, f"Hover power mismatch: {power} vs {expected}"

    def test_power_increases_at_high_speed(self):
        """Parasite drag should dominate at high speed → P(30) > P(10)."""
        p10 = EnergyModel.propulsion_power(10.0)
        p30 = EnergyModel.propulsion_power(30.0)
        assert p30 > p10, "Power should increase at high speed due to parasite drag"

    def test_power_always_positive(self):
        """Power must be strictly positive for all non-negative speeds."""
        for v in [0.0, 1.0, 5.0, 15.0, 25.0, 40.0]:
            assert EnergyModel.propulsion_power(v) > 0


class TestEnergyForDistance:
    """Validate distance-based energy computation."""

    def test_zero_distance_minimal_energy(self, make_node):
        """Zero distance should produce energy close to zero (hover base cost)."""
        node = make_node()
        energy = EnergyModel.energy_for_distance(node, 0.0)
        e100 = EnergyModel.energy_for_distance(node, 100.0)
        assert energy < e100, "Zero-distance energy should be less than 100 m flight"

    def test_energy_proportional_to_distance(self, make_node):
        """Doubling distance should roughly double energy (constant speed assumption)."""
        node = make_node()
        e1 = EnergyModel.energy_for_distance(node, 100.0)
        e2 = EnergyModel.energy_for_distance(node, 200.0)
        assert e2 > e1 * 1.5, "Energy should scale with distance"

    def test_energy_is_positive(self, make_node):
        node = make_node()
        assert EnergyModel.energy_for_distance(node, 50.0) > 0


class TestHoverEnergy:
    """Validate hover energy computation."""

    def test_hover_zero_time(self, make_node):
        node = make_node()
        assert EnergyModel.hover_energy(node, 0.0) == 0.0

    def test_hover_energy_proportional_to_time(self, make_node):
        node = make_node()
        e5 = EnergyModel.hover_energy(node, 5.0)
        e10 = EnergyModel.hover_energy(node, 10.0)
        assert abs(e10 - 2.0 * e5) < 1e-6, "Hover energy should be linear in time"


class TestCanTravel:
    """Validate feasibility checks."""

    def test_full_battery_can_travel(self, make_node):
        node = make_node(battery_capacity=600000.0)
        assert EnergyModel.can_travel(node, 100.0) is True

    def test_depleted_battery_cannot_travel(self, make_node):
        node = make_node(battery_capacity=600000.0, current_battery=0.0)
        assert EnergyModel.can_travel(node, 100.0) is False


class TestConsume:
    """Validate energy consumption."""

    def test_consume_reduces_battery(self, make_node):
        node = make_node(battery_capacity=1000.0)
        EnergyModel.consume(node, 200.0)
        assert node.current_battery == 800.0

    def test_consume_floors_at_zero(self, make_node):
        node = make_node(battery_capacity=100.0, current_battery=50.0)
        EnergyModel.consume(node, 200.0)
        assert node.current_battery == 0.0


class TestCanReturnToBase:
    """Validate the conservative return-to-base feasibility check."""

    def test_return_with_full_battery(self, make_node):
        node = make_node(battery_capacity=600000.0)
        current_pos = (200.0, 200.0, Config.UAV_FLIGHT_ALTITUDE)
        base_pos = (0.0, 0.0, Config.UAV_FLIGHT_ALTITUDE)
        assert EnergyModel.can_return_to_base(node, current_pos, base_pos) is True

    def test_return_fails_when_depleted(self, make_node):
        node = make_node(battery_capacity=600000.0, current_battery=1.0)
        current_pos = (500.0, 500.0, Config.UAV_FLIGHT_ALTITUDE)
        base_pos = (0.0, 0.0, Config.UAV_FLIGHT_ALTITUDE)
        assert EnergyModel.can_return_to_base(node, current_pos, base_pos) is False


class TestShouldReturn:
    """Validate preventive return trigger."""

    def test_not_triggered_with_full_battery(self, make_node):
        node = make_node(battery_capacity=600000.0)
        assert EnergyModel.should_return(node) is False

    def test_triggered_with_low_battery(self, make_node):
        node = make_node(battery_capacity=600000.0, current_battery=600000.0 * 0.05)
        assert EnergyModel.should_return(node) is True
