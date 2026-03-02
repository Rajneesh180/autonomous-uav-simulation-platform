"""
Shared pytest fixtures for the UAV Simulation Platform test suite.
Provides reusable Node, Obstacle, Environment, and configuration factories.
"""

import pytest
import math

from core.models.node_model import Node
from core.models.obstacle_model import Obstacle, ObstacleHeightModel
from core.models.energy_model import EnergyModel
from core.models.environment_model import Environment
from core.dataset_generator import generate_nodes
from config.config import Config


# ------------------------------------------------------------------
# Node factories
# ------------------------------------------------------------------

@pytest.fixture
def make_node():
    """Factory fixture — returns a function that creates a Node with sensible defaults."""
    _counter = [0]

    def _factory(
        x=100.0, y=100.0, z=0.0,
        priority=5, risk=0.1, signal_strength=0.8,
        deadline=500.0, reliability=0.9,
        buffer_capacity=50.0, current_buffer=25.0,
        data_generation_rate=0.5,
        battery_capacity=None,
        **kwargs,
    ) -> Node:
        _counter[0] += 1
        nid = kwargs.pop("id", _counter[0])
        bc = battery_capacity or Config.BATTERY_CAPACITY
        node = Node(
            id=nid, x=x, y=y, z=z,
            priority=priority, risk=risk,
            signal_strength=signal_strength,
            reliability=reliability,
            buffer_capacity=buffer_capacity,
            current_buffer=current_buffer,
            data_generation_rate=data_generation_rate,
            battery_capacity=bc,
            time_window_end=deadline,
        )
        # current_battery is init=False; override if caller wants a specific value
        if "current_battery" in kwargs:
            node.current_battery = kwargs.pop("current_battery")
        for k, v in kwargs.items():
            if hasattr(node, k):
                setattr(node, k, v)
        return node

    return _factory


@pytest.fixture
def sample_nodes(make_node):
    """A list of 10 spatially diverse nodes for routing / clustering tests."""
    positions = [
        (50, 50), (150, 80), (300, 200), (400, 100), (500, 300),
        (600, 150), (700, 400), (250, 450), (100, 350), (450, 500),
    ]
    return [
        make_node(x=px, y=py, id=i + 1, priority=i + 1)
        for i, (px, py) in enumerate(positions)
    ]


# ------------------------------------------------------------------
# Obstacle factories
# ------------------------------------------------------------------

@pytest.fixture
def make_obstacle():
    """Factory fixture for creating Obstacle instances."""

    def _factory(x1=200.0, y1=200.0, x2=300.0, y2=300.0, height=50.0):
        return Obstacle(x1=x1, y1=y1, x2=x2, y2=y2, height=height)

    return _factory


@pytest.fixture
def sample_obstacles(make_obstacle):
    """Two non-overlapping obstacles."""
    return [
        make_obstacle(x1=200, y1=200, x2=300, y2=300, height=50),
        make_obstacle(x1=500, y1=100, x2=600, y2=200, height=70),
    ]


# ------------------------------------------------------------------
# Environment factory
# ------------------------------------------------------------------

@pytest.fixture
def sample_environment(sample_nodes, sample_obstacles):
    """An Environment populated with sample nodes and obstacles."""
    return Environment(
        nodes=sample_nodes,
        obstacles=sample_obstacles,
        width=Config.MAP_WIDTH,
        height=Config.MAP_HEIGHT,
    )


# ------------------------------------------------------------------
# UAV position helpers
# ------------------------------------------------------------------

@pytest.fixture
def uav_start_pos():
    """Default UAV start position (center of map, at flight altitude)."""
    return (Config.MAP_WIDTH / 2, Config.MAP_HEIGHT / 2, Config.UAV_FLIGHT_ALTITUDE)


@pytest.fixture
def uav_above_node():
    """UAV hovering directly above (100, 100) at flight altitude."""
    return (100.0, 100.0, Config.UAV_FLIGHT_ALTITUDE)
