"""
Core Simulation Engine
======================
Top-level re-exports for the simulation pipeline.

Usage::

    from core import run_simulation, BatchRunner, MissionController
    from core.models import Node, EnergyModel, Environment
    from core.comms import CommunicationEngine, BufferAwareManager
    from core.sensing import DigitalTwinMap, AgentCentricTransform
"""

from core.simulation_runner import run_simulation
from core.batch_runner import BatchRunner
from core.mission_controller import MissionController
from core.dataset_generator import generate_nodes, spawn_single_node
from core.rendezvous_selector import RendezvousSelector
from core.run_manager import RunManager
from core.seed_manager import set_global_seed
from core.stability_monitor import StabilityMonitor
from core.telemetry_logger import TelemetryLogger
from core.temporal_engine import TemporalEngine

__all__ = [
    # Pipeline
    "run_simulation",
    "BatchRunner",
    "MissionController",
    # Data
    "generate_nodes",
    "spawn_single_node",
    # Infrastructure
    "RendezvousSelector",
    "RunManager",
    "set_global_seed",
    "StabilityMonitor",
    "TelemetryLogger",
    "TemporalEngine",
]
