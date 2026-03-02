"""
Communication Layer
===================
Channel models, buffer-aware data collection, and base-station uplink logic.

Re-exports for convenience:
    from core.comms import CommunicationEngine, BufferAwareManager, BaseStationUplinkModel
"""

from core.comms.communication import CommunicationEngine
from core.comms.buffer_aware_manager import BufferAwareManager
from core.comms.base_station_uplink import BaseStationUplinkModel

__all__ = [
    "CommunicationEngine",
    "BufferAwareManager",
    "BaseStationUplinkModel",
]
