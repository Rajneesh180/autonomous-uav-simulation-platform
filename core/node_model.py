from dataclasses import dataclass, field
from typing import Optional, Tuple

from config.config import Config


@dataclass
class Node:
    id: int
    x: float
    y: float
    z: float = 0.0

    # ---------------------------------------------------------
    # Semantic Fields
    # ---------------------------------------------------------
    priority: int = 1
    risk: float = 0.0
    signal_strength: float = 1.0
    deadline: Optional[float] = None
    reliability: float = 1.0

    # ---------------------------------------------------------
    # Buffer & Data Profile (DST-BA Implementation)
    # ---------------------------------------------------------
    buffer_capacity: float = field(default_factory=lambda: Config.DEFAULT_BUFFER_CAP_MBITS)
    current_buffer: float = 0.0
    data_generation_rate: float = field(default_factory=lambda: Config.DEFAULT_DATA_RATE_MBPS)
    
    time_window_start: float = 0.0
    time_window_end: float = float("inf")

    # ---------------------------------------------------------
    # Energy Model (Initialized from Config)
    # ---------------------------------------------------------
    battery_capacity: float = field(default_factory=lambda: Config.BATTERY_CAPACITY)
    current_battery: float = field(init=False)

    energy_per_meter: float = field(default_factory=lambda: Config.ENERGY_PER_METER)
    hover_cost: float = field(default_factory=lambda: Config.HOVER_COST)
    return_threshold: float = field(default_factory=lambda: Config.RETURN_THRESHOLD)

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------

    def __post_init__(self):
        self.current_battery = self.battery_capacity

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def reset_battery(self):
        self.current_battery = self.battery_capacity

    def get_feature_vector(self) -> list:
        """
        Extracts the node properties as a numeric vector for scaling and semantic clustering.
        Vector: [x, y, z, priority, risk, signal_strength, deadline/time_window, buffer, reliability]
        """
        safe_deadline = self.time_window_end if self.time_window_end != float('inf') else 9999.0
        return [
            self.x, 
            self.y, 
            self.z, 
            float(self.priority), 
            self.risk, 
            self.signal_strength, 
            safe_deadline, 
            self.current_buffer,
            self.reliability
        ]
