from dataclasses import dataclass, field
from typing import Optional, Tuple

from config.config import Config


@dataclass
class Node:
    id: int
    x: float
    y: float

    # ---------------------------------------------------------
    # Semantic Fields (Phase-4 ready but passive in Phase-3)
    # ---------------------------------------------------------
    priority: int = 1
    risk: float = 0.0
    signal_strength: float = 1.0
    deadline: Optional[float] = None

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

    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def reset_battery(self):
        self.current_battery = self.battery_capacity
