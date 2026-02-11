from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Node:
    id: int
    x: float
    y: float

    # -------- Semantic Fields --------
    priority: int = 1
    risk: float = 0.0
    signal_strength: float = 1.0
    deadline: Optional[float] = None

    # -------- Energy / Constraint Placeholders --------
    battery_capacity: float = 100.0
    current_battery: float = 100.0
    energy_per_meter: float = 0.1
    hover_cost: float = 0.0
    return_threshold: float = 0.2  # 20%

    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)
