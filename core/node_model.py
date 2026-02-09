from dataclasses import dataclass

@dataclass
class Node:
    id: int
    x: float
    y: float

    # Future semantic attributes
    priority: int = 1
    risk: float = 0.0
    energy_cost: float = 0.0

    def position(self):
        return (self.x, self.y)
