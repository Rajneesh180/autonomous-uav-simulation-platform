# all simulation constants and type aliases

from __future__ import annotations

import os
from typing import Any

# type aliases
Node = dict[str, Any]
Obstacle = dict[str, Any]
Coord = tuple[float, float]

# seed for reproducibility
SEED: int = 42

# deployment area
MAP_W: int = 800          # metres
MAP_H: int = 600          # metres

# sensor network
NODE_COUNT: int = 20
NODE_MARGIN: int = 40     # min distance from boundary (metres)

# obstacles
OBSTACLE_COUNT: int = 3

# rendezvous points
RP_RADIUS: float = 120.0  # communication radius R_max (metres)

# uav params
BASE_STATION: Coord = (400, 300)   # centre of deployment area
UAV_ALTITUDE: int = 100            # metres

# energy model (rotary-wing) - Eq. 3-5 in report
BATTERY_CAPACITY: float = 600_000.0   # Joules
E_FLY_PER_M: float = 17.0            # J/m at cruise speed
P_HOVER: float = 168.483             # Watts   (P_0 + P_i)
P_0: float = 79.856                  # blade profile power (W)
P_i: float = 88.627                  # induced power (W)
CRUISE_SPEED: float = 10.0           # m/s
HOVER_TIME: float = 10.0             # seconds per RP

# output dir for figures
_PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR: str = os.path.join(_PROJECT_ROOT, "report", "figures")
