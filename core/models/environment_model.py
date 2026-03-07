import random

from typing import List, Optional, Tuple
from core.models.node_model import Node, UAVState, SensorNode

COLLISION_MARGIN = 5


class Environment:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # -------- Core Containers --------
        self.nodes: List[Node] = []
        # Typed accessors for Stage 1 entity separation: env.uav and env.sensors
        # self.uav is set to nodes[0] when the first node is added via add_node().
        self.uav: Optional[UAVState] = None

        # -------- Reserved Containers (Phase-2 Ready) --------
        self.cluster_centers = []
        self.obstacles = []
        self.no_fly_zones = []
        self.risk_zones = []

        self.dataset_mode = "random"
        self.environment_changed = False

    # ---------------- Nodes ----------------
    @property
    def sensors(self) -> List[SensorNode]:
        """All ground IoT sensor nodes (nodes[1:])."""
        return self.nodes[1:]

    def add_node(self, node: Node):
        self.nodes.append(node)
        if self.uav is None:
            self.uav = node  # First node added is always the UAV anchor

    def get_node_count(self) -> int:
        return len(self.nodes)

    def remove_random_node(self, min_floor=5):
        if len(self.nodes) <= min_floor:
            return False

        # never remove UAV anchor node
        removable = self.nodes[1:]
        if not removable:
            return False

        node = random.choice(removable)
        self.nodes.remove(node)
        self.mark_changed()
        return True

    # ---------------- Obstacles ----------------
    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def add_no_fly_zone(self, zone):
        self.no_fly_zones.append(zone)

    def add_risk_zone(self, zone):
        self.risk_zones.append(zone)

    # ---------------- Collision Checks ----------------

    def has_collision(
        self, start_pos, end_pos
    ) -> bool:
        """
        Soft collision check using sampled points along the path.
        Accepts 2D (x, y) or 3D (x, y, z) tuples.  When z is provided the
        check uses Obstacle.contains_point_3d — UAV flying above the obstacle
        ceiling is not a collision.
        """
        steps = 8  # sampling resolution
        has_z = len(start_pos) >= 3 and len(end_pos) >= 3

        for obs in self.obstacles:
            for i in range(1, steps + 1):
                t = i / steps
                x = start_pos[0] + t * (end_pos[0] - start_pos[0])
                y = start_pos[1] + t * (end_pos[1] - start_pos[1])

                if has_z:
                    z = start_pos[2] + t * (end_pos[2] - start_pos[2])
                    if obs.contains_point_3d(x, y, z):
                        return True
                else:
                    if obs.contains_point(x, y):
                        return True

        return False

    def risk_multiplier(self, pos: Tuple[float, float]) -> float:
        """
        Soft constraint — increases path cost but does not block movement.
        """
        multiplier = 1.0
        for zone in self.risk_zones:
            if zone.contains_point(pos[0], pos[1]):
                multiplier *= zone.current_multiplier
        return multiplier

    def point_in_obstacle(self, pos, z=None):
        """Check if a point is inside any obstacle.

        If *z* is provided (or *pos* has 3 elements), uses the 3D Gaussian
        ceiling check — returns False when the point is above the obstacle.
        Otherwise falls back to the legacy 2D bounding-box test.
        """
        x, y = pos[0], pos[1]
        if z is None and len(pos) >= 3:
            z = pos[2]
        for obs in self.obstacles:
            if z is not None:
                if obs.contains_point_3d(x, y, z):
                    return True
            else:
                if obs.contains_point(x, y):
                    return True
        return False

    def mark_changed(self):
        self.environment_changed = True

    def reset_change_flag(self):
        self.environment_changed = False

    def update_risk_zones(self, step):
        for zone in self.risk_zones:
            zone.fluctuate(step)

    def update_obstacles(self):
        for obs in self.obstacles:
            if hasattr(obs, "move"):
                obs.move(self.width, self.height)

    def get_safe_start(self, default=(0, 0)):
        x, y = default

        if not self.point_in_obstacle((x, y)):
            return (x, y)

        # search outward
        for r in range(5, 100, 5):
            for dx in (-r, 0, r):
                for dy in (-r, 0, r):
                    candidate = (x + dx, y + dy)
                    if not self.point_in_obstacle(candidate):
                        return candidate

        return default

    # ---------------- Summary ----------------
    def summary(self):
        return {
            "width": self.width,
            "height": self.height,
            "node_count": len(self.nodes),
            "dataset_mode": self.dataset_mode,
            "obstacle_count": len(self.obstacles),
            "no_fly_zone_count": len(self.no_fly_zones),
            "risk_zone_count": len(self.risk_zones),
        }
