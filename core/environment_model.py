from typing import List, Tuple
from core.node_model import Node


class Environment:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # -------- Core Containers --------
        self.nodes: List[Node] = []

        # -------- Reserved Containers (Phase-2 Ready) --------
        self.cluster_centers = []
        self.obstacles = []
        self.no_fly_zones = []
        self.risk_zones = []

        self.dataset_mode = "random"
        self.environment_changed = False

    # ---------------- Nodes ----------------
    def add_node(self, node: Node):
        self.nodes.append(node)

    def get_node_count(self) -> int:
        return len(self.nodes)

    # ---------------- Obstacles ----------------
    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def add_no_fly_zone(self, zone):
        self.no_fly_zones.append(zone)

    def add_risk_zone(self, zone):
        self.risk_zones.append(zone)

    # ---------------- Collision Checks ----------------
    def has_collision(
        self, start_pos: Tuple[float, float], end_pos: Tuple[float, float]
    ) -> bool:
        """
        Hard collision check against obstacles and no-fly zones.
        """
        for obs in self.obstacles:
            if obs.intersects_line(start_pos[0], start_pos[1], end_pos[0], end_pos[1]):
                return True

        for zone in self.no_fly_zones:
            if zone.intersects_line(start_pos[0], start_pos[1], end_pos[0], end_pos[1]):
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

    def point_in_obstacle(self, pos):
        x, y = pos
        for obs in self.obstacles:
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
