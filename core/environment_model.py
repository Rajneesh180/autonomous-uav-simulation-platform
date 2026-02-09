from typing import List
from core.node_model import Node

class Environment:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.nodes = []

        # -------- Reserved Containers (Phase-2 Ready) --------
        self.cluster_centers = []
        self.obstacles = []
        self.no_fly_zones = []
        self.risk_zones = []

        self.dataset_mode = "random"

    def add_node(self, node):
        self.nodes.append(node)

    def get_node_count(self):
        return len(self.nodes)

    def summary(self):
        return {
            "width": self.width,
            "height": self.height,
            "node_count": len(self.nodes),
            "dataset_mode": self.dataset_mode
        }
