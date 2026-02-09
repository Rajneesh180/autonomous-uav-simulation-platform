from typing import List
from core.node_model import Node

class Environment:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.nodes: List[Node] = []

    def add_node(self, node: Node):
        self.nodes.append(node)

    def get_node_count(self):
        return len(self.nodes)

    def summary(self):
        return {
            "width": self.width,
            "height": self.height,
            "node_count": len(self.nodes)
        }
