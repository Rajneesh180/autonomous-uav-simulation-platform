import random
from core.node_model import Node


def generate_nodes(
    mode: str, count: int, width: int, height: int, seed: int | None = None
):
    if seed is not None:
        random.seed(seed)

    nodes = []

    if mode == "random":
        for i in range(count):
            x = random.randint(0, width)
            y = random.randint(0, height)
            nodes.append(Node(id=i, x=x, y=y))

    # Future modes can be added here (spiral, clustered, etc.)
    return nodes


def spawn_single_node(width, height, node_id, env):
    import random
    from core.node_model import Node

    attempts = 0

    while attempts < 5:
        x = random.randint(0, width)
        y = random.randint(0, height)

        if not env.point_in_obstacle((x, y)):
            return Node(node_id, x, y)

        attempts += 1

    return None
