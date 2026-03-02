import random
from core.models.node_model import Node


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

    elif mode == "priority_heavy":
        for i in range(count):
            x = random.randint(0, width)
            y = random.randint(0, height)
            node = Node(id=i, x=x, y=y)
            # 20% of nodes are critically important
            node.priority = 10 if random.random() < 0.2 else 1
            nodes.append(node)

    elif mode == "deadline_critical":
        for i in range(count):
            x = random.randint(0, width)
            y = random.randint(0, height)
            node = Node(id=i, x=x, y=y)
            # 30% of nodes have extremely tight time windows
            if random.random() < 0.3:
                node.time_window_end = random.uniform(50.0, 150.0) # Tight bound in steps
            nodes.append(node)

    elif mode == "risk_dense":
        for i in range(count):
            x = random.randint(0, width)
            y = random.randint(0, height)
            node = Node(id=i, x=x, y=y)
            # Higher risk near the map center
            dist_to_center = ((x - width/2)**2 + (y - height/2)**2)**0.5
            if dist_to_center < min(width, height) * 0.3:
                node.risk = random.uniform(0.7, 1.0)
            else:
                node.risk = random.uniform(0.0, 0.2)
            nodes.append(node)

    elif mode == "mixed_feature":
        for i in range(count):
            x = random.randint(0, width)
            y = random.randint(0, height)
            node = Node(id=i, x=x, y=y)
            node.priority = random.choice([1, 5, 10])
            node.risk = random.uniform(0.0, 0.9)
            if random.random() < 0.5:
                node.time_window_end = random.uniform(100.0, 500.0)
            # Simulate data generation variance
            node.current_buffer = random.uniform(1.0, node.buffer_capacity)
            nodes.append(node)
    else:
        # Fallback
        for i in range(count):
            nodes.append(Node(id=i, x=random.randint(0, width), y=random.randint(0, height)))

    # Physics is always 3D.  Ground IoT sensors sit at z = 0.
    for node in nodes:
        node.z = 0.0

    return nodes


def spawn_single_node(width, height, node_id, env):
    import random
    from core.models.node_model import Node

    attempts = 0

    while attempts < 5:
        x = random.randint(0, width)
        y = random.randint(0, height)

        if not env.point_in_obstacle((x, y)):
            # Ground IoT sensor — z is always 0
            return Node(node_id, x, y, z=0.0)

        attempts += 1

    return None
