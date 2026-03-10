from __future__ import annotations

import random

from config.settings import (
    SEED, MAP_W, MAP_H, NODE_COUNT, NODE_MARGIN, OBSTACLE_COUNT,
    Node, Obstacle,
)


def deploy_nodes(n: int = NODE_COUNT, seed: int = SEED) -> list[Node]:
    # place n sensors at random positions with priorities 1-10
    rng = random.Random(seed)
    nodes: list[Node] = []
    for i in range(n):
        nodes.append({
            "id": i,
            "x": rng.randint(NODE_MARGIN, MAP_W - NODE_MARGIN),
            "y": rng.randint(NODE_MARGIN, MAP_H - NODE_MARGIN),
            "priority": rng.randint(1, 10),
            "data_level": round(rng.uniform(0.1, 1.0), 2),
        })
    return nodes


def deploy_obstacles(k: int = OBSTACLE_COUNT, seed: int = SEED) -> list[Obstacle]:
    # place k rectangular obstacles (cuboids with height)
    rng = random.Random(seed + 100)
    obstacles: list[Obstacle] = []
    for i in range(k):
        x1: int = rng.randint(60, MAP_W - 200)
        y1: int = rng.randint(60, MAP_H - 180)
        w: int = rng.randint(80, 150)
        h: int = rng.randint(70, 130)
        height: int = rng.randint(25, 50)
        obstacles.append({
            "id": i,
            "x1": x1, "y1": y1,
            "x2": x1 + w, "y2": y1 + h,
            "cx": x1 + w / 2, "cy": y1 + h / 2,
            "width": w, "depth": h, "height": height,
        })
    return obstacles
