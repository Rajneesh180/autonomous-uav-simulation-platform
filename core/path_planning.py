# nearest-neighbour path through RPs (BS -> RPs -> BS)

from __future__ import annotations

import math

from config.settings import BASE_STATION, Coord, Node


def compute_expected_path(
    nodes: list[Node],
    rps: list[int],
    base: Coord = BASE_STATION,
) -> tuple[list[Coord], float]:
    # returns (waypoints, total distance)
    if not rps:
        return [base, base], 0.0

    rp_coords: list[Coord] = [(nodes[r]["x"], nodes[r]["y"]) for r in rps]
    path: list[Coord] = [base]
    remaining: list[int] = list(range(len(rps)))
    current: Coord = base

    while remaining:
        # pick the closest unvisited RP from current position
        closest: int = min(
            remaining,
            key=lambda i: math.hypot(
                rp_coords[i][0] - current[0],
                rp_coords[i][1] - current[1],
            ),
        )
        path.append(rp_coords[closest])
        current = rp_coords[closest]
        remaining.remove(closest)

    path.append(base)
    total_dist: float = sum(
        math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    )
    return path, total_dist
