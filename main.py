#!/usr/bin/env python3

from __future__ import annotations

import argparse

from config.settings import (
    MAP_W, MAP_H, BATTERY_CAPACITY,
    Node, Obstacle,
)
from core.deployment import deploy_nodes, deploy_obstacles
from core.rp_selection import select_rendezvous_points
from core.path_planning import compute_expected_path
from core.energy import estimate_energy
from visualization.diagrams import generate_all_diagrams


# runs all 4 steps and prints results
def run_simulation() -> tuple[list[Node], list[Obstacle], list[int], dict[int, list[int]]]:
    print("UAV Data Collection - Phase 1 Simulation")
    print("-" * 40)

    # Step 1 - place sensor nodes
    nodes: list[Node] = deploy_nodes()
    p_lo = min(n["priority"] for n in nodes)
    p_hi = max(n["priority"] for n in nodes)
    print(f"\n[nodes]  {len(nodes)} sensors placed in {MAP_W}x{MAP_H} m area "
          f"(priority {p_lo}-{p_hi})")

    # Step 2 - place obstacles
    obstacles: list[Obstacle] = deploy_obstacles()
    print(f"[obstacles]  {len(obstacles)} rectangular obstacles:")
    for o in obstacles:
        print(f"   #{o['id']}  ({o['x1']},{o['y1']})->({o['x2']},{o['y2']})  "
              f"{o['width']}x{o['depth']}x{o['height']}m")

    # Step 3 - pick rendezvous points
    rps, members = select_rendezvous_points(nodes, obstacles)
    ratio = 100 * (1 - len(rps) / len(nodes))
    print(f"\n[RP select]  {len(rps)} RPs out of {len(nodes)} nodes  ({ratio:.0f}% reduced)")
    for rp in rps:
        mlist = [f"N{m}" for m in members.get(rp, [])]
        print(f"   N{rp} ({nodes[rp]['x']},{nodes[rp]['y']})  ->  {mlist}")

    # Step 4 - compute path & energy
    path, total_dist = compute_expected_path(nodes, rps)
    e_total, pct = estimate_energy(total_dist, len(rps))
    print(f"\n[path]  BS -> {len(rps)} RPs -> BS  ({len(path)} waypoints)")
    print(f"   distance = {total_dist:.1f} m")
    print(f"   energy   = {e_total / 1000:.1f} kJ  "
          f"({pct:.1f}% of {BATTERY_CAPACITY / 1000:.0f} kJ capacity)")

    print("\ndone.")
    return nodes, obstacles, rps, members


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UAV-Assisted Obstacle-Aware Data Collection Framework",
    )
    parser.add_argument(
        "--diagrams", action="store_true",
        help="Also generate report diagrams after simulation",
    )
    parser.add_argument(
        "--diagrams-only", action="store_true",
        help="Only generate diagrams (skip simulation)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run simulation + generate diagrams (same as --diagrams)",
    )
    args = parser.parse_args()

    # --diagrams-only skips the simulation, just makes the figures
    if args.diagrams_only:
        nodes = deploy_nodes()
        obstacles = deploy_obstacles()
        rps, members = select_rendezvous_points(nodes, obstacles)
        generate_all_diagrams(nodes, obstacles, rps, members)
        return

    nodes, obstacles, rps, members = run_simulation()

    if args.diagrams or args.all:
        print()
        generate_all_diagrams(nodes, obstacles, rps, members)


if __name__ == "__main__":
    main()
