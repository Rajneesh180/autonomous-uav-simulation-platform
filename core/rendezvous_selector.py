"""
Rendezvous Point (RP) Selection Module
=======================================
Implements the Greedy Neighbourhood RP Selection algorithm from:
    Donipati et al. "Optimizing UAV-Based Data Collection in IoT Networks
    with Dynamic Service Time and Buffer-Aware Trajectory Planning"
    IEEE TNSM, April 2025 — Section III-A, Algorithm 1.

Core idea: compress a dense IoT node set N into a minimal RP subset R ⊆ N
such that each non-RP node lies within radius R_max of at least one RP.
The UAV then visits only R, collecting data on behalf of all member nodes.

Complexity: O(|N|²) — dominated by pairwise distance computation.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from config.config import Config
from core.models.node_model import Node


class RendezvousSelector:
    """
    Greedy Neighbourhood Rendezvous Point (RP) Selector.

    Parameters
    ----------
    r_max : float
        Maximum radio transmission range of each ground IoT node (metres).
        Nodes within this radius of an RP can offload data to it directly.
        Defaults to Config.RP_COVERAGE_RADIUS.
    """

    def __init__(self, r_max: float = None):
        self.r_max = r_max if r_max is not None else Config.RP_COVERAGE_RADIUS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self, nodes: List[Node], obstacles: list = None) -> Tuple[List[Node], dict]:
        """
        Execute the greedy RP selection algorithm.

        Parameters
        ----------
        nodes     : list of Node — the full ground-node set (exclude UAV anchor 0).
        obstacles : list of Obstacle — if provided, nodes too close to any obstacle
                    footprint are excluded from RP candidates to avoid trap zones.

        Returns
        -------
        rp_nodes   : list of Node — selected Rendezvous Point nodes.
        member_map : dict {rp_id: [member_node_ids]}
        """
        if not nodes:
            return [], {}

        # Filter out nodes that are too close to obstacles (Fix 3 — avoid trap zones)
        buffer_r = Config.RP_OBSTACLE_BUFFER
        if obstacles and buffer_r > 0:
            safe_nodes = [n for n in nodes if not self._near_obstacle(n, obstacles, buffer_r)]
            # keep all nodes if filtering is too aggressive (< 3 safe nodes)
            if len(safe_nodes) >= 3:
                nodes = safe_nodes

        node_map = {n.id: n for n in nodes}

        # ---- Step 1: compute pairwise neighbourhoods ----
        neighbourhood = {n.id: self._neighbours(n, nodes) for n in nodes}

        # ---- Step 2: greedy dominating set selection ----
        remaining_ids = set(n.id for n in nodes)
        rp_ids: List[int] = []
        member_map: dict = {}

        while remaining_ids:
            # Pick the node with the most uncovered neighbours
            best_id = max(
                remaining_ids,
                key=lambda nid: len(neighbourhood[nid] & remaining_ids)
            )
            rp_ids.append(best_id)

            # Covered set = best node + all its neighbours still uncovered
            covered_now = (neighbourhood[best_id] & remaining_ids) | {best_id}
            member_map[best_id] = list(covered_now - {best_id})

            remaining_ids -= covered_now

        rp_nodes = [node_map[rid] for rid in rp_ids if rid in node_map]

        print(
            f"[RendezvousSelector] N={len(nodes)} nodes → "
            f"{len(rp_ids)} RPs selected "
            f"(r_max={self.r_max:.1f} m, "
            f"reduction={100*(1-len(rp_ids)/len(nodes)):.1f}%)"
        )

        return rp_nodes, member_map

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _neighbours(self, node: Node, all_nodes: List[Node]) -> set:
        """
        Return the set of node IDs within r_max of *node*, including itself.
        """
        return {
            n.id for n in all_nodes
            if self._dist(node, n) <= self.r_max
        }

    @staticmethod
    def _near_obstacle(node: Node, obstacles: list, buffer_r: float) -> bool:
        """Return True if node is within buffer_r of any obstacle bounding box."""
        for obs in obstacles:
            cx = max(obs.x1, min(node.x, obs.x2))
            cy = max(obs.y1, min(node.y, obs.y2))
            if math.hypot(node.x - cx, node.y - cy) < buffer_r:
                return True
        return False

    @staticmethod
    def _dist(a: Node, b: Node) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    # ------------------------------------------------------------------
    # Static convenience wrapper
    # ------------------------------------------------------------------

    @staticmethod
    def apply(nodes: List[Node], r_max: float = None,
              obstacles: list = None) -> Tuple[List[Node], dict]:
        """
        Single-call convenience wrapper. Returns (rp_nodes, member_map).
        """
        selector = RendezvousSelector(r_max=r_max)
        return selector.select(nodes, obstacles=obstacles)
