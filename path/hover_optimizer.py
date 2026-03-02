"""
SCA-Inspired Hover Position Optimizer
=======================================
Implements hover position refinement via a Successive Convex Approximation (SCA)-
inspired gradient descent, aligned with:
    Zheng & Liu, "3D UAV Trajectory Planning With Obstacle Avoidance for
    Time-Constrained Data Collection", IEEE TVT, January 2025 — Section III-E.

For each Rendezvous Point in the visiting sequence, this module refines the exact
(x, y, z) hover position to minimise total travel cost (distance to previous and
next waypoints) while satisfying altitude constraints.

The SCA-inspired update is:
    p̃^{(i+1)} = p̃^{(i)} − α · ∇f(p̃^{(i)})
where ∇f is approximated via central finite differences.
"""

from __future__ import annotations

import math
from typing import Tuple, List, Optional

from config.config import Config
from core.models.obstacle_model import ObstacleHeightModel


class HoverOptimizer:
    """
    SCA-inspired hover position refinement.

    Parameters
    ----------
    max_iter      : maximum SCA gradient iterations
    step_size     : initial gradient descent step (metres)
    conv_tol      : convergence threshold (metres)
    hover_radius  : max radial displacement from the original node position (metres)
    """

    def __init__(
        self,
        max_iter: int = None,
        step_size: float = None,
        conv_tol: float = None,
        hover_radius: float = 30.0,
    ):
        self.max_iter    = max_iter    or Config.SCA_MAX_ITERATIONS
        self.step_size   = step_size   or Config.SCA_STEP_SIZE
        self.conv_tol    = conv_tol    or Config.SCA_CONVERGENCE_TOL
        self.hover_radius = hover_radius

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimise(
        self,
        node_pos: Tuple[float, float, float],
        prev_pos: Optional[Tuple[float, float, float]],
        next_pos: Optional[Tuple[float, float, float]],
        obstacles: list,
    ) -> Tuple[float, float, float]:
        """
        Refine the hover position above *node_pos* using SCA gradient descent.

        Parameters
        ----------
        node_pos  : (x, y, z) ground position of the target RP node
        prev_pos  : prior waypoint (or UAV current position)
        next_pos  : following waypoint (or None if last node)
        obstacles : list of Obstacle instances (for altitude constraint)

        Returns
        -------
        (x*, y*, z*) : optimised hover position
        """
        # Safe initial altitude
        z_init = ObstacleHeightModel.required_altitude(node_pos[0], node_pos[1], obstacles)
        z_init = max(z_init, Config.UAV_FLIGHT_ALTITUDE)

        px, py, pz = node_pos[0], node_pos[1], z_init
        alpha = self.step_size
        eps = 1.0   # finite difference step

        for _ in range(self.max_iter):
            f0 = self._cost(px, py, pz, node_pos, prev_pos, next_pos)

            # Central finite differences on x and y
            grad_x = (self._cost(px + eps, py, pz, node_pos, prev_pos, next_pos) -
                      self._cost(px - eps, py, pz, node_pos, prev_pos, next_pos)) / (2 * eps)
            grad_y = (self._cost(px, py + eps, pz, node_pos, prev_pos, next_pos) -
                      self._cost(px, py - eps, pz, node_pos, prev_pos, next_pos)) / (2 * eps)

            nx = px - alpha * grad_x
            ny = py - alpha * grad_y

            # Clamp to hover radius around original node
            dr = math.hypot(nx - node_pos[0], ny - node_pos[1])
            if dr > self.hover_radius:
                scale = self.hover_radius / (dr + 1e-8)
                nx = node_pos[0] + (nx - node_pos[0]) * scale
                ny = node_pos[1] + (ny - node_pos[1]) * scale

            # Enforce altitude constraint
            nz = ObstacleHeightModel.enforce_altitude(nx, ny, pz, obstacles)

            # Convergence check
            step_taken = math.sqrt((nx - px) ** 2 + (ny - py) ** 2)
            px, py, pz = nx, ny, nz

            if step_taken < self.conv_tol:
                break

            # Backtracking step-size reduction if cost increased
            if self._cost(px, py, pz, node_pos, prev_pos, next_pos) > f0 + 1e-6:
                alpha *= 0.5

        return (round(px, 4), round(py, 4), round(pz, 4))

    # ------------------------------------------------------------------
    # Cost function
    # ------------------------------------------------------------------

    def _cost(
        self,
        x: float, y: float, z: float,
        node_pos: Tuple,
        prev_pos: Optional[Tuple],
        next_pos: Optional[Tuple],
    ) -> float:
        """
        Travel cost objective:
            f = dist(prev, hover) + dist(hover, next)
        """
        total = 0.0
        if prev_pos is not None:
            total += math.sqrt((x - prev_pos[0])**2 + (y - prev_pos[1])**2 + (z - prev_pos[2])**2)
        if next_pos is not None:
            total += math.sqrt((x - next_pos[0])**2 + (y - next_pos[1])**2 + (z - next_pos[2])**2)
        return total

    # ------------------------------------------------------------------
    # Batch: optimise hover positions for the full visiting sequence
    # ------------------------------------------------------------------

    def optimise_sequence(
        self,
        uav_start: Tuple,
        nodes: list,
        obstacles: list,
    ) -> List[Tuple[float, float, float]]:
        """
        Optimise hover positions for each node in the visiting sequence.

        Returns a list of refined (x, y, z) hover positions, one per node.
        """
        hover_positions = []
        prev = uav_start

        for i, node in enumerate(nodes):
            node_pos = node.position()
            next_pos = nodes[i + 1].position() if i + 1 < len(nodes) else None

            h = self.optimise(node_pos, prev, next_pos, obstacles)
            hover_positions.append(h)
            prev = h

        return hover_positions

    # ------------------------------------------------------------------
    # Convenience static wrapper
    # ------------------------------------------------------------------

    @staticmethod
    def apply_sequence(uav_start: Tuple, nodes: list, obstacles: list) -> List[Tuple]:
        """Single-call convenience: returns refined hover positions for all nodes."""
        opt = HoverOptimizer()
        return opt.optimise_sequence(uav_start, nodes, obstacles)
