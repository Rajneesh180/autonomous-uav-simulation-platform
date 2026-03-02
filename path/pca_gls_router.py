import math
from typing import Dict, List, Tuple
from core.models.node_model import Node
from config.config import Config


class PCAGLSRouter:
    """
    Path Cheapest Arc with Guided Local Search (PCA-GLS) Optimizer.

    Implements the full GLS penalty augmentation per Donipati et al.
    (IEEE TNSM 2025):
      1. PCA greedy initialization (distance + deadline urgency + buffer weight)
      2. GLS iterative refinement:
         a) 2-opt on augmented costs  c'(i,j) = d(i,j) + λ·p[i][j]
         b) Penalise the edge with maximum utility  u(i,j) = d(i,j)/(1+p[i][j])
         c) Repeat until max_iters or patience exhausted
    """

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_distance(p1, p2) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def _route_distance(route: List[Node]) -> float:
        """Total unaugmented Euclidean distance of a route."""
        d = 0.0
        for i in range(len(route) - 1):
            d += PCAGLSRouter._compute_distance(
                route[i].position(), route[i + 1].position()
            )
        return d

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @staticmethod
    def optimize(uav_pos: Tuple[float, float, float], nodes: List[Node]) -> List[Node]:
        """
        Executes the PCA-GLS heuristic.

        1. Build greedy PCA sequence (distance + time-window urgency + buffer)
        2. Refine via full GLS penalty augmentation
        """
        if not nodes:
            return []

        # --- 1. PCA greedy initialization ---
        route = PCAGLSRouter._pca_init(uav_pos, nodes)

        # --- 2. GLS penalty refinement ---
        if len(route) >= 3:
            route = PCAGLSRouter._gls_refine(route)

        return route

    # ------------------------------------------------------------------
    # PCA initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _pca_init(
        uav_pos: Tuple[float, float, float], nodes: List[Node]
    ) -> List[Node]:
        """Greedy PCA construction guided by distance, deadline urgency, and buffer."""
        unvisited = list(nodes)
        route: List[Node] = []
        current_pos = uav_pos
        current_time = 0.0
        dist = 0.0  # will be overwritten; declared for scope

        while unvisited:
            best_node = None
            best_cost = float("inf")

            for node in unvisited:
                dist = PCAGLSRouter._compute_distance(current_pos, node.position())

                # Deadline urgency
                if math.isinf(node.time_window_end):
                    urgency = 0.0
                else:
                    margin = node.time_window_end - (current_time + dist)
                    urgency = 100.0 / (margin + 1.0) if margin >= 0 else 1000.0

                buffer_weight = node.current_buffer / (node.buffer_capacity + 1e-6)

                cost = dist + (10.0 * urgency) - (500.0 * buffer_weight)

                if cost < best_cost:
                    best_cost = cost
                    best_node = node

            if best_node is None:
                best_node = unvisited[0]

            route.append(best_node)
            unvisited.remove(best_node)
            current_pos = best_node.position()
            current_time += dist

        return route

    # ------------------------------------------------------------------
    # GLS penalty refinement (Donipati et al.)
    # ------------------------------------------------------------------

    @staticmethod
    def _gls_refine(route: List[Node]) -> List[Node]:
        """
        Full Guided Local Search with penalty augmentation.

        Maintains a penalty matrix p[i][j] (keyed by node-ID pairs).
        At each iteration:
          1. Run 2-opt on augmented costs until locally optimal
          2. Find the route edge with maximum utility u = d / (1 + p)
          3. Increment the penalty for that edge
        Terminates after *max_iters* or when real distance hasn't improved
        for *patience* consecutive iterations.
        """
        lam = Config.GLS_LAMBDA
        max_iters = Config.GLS_MAX_ITERATIONS
        patience = Config.GLS_PATIENCE

        # Penalty matrix: (node_id_a, node_id_b) → int  (symmetric)
        penalties: Dict[Tuple[int, int], int] = {}

        def _penalty_key(na: Node, nb: Node) -> Tuple[int, int]:
            return (min(na.id, nb.id), max(na.id, nb.id))

        def _get_penalty(na: Node, nb: Node) -> int:
            return penalties.get(_penalty_key(na, nb), 0)

        def _augmented_cost(na: Node, nb: Node) -> float:
            d = PCAGLSRouter._compute_distance(na.position(), nb.position())
            p = _get_penalty(na, nb)
            return d + lam * p

        # --- Initial 2-opt on raw distance ---
        best_route = PCAGLSRouter._two_opt(route)
        best_real_dist = PCAGLSRouter._route_distance(best_route)
        stale = 0

        for _ in range(max_iters):
            # 1. 2-opt on augmented costs
            candidate = PCAGLSRouter._two_opt_augmented(best_route, _augmented_cost)
            cand_real_dist = PCAGLSRouter._route_distance(candidate)

            if cand_real_dist < best_real_dist - 1e-9:
                best_route = candidate
                best_real_dist = cand_real_dist
                stale = 0
            else:
                stale += 1

            if stale >= patience:
                break

            # 2. Penalise the edge with maximum utility
            max_util = -1.0
            max_edge: Tuple[int, int] | None = None
            for i in range(len(best_route) - 1):
                na, nb = best_route[i], best_route[i + 1]
                d = PCAGLSRouter._compute_distance(na.position(), nb.position())
                p = _get_penalty(na, nb)
                util = d / (1.0 + p)
                if util > max_util:
                    max_util = util
                    max_edge = _penalty_key(na, nb)

            if max_edge is not None:
                penalties[max_edge] = penalties.get(max_edge, 0) + 1

        return best_route

    # ------------------------------------------------------------------
    # 2-opt variants
    # ------------------------------------------------------------------

    @staticmethod
    def _two_opt(route: List[Node]) -> List[Node]:
        """Standard 2-opt improvement on raw Euclidean distance."""
        best = list(route)
        best_dist = PCAGLSRouter._route_distance(best)
        improved = True

        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i + 2, len(best)):
                    new = best[:]
                    new[i:j] = reversed(best[i:j])
                    nd = PCAGLSRouter._route_distance(new)
                    if nd < best_dist - 1e-12:
                        best, best_dist = new, nd
                        improved = True
                        break
                if improved:
                    break
        return best

    @staticmethod
    def _two_opt_augmented(route: List[Node], cost_fn) -> List[Node]:
        """2-opt using a caller-supplied edge-cost function (for GLS penalties)."""

        def _total(r):
            return sum(cost_fn(r[k], r[k + 1]) for k in range(len(r) - 1))

        best = list(route)
        best_cost = _total(best)
        improved = True

        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i + 2, len(best)):
                    new = best[:]
                    new[i:j] = reversed(best[i:j])
                    nc = _total(new)
                    if nc < best_cost - 1e-12:
                        best, best_cost = new, nc
                        improved = True
                        break
                if improved:
                    break
        return best
