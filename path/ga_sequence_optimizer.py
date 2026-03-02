"""
Genetic Algorithm Visiting Sequence Optimizer
===============================================
Implements the GA-based visiting order optimisation from:
    Zheng & Liu, "3D UAV Trajectory Planning With Obstacle Avoidance for
    Time-Constrained Data Collection", IEEE TVT, January 2025 — Section III-D,
    Algorithm 2.

After Rendezvous Point (RP) selection compresses the visit set R,
this module refines the visiting order π* by minimising total path length
subject to time-window feasibility constraints [e_j, l_j].

GA operators used:
  - Tournament selection (k=3)
  - Order Crossover (OX)
  - Swap mutation
  - Elitist replacement
  - Time-window feasibility repair
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

from config.config import Config
from core.models.node_model import Node


class GASequenceOptimizer:
    """
    Genetic Algorithm for time-window-constrained UAV visiting sequence.

    Parameters
    ----------
    pop_size     : population size (chromosomes per generation)
    max_gen      : maximum number of generations
    p_crossover  : probability of applying OX crossover
    p_mutate     : probability of applying swap mutation
    tw_penalty   : weight for time-window violation penalty
    """

    def __init__(
        self,
        pop_size: int = None,
        max_gen: int = None,
        p_crossover: float = None,
        p_mutate: float = None,
        tw_penalty: float = None,
    ):
        self.pop_size    = pop_size    or Config.GA_POPULATION_SIZE
        self.max_gen     = max_gen     or Config.GA_MAX_GENERATIONS
        self.p_crossover = p_crossover or Config.GA_CROSSOVER_RATE
        self.p_mutate    = p_mutate    or Config.GA_MUTATION_RATE
        self.tw_penalty  = tw_penalty  or Config.GA_TW_PENALTY_WEIGHT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimise(self, start_pos: Tuple, nodes: List[Node], seed_order: List[Node] = None) -> List[Node]:
        """
        Find the near-optimal visiting order for *nodes* starting from *start_pos*.

        Parameters
        ----------
        start_pos  : UAV current (x, y, z) position
        nodes      : list of Node objects to visit (already RP-filtered)
        seed_order : optional PCA-GLS pre-computed order used to seed the GA

        Returns
        -------
        list of Node in optimised visiting order
        """
        n = len(nodes)
        if n <= 2:
            return nodes[:]

        # ---- Initialise population as index permutations ----
        population = self._init_population(n, seed_order, nodes)

        # ---- Evaluate fitness ----
        fitness_scores = [self._fitness(chrom, start_pos, nodes) for chrom in population]

        best_chrom = max(population, key=lambda c: self._fitness(c, start_pos, nodes))

        for _ in range(self.max_gen):
            new_pop = [best_chrom[:]]  # elitism: carry best forward

            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(population, fitness_scores)
                p2 = self._tournament_select(population, fitness_scores)

                if random.random() < self.p_crossover:
                    child = self._ox_crossover(p1, p2)
                else:
                    child = p1[:]

                if random.random() < self.p_mutate:
                    child = self._swap_mutate(child)

                # Feasibility repair: ensure no duplicate indices
                child = self._repair(child, n)
                new_pop.append(child)

            population = new_pop
            fitness_scores = [self._fitness(c, start_pos, nodes) for c in population]
            best_chrom = population[fitness_scores.index(max(fitness_scores))]

        return [nodes[i] for i in best_chrom]

    # ------------------------------------------------------------------
    # Fitness function
    # ------------------------------------------------------------------

    def _fitness(self, chrom: List[int], start_pos: Tuple, nodes: List[Node]) -> float:
        """
        f(π) = 1 / (1 + path_cost(π) + tw_penalty * violation(π))
        """
        total_dist = 0.0
        violation  = 0.0
        prev = start_pos
        t = 0.0

        for idx in chrom:
            node = nodes[idx]
            npos = node.position()
            d = math.sqrt(sum((a-b)**2 for a, b in zip(prev[:2], npos[:2])))
            total_dist += d
            # Approximate travel time (steps) at UAV_STEP_SIZE
            t += d / max(Config.UAV_STEP_SIZE, 1e-3)

            # Time-window violation
            if t < node.time_window_start:
                violation += node.time_window_start - t
            if t > node.time_window_end:
                violation += t - node.time_window_end

            prev = npos

        return 1.0 / (1.0 + total_dist + self.tw_penalty * violation)

    # ------------------------------------------------------------------
    # GA operators
    # ------------------------------------------------------------------

    def _init_population(self, n: int, seed_order: List[Node], nodes: List[Node]) -> List[List[int]]:
        """Generate initial population; optionally seed one individual from PCA-GLS."""
        pop = []
        if seed_order is not None:
            node_id_to_idx = {node.id: i for i, node in enumerate(nodes)}
            seed_chrom = [node_id_to_idx[n.id] for n in seed_order if n.id in node_id_to_idx]
            if len(seed_chrom) == n:
                pop.append(seed_chrom)

        while len(pop) < self.pop_size:
            chrom = list(range(n))
            random.shuffle(chrom)
            pop.append(chrom)
        return pop

    def _tournament_select(self, population: List[List[int]], scores: List[float], k: int = 3) -> List[int]:
        """K-tournament selection — pick the fittest among k random chromosomes."""
        contestants = random.choices(range(len(population)), k=k)
        winner = max(contestants, key=lambda i: scores[i])
        return population[winner][:]

    def _ox_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        """Order Crossover (OX): preserves relative order of a subsequence from p1."""
        n = len(p1)
        a, b = sorted(random.sample(range(n), 2))
        child = [-1] * n
        child[a:b+1] = p1[a:b+1]
        present = set(child[a:b+1])
        fill = [x for x in p2 if x not in present]
        ptr = 0
        for i in list(range(b+1, n)) + list(range(a)):
            child[i] = fill[ptr]
            ptr += 1
        return child

    def _swap_mutate(self, chrom: List[int]) -> List[int]:
        """Swap two random genes."""
        c = chrom[:]
        i, j = random.sample(range(len(c)), 2)
        c[i], c[j] = c[j], c[i]
        return c

    def _repair(self, chrom: List[int], n: int) -> List[int]:
        """Ensure chrom is a valid permutation of 0..n-1."""
        seen = set()
        missing = list(set(range(n)) - set(chrom))
        result = []
        for x in chrom:
            if x not in seen and 0 <= x < n:
                seen.add(x)
                result.append(x)
            else:
                if missing:
                    result.append(missing.pop())
        return result

    # ------------------------------------------------------------------
    # Convenience static wrapper
    # ------------------------------------------------------------------

    @staticmethod
    def apply(start_pos: Tuple, nodes: List[Node], seed_order: List[Node] = None) -> List[Node]:
        """Single-call convenience: returns GA-optimised visiting order."""
        ga = GASequenceOptimizer()
        return ga.optimise(start_pos, nodes, seed_order=seed_order)
