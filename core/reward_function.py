"""
Reward Function — Modular RL Reward Computation
================================================
Extracted from MissionController (Stage 5 — Decomposition).

Provides a configurable weighted reward for RL training:
    r = α₁·r_data + α₂·r_energy + α₃·r_aoi + α₄·r_collision + α₅·r_deadline
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardWeights:
    """Tunable reward component weights."""
    data: float = 1.0
    energy: float = 0.5
    aoi: float = 0.3
    collision: float = 5.0
    deadline: float = 0.2


class RewardFunction:
    """Compute decomposed scalar reward for UAV data-collection RL."""

    def __init__(self, weights: RewardWeights | None = None):
        self.weights = weights or RewardWeights()

    def compute(
        self,
        data_collected: float,
        max_data: float,
        energy_consumed: float,
        energy_budget: float,
        aoi_sum: float,
        num_nodes: int,
        max_steps: int,
        collision: bool,
        deadline_violations: int,
    ) -> float:
        """
        Compute weighted reward from individual components.

        Returns a single scalar reward value.
        """
        w = self.weights

        r_data = data_collected / max(max_data, 1e-6)
        r_energy = -energy_consumed / max(energy_budget, 1e-6)
        r_aoi = -aoi_sum / max(num_nodes * max_steps, 1)
        r_collision = -1.0 if collision else 0.0
        r_deadline = -deadline_violations / max(num_nodes, 1)

        return (
            w.data * r_data
            + w.energy * r_energy
            + w.aoi * r_aoi
            + w.collision * r_collision
            + w.deadline * r_deadline
        )
