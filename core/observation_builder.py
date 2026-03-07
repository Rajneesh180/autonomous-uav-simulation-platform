"""
Observation Builder — State Vector Construction for RL Agents
=============================================================
Extracted from MissionController (Stage 5 — Decomposition).

Wraps AgentCentricTransform to build the D3QN observation vector
with appended K nearest-obstacle features, normalised to [-1, 1].
"""

from __future__ import annotations

import math
from typing import List

from config.config import Config
from core.sensing.agent_centric_transform import AgentCentricTransform


class ObservationBuilder:
    """Constructs normalised observation vectors for RL training."""

    @staticmethod
    def build(
        uav,
        target,
        current_step: int,
        obstacles: list,
        k_nearest: int = 3,
    ) -> List[float]:
        """
        Build state vector: base agent-centric features + K nearest obstacle distances.

        Returns a list of floats suitable for neural-network input.
        """
        base = AgentCentricTransform.build_state_vector(uav, target, current_step)

        uav_pos = uav.position()
        d_norm = math.sqrt(Config.MAP_WIDTH**2 + Config.MAP_HEIGHT**2)

        # Compute normalised distances to all obstacle centres
        obs_dists: List[float] = []
        for obs in obstacles:
            cx = (obs.x1 + obs.x2) / 2
            cy = (obs.y1 + obs.y2) / 2
            dist = math.hypot(uav_pos[0] - cx, uav_pos[1] - cy)
            obs_dists.append(dist / d_norm)

        obs_dists.sort()
        k_features = obs_dists[:k_nearest]

        # Pad with 1.0 (max normalised distance) if fewer obstacles exist
        while len(k_features) < k_nearest:
            k_features.append(1.0)

        return base + k_features
