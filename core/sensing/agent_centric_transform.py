"""
Agent-Centric Coordinate Transform
===================================
Implements the UAV body-frame state representation from:
    Wang et al. "Learning-Based UAV Path Planning for Data Collection With
    Integrated Collision Avoidance", IEEE IoT, 2022 — Section III-B, Table I.

Converts world-frame observations into the UAV's local body frame, giving a
translation-invariant and rotation-invariant state vector suitable for RL
(D3QN) generalisation across missions.

Transform:
    p_ac = R(ψ) · (p_world − p_uav)
where R(ψ) is the 2D rotation matrix by heading angle ψ.
"""

from __future__ import annotations

import math
from typing import Tuple, List, Optional

from config.config import Config


class AgentCentricTransform:
    """
    Agent-Centric (body-frame) Coordinate Transformer.

    Converts world-frame positions into the UAV's local coordinate frame,
    normalises features to [0, 1], and assembles the D3QN state vector.

    Aligned with: Wang et al. (IEEE IoT 2022) — Section III-B, Eq. 8–10, Table I.
    """

    # ---------------------------------------------------------------
    # Core geometric transform
    # ---------------------------------------------------------------

    @staticmethod
    def world_to_agent(
        uav_pos: Tuple[float, float, float],
        uav_yaw: float,
        target_pos: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """
        Transform a world-frame point into the UAV body frame.

            Δp = p_target − p_uav
            p_ac_x = Δx·cos(ψ) + Δy·sin(ψ)
            p_ac_y = −Δx·sin(ψ) + Δy·cos(ψ)
            p_ac_z = Δz  (no roll/pitch in 2.5D model)

        Parameters
        ----------
        uav_pos    : (x, y, z) UAV world position
        uav_yaw    : UAV heading angle ψ (radians)
        target_pos : (x, y, z) target world position

        Returns
        -------
        (p_ac_x, p_ac_y, p_ac_z) : body-frame coordinates
        """
        dx = target_pos[0] - uav_pos[0]
        dy = target_pos[1] - uav_pos[1]
        dz = (target_pos[2] - uav_pos[2]) if len(target_pos) > 2 else 0.0

        cos_psi = math.cos(uav_yaw)
        sin_psi = math.sin(uav_yaw)

        p_ac_x =  dx * cos_psi + dy * sin_psi
        p_ac_y = -dx * sin_psi + dy * cos_psi
        p_ac_z = dz

        return (p_ac_x, p_ac_y, p_ac_z)

    # ---------------------------------------------------------------
    # Normalised state vector (D3QN observation)
    # ---------------------------------------------------------------

    @staticmethod
    def build_state_vector(
        uav,
        target,
        current_step: int,
        map_width: float = None,
        map_height: float = None,
        max_steps: int = None,
    ) -> List[float]:
        """
        Constructs the normalised state vector s_t for the D3QN agent.

        Features (Wang et al., Table I):
          [0] agent-centric target x  (p_ac_x / d_norm)
          [1] agent-centric target y  (p_ac_y / d_norm)
          [2] agent-centric altitude Δz / H_max
          [3] node residual buffer fraction (buffer / capacity)
          [4] node priority fraction (priority / P_max)
          [5] UAV residual battery fraction
          [6] elapsed time fraction (t / T_max)

        Returns a length-7 list of floats in [−1, 1] or [0, 1].
        """
        if map_width is None:
            map_width = Config.MAP_WIDTH
        if map_height is None:
            map_height = Config.MAP_HEIGHT
        if max_steps is None:
            max_steps = Config.MAX_TIME_STEPS

        d_norm = math.sqrt(map_width ** 2 + map_height ** 2)
        h_norm = max(Config.UAV_FLIGHT_ALTITUDE, 1.0)
        p_max = 10.0  # assumed max priority

        uav_pos = uav.position()
        tgt_pos = target.position()

        p_ac_x, p_ac_y, p_ac_z = AgentCentricTransform.world_to_agent(
            uav_pos, uav.yaw, tgt_pos
        )

        buf_frac = (target.current_buffer / target.buffer_capacity
                    if target.buffer_capacity > 0 else 0.0)
        pri_frac = target.priority / p_max
        bat_frac = (uav.current_battery / uav.battery_capacity
                    if uav.battery_capacity > 0 else 0.0)
        time_frac = current_step / max(max_steps, 1)

        return [
            p_ac_x / (d_norm + 1e-8),
            p_ac_y / (d_norm + 1e-8),
            p_ac_z / (h_norm + 1e-8),
            min(1.0, max(0.0, buf_frac)),
            min(1.0, max(0.0, pri_frac)),
            min(1.0, max(0.0, bat_frac)),
            min(1.0, max(0.0, time_frac)),
        ]

    # ---------------------------------------------------------------
    # Convenience: batch transform for multiple targets
    # ---------------------------------------------------------------

    @staticmethod
    def batch_state_vectors(uav, targets: list, current_step: int) -> List[List[float]]:
        """
        Build state vectors for all candidate target nodes simultaneously.
        Returns a list of length-7 state vectors, one per target.
        """
        return [
            AgentCentricTransform.build_state_vector(uav, t, current_step)
            for t in targets
        ]

    # ---------------------------------------------------------------
    # Inverse transform (body frame → world frame, for action decoding)
    # ---------------------------------------------------------------

    @staticmethod
    def agent_to_world(
        uav_pos: Tuple[float, float, float],
        uav_yaw: float,
        p_ac: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """
        Inverse rotation: body frame → world frame.
        Used to decode RL actions from body-frame offsets to absolute positions.
        """
        cos_psi = math.cos(uav_yaw)
        sin_psi = math.sin(uav_yaw)

        dx = p_ac[0] * cos_psi - p_ac[1] * sin_psi
        dy = p_ac[0] * sin_psi + p_ac[1] * cos_psi
        dz = p_ac[2] if len(p_ac) > 2 else 0.0

        return (
            uav_pos[0] + dx,
            uav_pos[1] + dy,
            uav_pos[2] + dz,
        )
