"""
Base Station Uplink Transmission Model
=======================================
Implements the Data Transmission (DT) sub-phase and AoI data-age constraint
from: Zheng & Liu, "3D UAV Trajectory Planning With Obstacle Avoidance for
Time-Constrained Data Collection", IEEE TVT, January 2025 — Section III-B,
Equations 11 and 24-25.

In the Successive-Hover-Fly (SHF) model, after collecting data at a DCA node
the UAV must offload to the Base Station before the data-age limit expires:
    T_data = t_collect + t_fly + t_uplink  ≤  T_data_limit

Key additions:
  - Shannon uplink rate to BS: R_BS(hover_pos)
  - Per-step uplink urgency check: must_uplink_now()
  - Uplink execution: clear payload buffer on MissionController
"""

from __future__ import annotations

import math
from config.config import Config


class BaseStationUplinkModel:
    """
    Manages UAV ↔ Base Station uplink dynamics including:
      1. Computing achievable Shannon uplink rate at any UAV hover position.
      2. Evaluating data-age urgency to trigger early BS return.
      3. Simulating payload offload (clearing collected_data_mbits).

    Aligned with: Zheng & Liu (IEEE TVT 2025) — Section III-B, Eq. 11 & 24–25.
    """

    # ------------------------------------------------------------------
    # Uplink Rate Model
    # ------------------------------------------------------------------

    @staticmethod
    def uplink_rate_mbps(uav_pos: tuple, base_pos: tuple) -> float:
        """
        Achievable Shannon uplink rate from UAV to Base Station:

            R_BS = B · log₂(1 + γ₀ / d_3d^α)    [Mbps]

        where d_3d is the 3D distance from UAV to BS.

        Parameters
        ----------
        uav_pos  : (x, y, z) UAV position
        base_pos : (x, y) or (x, y, z) Base Station position

        Returns
        -------
        float : uplink rate in Mbps (≥ 0)
        """
        bx = base_pos[0]
        by = base_pos[1]
        bz = base_pos[2] if len(base_pos) > 2 else Config.BS_HEIGHT_M

        dx = uav_pos[0] - bx
        dy = uav_pos[1] - by
        dz = uav_pos[2] - bz if len(uav_pos) > 2 else Config.UAV_FLIGHT_ALTITUDE - bz

        d_3d = math.sqrt(dx**2 + dy**2 + dz**2)
        if d_3d < 1e-3:
            d_3d = 1e-3

        # Convert γ₀ from dB to linear
        gamma0_linear = 10 ** (Config.BS_GAMMA_0_DB / 10.0)
        snr = gamma0_linear / (d_3d ** Config.BS_PATH_LOSS_EXP)

        rate_bps = Config.BANDWIDTH * math.log2(1.0 + snr)
        return rate_bps / 1e6   # Mbps

    # ------------------------------------------------------------------
    # Data-Age Urgency Check
    # ------------------------------------------------------------------

    @staticmethod
    def uplink_time(payload_mbits: float, uav_pos: tuple, base_pos: tuple) -> float:
        """
        Estimated time to offload payload_mbits to the BS from uav_pos (seconds).
        Returns float('inf') if the BS is unreachable.
        """
        rate = BaseStationUplinkModel.uplink_rate_mbps(uav_pos, base_pos)
        if rate <= 0.0:
            return float("inf")
        return payload_mbits / rate

    @staticmethod
    def must_uplink_now(uav_pos: tuple, base_pos: tuple,
                        payload_mbits: float, current_step: int,
                        last_uplink_step: int,
                        uav_speed: float = None) -> bool:
        """
        Returns True if the UAV must return to the BS immediately to avoid
        exceeding the data-age limit T_data_limit (Eq. 25).

        Evaluates:
            T_remaining_until_limit  =  T_data_limit - (current_step - last_uplink_step)
            T_needed_to_uplink       =  t_fly_to_bs + t_uplink

        If T_needed >= T_remaining: trigger early return.

        Parameters
        ----------
        payload_mbits   : data currently held by UAV (Mbits)
        current_step    : simulation time step counter
        last_uplink_step: step at which last successful uplink completed
        uav_speed       : UAV speed m/step (defaults to Config.UAV_STEP_SIZE)
        """
        if uav_speed is None:
            uav_speed = Config.UAV_STEP_SIZE
        if uav_speed <= 0:
            return False

        # Time since last offload
        age_steps = current_step - last_uplink_step
        remaining_budget = Config.BS_DATA_AGE_LIMIT - age_steps

        if remaining_budget <= 0:
            return True   # Already exceeded — must uplink immediately

        # Estimate fly-time to BS
        bx = base_pos[0]
        by = base_pos[1]
        dist_to_bs = math.hypot(uav_pos[0] - bx, uav_pos[1] - by)
        t_fly_steps = dist_to_bs / max(uav_speed, 1e-3)

        # Estimate uplink time in simulation steps
        t_uplink_s = BaseStationUplinkModel.uplink_time(payload_mbits, uav_pos, base_pos)
        t_uplink_steps = t_uplink_s / float(Config.TIME_STEP)

        total_needed = t_fly_steps + t_uplink_steps
        return total_needed >= remaining_budget

    # ------------------------------------------------------------------
    # Payload Offload
    # ------------------------------------------------------------------

    @staticmethod
    def execute_uplink(mission, current_step: int) -> float:
        """
        Simulate payload offload at the Base Station.
        Clears mission.collected_data_mbits and records the uplink event.

        Returns
        -------
        float : data offloaded in Mbits
        """
        offloaded = mission.collected_data_mbits
        mission.collected_data_mbits = 0.0
        mission.last_uplink_step = current_step
        mission.total_uplinked_mbits = getattr(mission, "total_uplinked_mbits", 0.0) + offloaded

        if offloaded > 0:
            print(
                f"[BS Uplink] Step {current_step}: offloaded {offloaded:.2f} Mbits → "
                f"total uplinked = {mission.total_uplinked_mbits:.2f} Mbits"
            )
        return offloaded
