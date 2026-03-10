# energy estimation for rotary-wing UAV (Eq. 3-5 in report)
# E_total = E_flight + E_hover

from __future__ import annotations

from config.settings import (
    BATTERY_CAPACITY,
    E_FLY_PER_M,
    P_HOVER,
    HOVER_TIME,
    CRUISE_SPEED,
)


def estimate_energy(
    total_dist: float,
    n_rps: int,
    hover_time: float = HOVER_TIME,
    speed: float = CRUISE_SPEED,
) -> tuple[float, float]:
    # returns (total energy in joules, % of battery used)
    e_flight: float = total_dist * E_FLY_PER_M
    e_hover: float = n_rps * hover_time * P_HOVER
    e_total: float = e_flight + e_hover
    pct_used: float = (e_total / BATTERY_CAPACITY) * 100.0
    return e_total, pct_used
