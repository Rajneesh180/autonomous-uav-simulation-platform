"""
Per-Step Telemetry Logger
=========================
Records UAV state and mission metrics at every simulation step into a CSV file
for post-run analysis, reproducibility, and IEEE-paper figure generation.

Output files (saved to telemetry/ subdirectory of the run):
  - step_telemetry.csv  : per-step UAV state vector
  - node_state.csv      : final state of all IoT nodes at mission end
"""

from __future__ import annotations

import csv
import os
import math
from config.config import Config


class TelemetryLogger:
    """
    Lightweight CSV writer that appends one row per simulation step.
    Initialised once at mission start; flushed on close().
    """

    STEP_COLUMNS = [
        "step",
        "uav_x",
        "uav_y",
        "uav_z",
        "battery_J",
        "speed_mps",
        "nodes_visited",
        "current_target_id",
        "collected_mbits",
        "uplinked_mbits",
        "aoi_avg",
        "replan_count",
        "collision_count",
    ]

    NODE_COLUMNS = [
        "node_id",
        "x",
        "y",
        "z",
        "buffer_capacity_mb",
        "residual_buffer_mb",
        "aoi_timer_s",
        "max_aoi_s",
        "node_battery_J",
        "initial_battery_J",
        "visited",
    ]

    def __init__(self, telemetry_dir: str):
        self.telemetry_dir = telemetry_dir
        os.makedirs(telemetry_dir, exist_ok=True)

        self._step_path = os.path.join(telemetry_dir, "step_telemetry.csv")
        self._node_path = os.path.join(telemetry_dir, "node_state.csv")

        # Open step CSV and write header
        self._step_file = open(self._step_path, "w", newline="")
        self._step_writer = csv.writer(self._step_file)
        self._step_writer.writerow(self.STEP_COLUMNS)

        self._prev_pos = None  # for speed computation

    # ------------------------------------------------------------------
    # Per-step recording
    # ------------------------------------------------------------------

    def log_step(self, step: int, mission) -> None:
        """Append one row of telemetry from the current mission state."""
        uav = mission.uav
        x, y, z = uav.x, uav.y, getattr(uav, "z", Config.UAV_FLIGHT_ALTITUDE)

        # Compute instantaneous speed
        if self._prev_pos is not None:
            dx = x - self._prev_pos[0]
            dy = y - self._prev_pos[1]
            dz = z - self._prev_pos[2]
            speed = math.sqrt(dx**2 + dy**2 + dz**2) / max(float(Config.TIME_STEP), 1e-3)
        else:
            speed = 0.0
        self._prev_pos = (x, y, z)

        # Average AoI across all non-UAV nodes
        ground_nodes = mission.env.nodes[1:]
        aoi_avg = 0.0
        if ground_nodes:
            aoi_avg = sum(n.aoi_timer for n in ground_nodes) / len(ground_nodes)

        target_id = mission.current_target.id if mission.current_target else -1

        self._step_writer.writerow([
            step,
            round(x, 2),
            round(y, 2),
            round(z, 2),
            round(uav.current_battery, 2),
            round(speed, 4),
            len(mission.visited),
            target_id,
            round(mission.collected_data_mbits, 4),
            round(getattr(mission, "total_uplinked_mbits", 0.0), 4),
            round(aoi_avg, 4),
            getattr(mission, "replan_count", 0),
            getattr(mission, "collision_count", 0),
        ])

    # ------------------------------------------------------------------
    # End-of-mission: node state snapshot
    # ------------------------------------------------------------------

    def save_node_state(self, mission) -> None:
        """Write final state of all IoT nodes to node_state.csv."""
        with open(self._node_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.NODE_COLUMNS)

            for node in mission.env.nodes[1:]:
                writer.writerow([
                    node.id,
                    round(node.x, 2),
                    round(node.y, 2),
                    round(getattr(node, "z", 0.0), 2),
                    round(node.buffer_capacity, 4),
                    round(node.current_buffer, 4),
                    round(node.aoi_timer, 4),
                    round(getattr(node, "max_aoi_timer", 0.0), 4),
                    round(getattr(node, "node_battery_J", 0.0), 6),
                    round(getattr(node, "initial_node_battery_J", Config.NODE_BATTERY_J), 6),
                    1 if node.id in mission.visited else 0,
                ])

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the step CSV file handle."""
        if self._step_file and not self._step_file.closed:
            self._step_file.flush()
            self._step_file.close()
