"""
Physics Engine — Motion Primitive Selection & Kinematic Application
=====================================================================
Extracted from MissionController (Stage 5 — Decomposition).

Encapsulates all low-level UAV movement logic:
  • 3D motion primitive generation (spherical coordinate offsets)
  • Multi-objective scoring (alignment, obstacle proximity, risk, energy)
  • Kinematic constraint enforcement (acceleration, yaw/pitch rate limits)
  • Escape-trap bounce recovery
  • Propulsion + mechanical energy consumption
  • Return-to-base safety check
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

from config.config import Config
from core.models.energy_model import EnergyModel
from core.models.obstacle_model import ObstacleHeightModel


@dataclass
class MovementOutcome:
    """Result of a single physics step."""
    success: bool
    energy_consumed: float = 0.0
    collision: bool = False
    unsafe_return: bool = False
    replan_reason: Optional[str] = None


class PhysicsEngine:
    """Stateless motion-primitive selector and kinematic integrator."""

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    @staticmethod
    def execute_movement(
        uav,
        target_pos: Tuple[float, float, float],
        env,
        digital_twin,
        base_position: Tuple[float, float],
    ) -> MovementOutcome:
        """Generate motion primitives, score, pick best, apply kinematics."""
        current_pos = uav.position()
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        dz = target_pos[2] - current_pos[2]
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        base_pitch = math.atan2(dz, math.hypot(dx, dy))

        dt = float(Config.TIME_STEP)

        # --- Acceleration dynamics ---
        v_current_mag = math.sqrt(uav.vx**2 + uav.vy**2 + uav.vz**2)
        target_v_mag = min(Config.UAV_STEP_SIZE / dt, distance / dt) if dt > 0 else 0.0
        max_dv = Config.MAX_ACCELERATION * dt
        if target_v_mag - v_current_mag > max_dv:
            v_mag = v_current_mag + max_dv
        elif v_current_mag - target_v_mag > max_dv:
            v_mag = max(0.0, v_current_mag - max_dv)
        else:
            v_mag = target_v_mag
        step_size = v_mag * dt

        # --- Yaw / pitch rate limiting ---
        ideal_yaw = math.atan2(dy, dx)
        max_yaw_change = math.radians(Config.MAX_YAW_RATE) * dt
        max_pitch_change = math.radians(Config.MAX_PITCH_RATE) * dt
        base_yaw = _constrain_angle(ideal_yaw, uav.yaw, max_yaw_change)
        c_base_pitch = _constrain_angle(base_pitch, uav.pitch, max_pitch_change)

        # --- ISAC Digital Twin scan ---
        if Config.ENABLE_ISAC_DIGITAL_TWIN:
            digital_twin.scan_environment(uav.position(), env.obstacles)
            routing_obstacles = digital_twin.known_obstacles
        else:
            routing_obstacles = env.obstacles

        # --- Generate & score motion primitives ---
        best_score = float("inf")
        best_move = None

        yaw_offsets = [0] + Config.STEERING_ANGLES
        pitch_offsets = [0, -15, 15]

        for yaw_offset in yaw_offsets:
            for pitch_offset in pitch_offsets:
                yaw = base_yaw + math.radians(yaw_offset)
                pitch = c_base_pitch + math.radians(pitch_offset)

                # Spherical to Cartesian
                new_x = current_pos[0] + step_size * math.cos(pitch) * math.cos(yaw)
                new_y = current_pos[1] + step_size * math.cos(pitch) * math.sin(yaw)
                new_z = current_pos[2] + step_size * math.sin(pitch)

                # Gap 5: 3D Gaussian altitude constraint
                new_z = ObstacleHeightModel.enforce_altitude(
                    new_x, new_y, new_z, env.obstacles
                )

                # Hard collision check (2D footprint)
                if env.point_in_obstacle((new_x, new_y)):
                    continue

                travel_distance = math.sqrt(
                    (new_x - current_pos[0])**2
                    + (new_y - current_pos[1])**2
                    + (new_z - current_pos[2])**2
                )
                risk_mult = env.risk_multiplier((new_x, new_y))
                adjusted_distance = travel_distance * risk_mult

                if not EnergyModel.can_travel(uav, adjusted_distance):
                    continue

                # ---------- SCORING COMPONENTS ----------

                # 1. Alignment score (cosine similarity in 3D)
                to_target = (dx, dy, dz)
                move_vec = (
                    new_x - current_pos[0],
                    new_y - current_pos[1],
                    new_z - current_pos[2],
                )
                dot = sum(a * b for a, b in zip(to_target, move_vec))
                norm_prod = (
                    math.sqrt(sum(v**2 for v in to_target))
                    * math.sqrt(sum(v**2 for v in move_vec))
                    + Config.SCORE_EPS
                )
                alignment_score = 1 - (dot / norm_prod)

                # 2. Obstacle proximity penalty
                obstacle_penalty = 0.0
                for obs in routing_obstacles:
                    clearance = PhysicsEngine._predicted_clearance(new_x, new_y, obs)
                    if clearance < Config.COLLISION_MARGIN:
                        obstacle_penalty += 1000.0
                    else:
                        obstacle_penalty += 1.0 / (clearance + 1e-5)

                # 3. Risk penalty
                risk_penalty = risk_mult - 1.0

                # 4. Energy penalty
                energy_penalty = adjusted_distance

                # ---------- TOTAL SCORE ----------
                total_score = (
                    Config.ALIGNMENT_WEIGHT * alignment_score
                    + Config.OBSTACLE_PENALTY_WEIGHT * obstacle_penalty
                    + Config.RISK_PENALTY_WEIGHT * risk_penalty
                    + Config.ENERGY_PENALTY_WEIGHT * energy_penalty
                )

                if total_score < best_score:
                    best_score = total_score
                    best_move = (
                        new_x, new_y, new_z, adjusted_distance,
                        yaw, pitch, v_mag, step_size,
                    )

        # --- Escape trap: aggressive reverse + scatter bounce ---
        if best_move is None:
            bounce_dist = 15.0
            escape_yaw = uav.yaw + math.pi + random.uniform(-0.5, 0.5)
            uav.x = max(0.0, min(float(env.width), uav.x + bounce_dist * math.cos(escape_yaw)))
            uav.y = max(0.0, min(float(env.height), uav.y + bounce_dist * math.sin(escape_yaw)))
            uav.yaw = escape_yaw
            return MovementOutcome(
                success=False, collision=True, replan_reason="collision_escape"
            )

        (new_x, new_y, new_z, adjusted_distance,
         chosen_yaw, chosen_pitch, v_mag, _step) = best_move

        # --- Kinematics update ---
        new_vx = (new_x - uav.x) / dt if dt > 0 else 0.0
        new_vy = (new_y - uav.y) / dt if dt > 0 else 0.0
        new_vz = (new_z - uav.z) / dt if dt > 0 else 0.0
        ax = (new_vx - uav.vx) / dt if dt > 0 else 0.0
        ay = (new_vy - uav.vy) / dt if dt > 0 else 0.0
        az = (new_vz - uav.vz) / dt if dt > 0 else 0.0

        # --- Energy consumption ---
        mechanical_energy = EnergyModel.mechanical_energy(uav, (ax, ay, az))
        aerodynamic_energy = EnergyModel.energy_for_distance(uav, adjusted_distance)
        predicted_energy = mechanical_energy + aerodynamic_energy
        EnergyModel.consume(uav, predicted_energy)

        # --- Return-to-base safety ---
        if not EnergyModel.can_return_to_base(
            uav, (new_x, new_y), base_position, env.risk_multiplier((new_x, new_y))
        ):
            return MovementOutcome(
                success=False,
                energy_consumed=predicted_energy,
                unsafe_return=True,
                replan_reason="energy_risk",
            )

        # --- Apply position ---
        uav.x = new_x
        uav.y = new_y
        uav.z = new_z
        uav.yaw = chosen_yaw
        uav.pitch = chosen_pitch
        uav.vx = new_vx
        uav.vy = new_vy
        uav.vz = new_vz

        env.uav_trail.append((new_x, new_y, new_z))
        if len(env.uav_trail) > 30:
            env.uav_trail.pop(0)

        return MovementOutcome(success=True, energy_consumed=predicted_energy)

    # ----------------------------------------------------------
    # Obstacle clearance helpers
    # ----------------------------------------------------------

    @staticmethod
    def _rectangle_clearance(x: float, y: float, obs) -> float:
        dx = max(obs.x1 - x, 0, x - obs.x2)
        dy = max(obs.y1 - y, 0, y - obs.y2)
        return math.hypot(dx, dy)

    @staticmethod
    def _predicted_clearance(x: float, y: float, obs) -> float:
        if not Config.ENABLE_MOVING_OBSTACLES:
            return PhysicsEngine._rectangle_clearance(x, y, obs)
        pred_x1 = obs.x1 + obs.vx
        pred_x2 = obs.x2 + obs.vx
        pred_y1 = obs.y1 + obs.vy
        pred_y2 = obs.y2 + obs.vy
        dx = max(pred_x1 - x, 0, x - pred_x2)
        dy = max(pred_y1 - y, 0, y - pred_y2)
        return math.hypot(dx, dy)

    # ----------------------------------------------------------
    # Predictive direction safety (legacy — retained for API)
    # ----------------------------------------------------------

    @staticmethod
    def is_direction_safe(uav, current_pos, angle, step_size, env, digital_twin, base_position):
        """Check whether a straight-line step in *angle* direction is safe.

        .. warning:: Side-effects: consumes energy and updates UAV position
           when the step is safe.  Retained for backward compatibility.
        """
        for i in range(1, Config.PREDICTION_HORIZON + 1):
            sim_x = current_pos[0] + i * step_size * math.cos(angle)
            sim_y = current_pos[1] + i * step_size * math.sin(angle)
            routing_obstacles = (
                digital_twin.known_obstacles
                if Config.ENABLE_ISAC_DIGITAL_TWIN
                else env.obstacles
            )
            for obs in routing_obstacles:
                future_x1 = obs.x1 + i * obs.vx * Config.OBSTACLE_VELOCITY_SCALE
                future_y1 = obs.y1 + i * obs.vy * Config.OBSTACLE_VELOCITY_SCALE
                future_x2 = obs.x2 + i * obs.vx * Config.OBSTACLE_VELOCITY_SCALE
                future_y2 = obs.y2 + i * obs.vy * Config.OBSTACLE_VELOCITY_SCALE
                if future_x1 <= sim_x <= future_x2 and future_y1 <= sim_y <= future_y2:
                    return False

        new_x = current_pos[0] + step_size * math.cos(angle)
        new_y = current_pos[1] + step_size * math.sin(angle)
        travel_distance = math.hypot(new_x - current_pos[0], new_y - current_pos[1])
        risk_mult = env.risk_multiplier((new_x, new_y))
        adjusted_distance = travel_distance * risk_mult

        if not EnergyModel.can_travel(uav, adjusted_distance):
            return False

        energy = EnergyModel.energy_for_distance(uav, adjusted_distance)
        EnergyModel.consume(uav, energy)

        if not EnergyModel.can_return_to_base(uav, (new_x, new_y), base_position, risk_mult):
            return False

        uav.x = new_x
        uav.y = new_y

        env.uav_trail.append((new_x, new_y))
        if len(env.uav_trail) > 30:
            env.uav_trail.pop(0)

        return True


# ----------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------

def _constrain_angle(ideal: float, current: float, max_change: float) -> float:
    """Clamp angular change to ±max_change, handling wraparound."""
    diff = (ideal - current + math.pi) % (2 * math.pi) - math.pi
    if abs(diff) > max_change:
        return current + math.copysign(max_change, diff)
    return ideal
