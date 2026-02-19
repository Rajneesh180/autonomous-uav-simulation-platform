from typing import List, Optional, Set
import math
import random

from config.config import Config
from core.temporal_engine import TemporalEngine
from core.energy_model import EnergyModel
from core.environment_model import Environment
from core.node_model import Node
from core.dataset_generator import spawn_single_node
from visualization.plot_renderer import PlotRenderer


class MissionController:
    def __init__(self, env: Environment, temporal: TemporalEngine, run_manager=None):
        self.env = env
        self.temporal = temporal
        self.run_manager = run_manager

        # UAV anchor
        self.uav: Node = env.nodes[0]

        # Mission state
        self.target_queue: List[Node] = []
        self.current_target: Optional[Node] = None
        self.visited: Set[int] = set()

        # Safe start
        center = (self.env.width // 2, self.env.height // 2)
        safe_start = self.env.get_safe_start(center)
        self.uav.x, self.uav.y = safe_start
        self.base_position = safe_start

        # Metrics
        self.energy_consumed_total = 0.0
        self.collision_count = 0
        self.unsafe_return_count = 0

        # Histories
        self.visited_history = []
        self.battery_history = []
        self.replan_history = []

        # Replan cooldown
        self.last_replan_step = -100

        # ---------------------------------------------------------
        # Phase-3 Stability Instrumentation
        # ---------------------------------------------------------
        self.event_count = 0
        self.event_timestamps = []
        self.replan_timestamps = []

        # UAV trail
        self.env.uav = self.uav
        self.env.uav_trail = []

        self._initialize_queue()

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------

    def _initialize_queue(self):
        self._recompute_plan()

    def is_active(self):
        return self.temporal.active and self.uav.current_battery > 0

    # ---------------------------------------------------------
    # Main Loop
    # ---------------------------------------------------------

    def step(self):
        if not self.temporal.tick():
            return

        print(f"[Time Step] {self.temporal.current_step}")

        # Obstacle motion toggle
        if Config.ENABLE_MOVING_OBSTACLES:
            self.env.update_obstacles()

        self.env.update_risk_zones(self.temporal.current_step)

        # Dynamic node removal
        if (
            Config.ENABLE_NODE_REMOVAL
            and self.temporal.current_step % Config.NODE_REMOVAL_INTERVAL == 0
        ):
            if random.random() < Config.NODE_REMOVAL_PROBABILITY:
                removed = self.env.remove_random_node(Config.MIN_NODE_FLOOR)
                if removed:
                    self._trigger_replan("node_removed")

        # Dynamic node spawn
        if (
            Config.ENABLE_DYNAMIC_NODES
            and self.temporal.current_step % Config.DYNAMIC_NODE_INTERVAL == 0
            and len(self.env.nodes) < Config.NODE_COUNT + Config.MAX_DYNAMIC_NODES
        ):
            new_id = len(self.env.nodes)
            new_node = spawn_single_node(
                self.env.width,
                self.env.height,
                new_id,
                self.env,
            )
            if new_node:
                self.env.add_node(new_node)
                self.target_queue.append(new_node)
                self._trigger_replan("node_spawned")

        # Replan handling
        if self.temporal.replan_required:
            self._recompute_plan()
            self.temporal.reset_replan()

        # Energy threshold
        if EnergyModel.should_return(self.uav):
            print("[Mission] Battery threshold reached.")
            self.temporal.active = False
            return

        self._move_one_step()

        # Histories
        self.visited_history.append(len(self.visited))
        self.battery_history.append(self.uav.current_battery)
        self.replan_history.append(self.temporal.replan_count)

        # Frame saving
        if (
            Config.ENABLE_VISUALS
            and self.run_manager
            and self.temporal.current_step % Config.FRAME_SAVE_INTERVAL == 0
        ):
            PlotRenderer.render_environment_frame(
                self.env,
                self.run_manager.get_frames_path(),
                self.temporal.current_step,
            )

        self._check_terminal_conditions()

    # ---------------------------------------------------------
    # Replanning
    # ---------------------------------------------------------

    def _trigger_replan(self, reason):
        if (
            self.temporal.current_step - self.last_replan_step
            >= Config.REPLAN_COOLDOWN_STEPS
        ):
            print(f"[Replan Triggered] Reason: {reason}")

            # --- instrumentation ---
            self.event_count += 1
            self.event_timestamps.append(self.temporal.current_step)

            self.temporal.trigger_replan(reason)
            self.last_replan_step = self.temporal.current_step

    def _recompute_plan(self):
        # Remaining unvisited nodes (exclude UAV anchor at index 0)
        remaining = [n for n in self.env.nodes[1:] if n.id not in self.visited]

        if not remaining:
            self.target_queue = []
            self.current_target = None
            return

        # Sort by Euclidean distance from current UAV position
        ux, uy = self.uav.position()

        remaining.sort(key=lambda n: math.hypot(n.x - ux, n.y - uy))

        self.target_queue = remaining
        self.current_target = None

    # ---------------------------------------------------------
    # Predictive Movement
    # ---------------------------------------------------------

    def _move_one_step(self):

        if not self.current_target:
            if not self.target_queue:
                return
            self.current_target = self.target_queue.pop(0)

        target_pos = self.current_target.position()
        current_pos = self.uav.position()

        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        distance = math.hypot(dx, dy)

        if distance < 1e-3:
            self.visited.add(self.current_target.id)
            print(f"[Visited] Node {self.current_target.id}")
            self.current_target = None
            return

        step_size = min(Config.UAV_STEP_SIZE, distance)
        base_angle = math.atan2(dy, dx)

        candidate_angles = [0] + Config.STEERING_ANGLES

        for angle_offset in candidate_angles:
            angle = base_angle + math.radians(angle_offset)

            if self._is_direction_safe(current_pos, angle, step_size):
                return

        # All directions failed
        self.collision_count += 1
        self._trigger_replan("collision")
        self.current_target = None

    # ---------------------------------------------------------
    # Predictive Safety Check
    # ---------------------------------------------------------

    def _is_direction_safe(self, current_pos, angle, step_size):

        for i in range(1, Config.PREDICTION_HORIZON + 1):

            sim_x = current_pos[0] + i * step_size * math.cos(angle)
            sim_y = current_pos[1] + i * step_size * math.sin(angle)

            # Predict obstacle future positions
            for obs in self.env.obstacles:
                future_x1 = obs.x1 + i * obs.vx * Config.OBSTACLE_VELOCITY_SCALE
                future_y1 = obs.y1 + i * obs.vy * Config.OBSTACLE_VELOCITY_SCALE
                future_x2 = obs.x2 + i * obs.vx * Config.OBSTACLE_VELOCITY_SCALE
                future_y2 = obs.y2 + i * obs.vy * Config.OBSTACLE_VELOCITY_SCALE

                if future_x1 <= sim_x <= future_x2 and future_y1 <= sim_y <= future_y2:
                    return False

        # Final immediate step energy check
        new_x = current_pos[0] + step_size * math.cos(angle)
        new_y = current_pos[1] + step_size * math.sin(angle)

        travel_distance = math.hypot(new_x - current_pos[0], new_y - current_pos[1])
        risk_mult = self.env.risk_multiplier((new_x, new_y))
        adjusted_distance = travel_distance * risk_mult

        if not EnergyModel.can_travel(self.uav, adjusted_distance):
            return False

        energy = EnergyModel.energy_for_distance(self.uav, adjusted_distance)
        EnergyModel.consume(self.uav, energy)
        self.energy_consumed_total += energy

        if not EnergyModel.can_return_to_base(
            self.uav, (new_x, new_y), self.base_position, risk_mult
        ):
            self.unsafe_return_count += 1
            self._trigger_replan("energy_risk")
            return False

        self.uav.x = new_x
        self.uav.y = new_y

        self.env.uav_trail.append((new_x, new_y))
        if len(self.env.uav_trail) > 30:
            self.env.uav_trail.pop(0)

        return True

    # ---------------------------------------------------------
    # Terminal
    # ---------------------------------------------------------

    def _check_terminal_conditions(self):
        if not self.target_queue and not self.current_target:
            print("[Mission] All targets processed.")
            self.temporal.active = False
