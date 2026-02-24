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
from core.communication import CommunicationEngine
from core.buffer_aware_manager import BufferAwareManager

from core.clustering.cluster_manager import ClusterManager
from path.pca_gls_router import PCAGLSRouter
from path.ga_sequence_optimizer import GASequenceOptimizer
from path.hover_optimizer import HoverOptimizer
from core.digital_twin_map import DigitalTwinMap
from core.rendezvous_selector import RendezvousSelector
from core.obstacle_model import ObstacleHeightModel
from core.base_station_uplink import BaseStationUplinkModel


class MissionController:
    def __init__(
        self, env: Environment, temporal: TemporalEngine, run_manager=None, render=True
    ):
        self.env = env
        self.temporal = temporal
        self.run_manager = run_manager
        self.render_enabled = render

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
        
        # ISAC Digital Twin
        self.digital_twin = DigitalTwinMap()

        # Semantic Intelligence Engine
        self.cluster_manager = ClusterManager()
        self.active_centroids = []
        self.active_labels = []

        # Rendezvous Point analytics (Gap 1)
        self.rp_member_map: dict = {}  # {rp_id: [member_node_ids]}

        # RP+GA plan cache — recompute only when visit set changes (perf fix)
        self._rp_cache_key: frozenset = frozenset()
        self._cached_queue: list = []

        # Metrics
        self.energy_consumed_total = 0.0
        self.collision_count = 0
        self.unsafe_return_count = 0

        # Phase-3 Energy Prediction Error
        self.energy_prediction_error_sum = 0.0
        self.energy_prediction_samples = 0

        # IEEE-aligned metric instrumentation (MetricsDashboard)
        self.rate_log: list = []            # per-step achievable Shannon rate (Mbps)
        self.collected_data_mbits: float = 0.0  # cumulative data collected across all nodes

        # Base Station Uplink state (Gap 10)
        self.last_uplink_step: int = 0
        self.total_uplinked_mbits: float = 0.0

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

        if self.render_enabled:
            # We enforce Matplotlib interactive dash for Phase 3.6 fidelity
            if not hasattr(self, 'interactive_dash'):
                from visualization.interactive_dashboard import InteractiveDashboard
                self.interactive_dash = InteractiveDashboard(self.env)
                
            self.interactive_dash.render(
                self.uav, 
                self.current_target, 
                self.temporal.current_step, 
                self.base_position, 
                self.active_centroids
            )
            # print(f"[Time Step] {self.temporal.current_step}")

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

        # Periodically re-evaluate Semantic Clusters
        unvisited_nodes = [n for n in self.env.nodes[1:] if n.id not in self.visited]
        if self.temporal.current_step % 50 == 0 or self.cluster_manager.should_recluster(len(unvisited_nodes)):
            self.active_centroids, self.active_labels = self.cluster_manager.perform_clustering(
                unvisited_nodes, self.temporal.current_step
            )

        # Replan handling
        if self.temporal.replan_required:
            self._recompute_plan()
            self.temporal.reset_replan()

        # Energy threshold
        if EnergyModel.should_return(self.uav):
            print("[Mission] Battery threshold reached.")
            self.temporal.active = False
            return

        # Fill buffers for all unvisited nodes based on Generation Rate
        # Pass UAV position only for nodes within sensing radius so the first-order
        # radio TX energy model (Gap 2) drains their battery during active transmission.
        dt = float(Config.TIME_STEP)
        uav_x, uav_y, uav_z = self.uav.position()
        for node in self.env.nodes[1:]:
            if node.id not in self.visited:
                import math
                dist_to_node = math.hypot(node.x - uav_x, node.y - uav_y)
                uav_nearby = dist_to_node <= Config.ISAC_SENSING_RADIUS
                CommunicationEngine.fill_buffer(
                    node, dt,
                    uav_pos=self.uav.position() if uav_nearby else None
                )

        # Gap 10: BS Uplink Urgency Check (Zheng & Liu, IEEE TVT 2025 — Eq. 25)
        # Periodically evaluate data-age constraint: T_collect + T_fly + T_uplink ≤ T_limit
        if (Config.ENABLE_BS_UPLINK_MODEL
                and self.temporal.current_step % Config.BS_UPLINK_CHECK_INTERVAL == 0
                and self.collected_data_mbits > 0):
            must_uplink = BaseStationUplinkModel.must_uplink_now(
                uav_pos=self.uav.position(),
                base_pos=self.base_position,
                payload_mbits=self.collected_data_mbits,
                current_step=self.temporal.current_step,
                last_uplink_step=self.last_uplink_step,
            )
            if must_uplink:
                print(f"[BS Uplink] Step {self.temporal.current_step}: data-age limit approaching — offloading payload.")
                BaseStationUplinkModel.execute_uplink(self, self.temporal.current_step)
                self._trigger_replan("bs_uplink_return")

        self._move_one_step()


        # Histories
        self.visited_history.append(len(self.visited))
        self.battery_history.append(self.uav.current_battery)
        self.replan_history.append(self.temporal.replan_count)

        # Frame saving
        if (
            self.render_enabled
            and Config.ENABLE_VISUALS
            and self.run_manager
            and self.temporal.current_step % Config.FRAME_SAVE_INTERVAL == 0
        ):
            PlotRenderer.render_environment_frame(
                self.env,
                self.run_manager.get_path("frames"),
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

            # Log actual adaptation execution time (same step)
            self.replan_timestamps.append(self.temporal.current_step)

            self.last_replan_step = self.temporal.current_step

    def _recompute_plan(self):

        # Remaining unvisited nodes (exclude UAV anchor at index 0)
        remaining = [n for n in self.env.nodes[1:] if n.id not in self.visited]

        if not remaining:
            self.target_queue = []
            self.current_target = None
            return

        # ---- Cache check: skip RP+GA if visit set unchanged (e.g. trap escapes) ----
        current_key = frozenset(n.id for n in remaining)
        if current_key == self._rp_cache_key and self._cached_queue:
            # restore cached queue (filter out already-visited targets)
            self.target_queue = [n for n in self._cached_queue if n.id not in self.visited]
            if self.target_queue:
                return   # serve existing plan, skip expensive RP+GA

        # ---- Gap 1: Rendezvous Point Selection (Donipati et al., Algorithm 1) ----
        # Compress the full node set to a minimal RP subset so the UAV visits
        # far fewer waypoints, dramatically reducing path length and energy.
        # Fix 3: pass obstacles so obstacle-adjacent nodes are excluded as RPs.
        if Config.ENABLE_RENDEZVOUS_SELECTION and len(remaining) > 3:
            rp_nodes, rp_member_map = RendezvousSelector.apply(
                remaining, obstacles=self.env.obstacles
            )
            if rp_nodes:
                self.rp_member_map = rp_member_map
                remaining = rp_nodes


        ux, uy, uz = self.uav.position()

        # Phase-4 Hierarchical Semantic Routing: 
        # If Semantic Clustering is actively tracking centroids, 
        # find the closest latent density centroid and route to its member nodes first.
        if len(self.active_centroids) > 0 and len(self.active_labels) == len(remaining):
            import numpy as np
            distances_to_centroids = [
                (idx, np.linalg.norm(np.array([ux, uy, uz]) - centroid)) 
                for idx, centroid in enumerate(self.active_centroids) 
                if not np.all(centroid == 0) # Ignore empty noise clusters
            ]
            
            if distances_to_centroids:
                closest_cluster_idx = min(distances_to_centroids, key=lambda x: x[1])[0]
                # Filter remaining to only the semantic locality
                priority_subgroup = [
                    remaining[i] for i, label in enumerate(self.active_labels) 
                    if label == closest_cluster_idx
                ]
                # Fallback if DBSCAN noise filtered out all valid subgroup routers
                if priority_subgroup:
                    remaining = priority_subgroup

        # Stage 1: PCA-GLS Meta-Heuristic seed on the priority subgroup
        pca_gls_order = PCAGLSRouter.optimize((ux, uy, uz), remaining)

        # Stage 2: GA Visiting Sequence Refinement (Gap 4 — Zheng & Liu, IEEE TVT 2025)
        # Uses PCA-GLS result as warm-start seed; refines with OX crossover + swap mutation
        # subject to time-window feasibility constraints [e_j, l_j].
        if Config.ENABLE_GA_SEQUENCE and len(remaining) >= 4:
            self.target_queue = GASequenceOptimizer.apply(
                start_pos=(ux, uy, uz),
                nodes=remaining,
                seed_order=pca_gls_order,
            )
            print(f"[GA Optimizer] Refined sequence of {len(self.target_queue)} nodes via GA+PCA-GLS")
        else:
            self.target_queue = pca_gls_order

        # Write back cache so trap-escape replans reuse this plan without re-running GA
        self._rp_cache_key = current_key
        self._cached_queue = self.target_queue[:]

        self.current_target = None



    def _rectangle_clearance(self, x, y, obs):
        dx = max(obs.x1 - x, 0, x - obs.x2)
        dy = max(obs.y1 - y, 0, y - obs.y2)
        return math.hypot(dx, dy)

    def _predicted_clearance(self, x, y, obs):
        if not Config.ENABLE_MOVING_OBSTACLES:
            return self._rectangle_clearance(x, y, obs)

        # Predict next position (linear motion model)
        pred_x1 = obs.x1 + obs.vx
        pred_x2 = obs.x2 + obs.vx
        pred_y1 = obs.y1 + obs.vy
        pred_y2 = obs.y2 + obs.vy

        dx = max(pred_x1 - x, 0, x - pred_x2)
        dy = max(pred_y1 - y, 0, y - pred_y2)

        return math.hypot(dx, dy)

    # ---------------------------------------------------------
    # Motion Primitive Selection (Research-Clean Version)
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
        
        from config.feature_toggles import FeatureToggles
        if FeatureToggles.DIMENSIONS == "3D":
            dz = target_pos[2] - current_pos[2]
            distance = math.sqrt(dx**2 + dy**2 + dz**2)
            base_pitch = math.atan2(dz, math.hypot(dx, dy))
        else:
            dz = 0.0
            distance = math.hypot(dx, dy)
            base_pitch = 0.0

        dt = float(Config.TIME_STEP)

        # -------------------------------------------------------------
        # Buffer-Aware Dynamic Service Time (DST-BA): 'Chord-Fly' Check
        # -------------------------------------------------------------
        # The UAV is capable of collecting data while in transit 
        # (probabilistic sensing over distance).
        data_collected = BufferAwareManager.process_data_collection(
            self.uav.position(), self.current_target, dt, self.env,
            active_node_id=self.current_target.id  # Gap 7: TDMA — only this node transmits
        )


        # IEEE MetricsDashboard instrumentation: log achievable rate and cumulative data
        if data_collected > 0:
            self.collected_data_mbits += data_collected
            # Compute instantaneous achievable Shannon rate at current UAV-node distance
            _rate = CommunicationEngine.achievable_data_rate(
                self.current_target.position(), self.uav.position()
            )
            self.rate_log.append(_rate)

        if data_collected > 0 and distance > 5.0:
            print(f"[Chord-Fly] Node {self.current_target.id} at {distance:.1f}m | Collected {data_collected:.2f}Mb | Rem: {self.current_target.current_buffer:.2f}Mb")
            
        # Check if the buffer is empty. If so, we bypass reaching the exact coordinate!
        if self.current_target.current_buffer <= 1e-3:
            self.visited.add(self.current_target.id)
            print(f"[Visited] Node {self.current_target.id} (Buffer completely drained via DST-BA)")
            self.current_target = None
            return

        # -------------------------------------------------------------
        # Target Reached: 'Center-Hover' Fallback
        # -------------------------------------------------------------
        if distance < 1e-3:
            # We are exactly above the node, but buffer is still not empty. Hover in place.
            hover_e = EnergyModel.hover_energy(self.uav, dt)
            EnergyModel.consume(self.uav, hover_e)
            self.energy_consumed_total += hover_e
            
            hover_strategy = BufferAwareManager.get_optimal_hover_strategy(self.uav.position(), self.current_target, self.env)
            
            # Phase 3.8: LoS Occlusion Failsafe
            if hover_strategy['required_service_time'] == float('inf'):
                print(f"[Warning] Node {self.current_target.id} Data Rate is 0 Mbps (NLoS Blocked). Abandoning target.")
                self.visited.add(self.current_target.id) # Mark visited to avoid infinite loops
                self.current_target = None
                return
                
            print(f"[Center-Hover] Node {self.current_target.id} | Reqd Time: {hover_strategy['required_service_time']:.2f}s | Vol: {self.current_target.current_buffer:.2f}Mb")
            return

        dt = float(Config.TIME_STEP)
        
        # Acceleration Dynamics
        v_current_mag = math.sqrt(self.uav.vx**2 + self.uav.vy**2 + self.uav.vz**2)
        target_v_mag = min(Config.UAV_STEP_SIZE / dt, distance / dt) if dt > 0 else 0.0
        
        # Max acceleration constraint
        max_dv = Config.MAX_ACCELERATION * dt
        if target_v_mag - v_current_mag > max_dv:
            v_mag = v_current_mag + max_dv
        elif v_current_mag - target_v_mag > max_dv:
            v_mag = max(0.0, v_current_mag - max_dv)
        else:
            v_mag = target_v_mag
            
        step_size = v_mag * dt
        
        ideal_yaw = math.atan2(dy, dx)
        max_yaw_change = math.radians(Config.MAX_YAW_RATE) * dt
        max_pitch_change = math.radians(Config.MAX_PITCH_RATE) * dt
        
        def constrain_angle(ideal, current, max_change):
            diff = (ideal - current)
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) > max_change:
                return current + math.copysign(max_change, diff)
            return ideal
            
        base_yaw = constrain_angle(ideal_yaw, self.uav.yaw, max_yaw_change)
        # Using base_pitch calculated from earlier which correctly extracts dz against dx,dy
        c_base_pitch = constrain_angle(base_pitch, self.uav.pitch, max_pitch_change)

        best_score = float("inf")
        best_move = None
        
        # Phase 3.5: ISAC Digital Twin Scan (Update local map)
        if Config.ENABLE_ISAC_DIGITAL_TWIN:
            self.digital_twin.scan_environment(self.uav.position(), self.env.obstacles)
            routing_obstacles = self.digital_twin.known_obstacles
        else:
            routing_obstacles = self.env.obstacles

        # Generate motion primitives around the constrained base angles
        yaw_offsets = [0] + Config.STEERING_ANGLES
        pitch_offsets = [0, -15, 15] if FeatureToggles.DIMENSIONS == "3D" else [0]
        
        for yaw_offset in yaw_offsets:
            for pitch_offset in pitch_offsets:
                yaw = base_yaw + math.radians(yaw_offset)
                pitch = c_base_pitch + math.radians(pitch_offset)

                # Spherical to Cartesian representation
                new_x = current_pos[0] + step_size * math.cos(pitch) * math.cos(yaw)
                new_y = current_pos[1] + step_size * math.cos(pitch) * math.sin(yaw)
                new_z = current_pos[2] + step_size * math.sin(pitch) if FeatureToggles.DIMENSIONS == "3D" else current_pos[2]

                # Gap 5: 3D Gaussian altitude constraint (Zheng & Liu, IEEE TVT 2025)
                # Clamp z so UAV never flies below z_obs(x,y) + vertical clearance
                new_z = ObstacleHeightModel.enforce_altitude(new_x, new_y, new_z, self.env.obstacles)

                # Hard collision check (2D footprint)
                if self.env.point_in_obstacle((new_x, new_y)):
                    continue

                travel_distance = math.sqrt((new_x - current_pos[0])**2 + (new_y - current_pos[1])**2 + (new_z - current_pos[2])**2)

                risk_mult = self.env.risk_multiplier((new_x, new_y))
                adjusted_distance = travel_distance * risk_mult

                # Energy feasibility
                if not EnergyModel.can_travel(self.uav, adjusted_distance):
                    continue

                # ---------- SCORING COMPONENTS ----------

                # 1. Alignment score (cosine similarity in 3D)
                to_target_vec = (dx, dy, dz)
                move_vec = (new_x - current_pos[0], new_y - current_pos[1], new_z - current_pos[2])

                dot = to_target_vec[0] * move_vec[0] + to_target_vec[1] * move_vec[1] + to_target_vec[2] * move_vec[2]
                norm_prod = (
                    math.sqrt(sum(v**2 for v in to_target_vec)) * math.sqrt(sum(v**2 for v in move_vec)) + Config.SCORE_EPS
                )

                alignment_score = 1 - (dot / norm_prod)  # smaller is better

                # 2. Obstacle proximity penalty
                obstacle_penalty = 0.0

                for obs in routing_obstacles:
                    clearance = self._predicted_clearance(new_x, new_y, obs)

                    # Hard rejection if inside safety margin
                    if clearance < Config.COLLISION_MARGIN:
                        obstacle_penalty += 1000.0
                    else:
                        # Smooth inverse-distance penalty
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
                    best_move = (new_x, new_y, new_z, adjusted_distance, yaw, pitch, v_mag, step_size)

        # If no valid motion primitive found
        if best_move is None:
            self.collision_count += 1
            print(f"[Warning] UAV trapped near ({current_pos[0]:.1f}, {current_pos[1]:.1f}). Forcing aggressive escape bounce.")
            
            # Escape Traps: Aggressive reverse + scatter bounce
            import random
            bounce_dist = 15.0
            escape_yaw = self.uav.yaw + math.pi + random.uniform(-0.5, 0.5)
            
            self.uav.x = max(0.0, min(float(self.env.width), self.uav.x + bounce_dist * math.cos(escape_yaw)))
            self.uav.y = max(0.0, min(float(self.env.height), self.uav.y + bounce_dist * math.sin(escape_yaw)))
            self.uav.yaw = escape_yaw
            
            self._trigger_replan("collision_escape")
            
            # DO NOT set current_target to None here! That breaks the routing queue.
            # Instead, the replan will generate a new valid routing graph around the obstacle.
            return

        new_x, new_y, new_z, adjusted_distance, chosen_yaw, chosen_pitch, v_mag, step_size = best_move
        
        dt = float(Config.TIME_STEP)

        # Update Kinematics
        new_vx = (new_x - self.uav.x) / dt if dt > 0 else 0.0
        new_vy = (new_y - self.uav.y) / dt if dt > 0 else 0.0
        new_vz = (new_z - self.uav.z) / dt if dt > 0 else 0.0
        
        ax = (new_vx - self.uav.vx) / dt if dt > 0 else 0.0
        ay = (new_vy - self.uav.vy) / dt if dt > 0 else 0.0
        az = (new_vz - self.uav.vz) / dt if dt > 0 else 0.0

        # --- Energy Prediction ---
        # Propulsion aerodynamic power + Mechanical Acceleration Power
        mechanical_energy = EnergyModel.mechanical_energy(self.uav, (ax, ay, az))
        aerodynamic_energy = EnergyModel.energy_for_distance(self.uav, adjusted_distance)
        predicted_energy = mechanical_energy + aerodynamic_energy

        EnergyModel.consume(self.uav, predicted_energy)
        self.energy_consumed_total += predicted_energy

        # --- Phase-3 Energy Prediction Error Instrumentation ---
        actual_energy_drop = predicted_energy  # deterministic model (for now)
        prediction_error = abs(predicted_energy - actual_energy_drop)

        self.energy_prediction_error_sum += prediction_error
        self.energy_prediction_samples += 1

        # Return safety check
        if not EnergyModel.can_return_to_base(
            self.uav,
            (new_x, new_y),
            self.base_position,
            self.env.risk_multiplier((new_x, new_y)),
        ):
            self.unsafe_return_count += 1
            self._trigger_replan("energy_risk")
            self.temporal.active = False
            return

        # Apply movement
        self.uav.x = new_x
        self.uav.y = new_y
        self.uav.z = new_z
        self.uav.yaw = chosen_yaw
        self.uav.pitch = chosen_pitch
        self.uav.vx = new_vx
        self.uav.vy = new_vy
        self.uav.vz = new_vz

        self.env.uav_trail.append((new_x, new_y, new_z))
        if len(self.env.uav_trail) > 30:
            self.env.uav_trail.pop(0)

    # ---------------------------------------------------------
    # Predictive Safety Check
    # ---------------------------------------------------------

    def _is_direction_safe(self, current_pos, angle, step_size):

        for i in range(1, Config.PREDICTION_HORIZON + 1):

            sim_x = current_pos[0] + i * step_size * math.cos(angle)
            sim_y = current_pos[1] + i * step_size * math.sin(angle)

            # Predict obstacle future positions
            routing_obstacles = self.digital_twin.known_obstacles if Config.ENABLE_ISAC_DIGITAL_TWIN else self.env.obstacles
            for obs in routing_obstacles:
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
