from typing import List, Optional, Set
import math
import random

from config.config import Config
from core.temporal_engine import TemporalEngine
from core.models.energy_model import EnergyModel
from core.models.environment_model import Environment
from core.models.node_model import Node, UAVState, SensorNode
from core.dataset_generator import spawn_single_node
from core.comms.communication import CommunicationEngine
from core.comms.buffer_aware_manager import BufferAwareManager

from core.clustering.cluster_manager import ClusterManager
from path.pca_gls_router import PCAGLSRouter
from path.ga_sequence_optimizer import GASequenceOptimizer
from path.hover_optimizer import HoverOptimizer
from core.sensing.digital_twin_map import DigitalTwinMap
from core.rendezvous_selector import RendezvousSelector
from core.comms.base_station_uplink import BaseStationUplinkModel
from core.physics_engine import PhysicsEngine


class MissionController:
    def __init__(
        self, env: Environment, temporal: TemporalEngine, run_manager=None
    ):
        self.env = env
        self.temporal = temporal
        self.run_manager = run_manager

        # UAV anchor
        self.uav: UAVState = env.uav

        # Mission state
        self.target_queue: List[SensorNode] = []
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
        self.rp_nodes: list = []  # list of RP Node objects

        # RP+GA plan cache — recompute only when visit set changes (perf fix)
        self._rp_cache_key: frozenset = frozenset()
        self._cached_queue: list = []

        # SCA Hover Position Map — refined hover coords per node (Gap 5: Zheng & Liu §III-E)
        self._hover_positions: dict = {}  # {node_id: (x, y, z)}

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
        self.replan_history  = []
        self.aoi_history     = {}   # node_id → [aoi_per_step]
        self.aoi_mean_history         = []  # mean AoI across all nodes per step
        self.collected_data_history   = []  # cumulative Mbits collected per step

        # Replan cooldown
        self.last_replan_step = -100

        # ---------------------------------------------------------
        # Phase-3 Stability Instrumentation
        # ---------------------------------------------------------
        self.event_count = 0
        self.event_timestamps = []
        self.replan_timestamps = []

        # UAV trail (env.uav already set by add_node; keep trail attr)
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
            and len(self.env.sensors) < Config.NODE_COUNT + Config.MAX_DYNAMIC_NODES
        ):
            new_id = len(self.env.sensors) + 1
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
        unvisited_nodes = [n for n in self.env.sensors if n.id not in self.visited]
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
        for node in self.env.sensors:
            if node.id not in self.visited:
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
        self.collected_data_history.append(self.collected_data_mbits)

        # Per-node AoI snapshot and mean AoI history
        step_aoi_vals = []
        for node in self.env.sensors:
            if node.id not in self.aoi_history:
                self.aoi_history[node.id] = []
            self.aoi_history[node.id].append(node.aoi_timer)
            step_aoi_vals.append(node.aoi_timer)
        self.aoi_mean_history.append(
            sum(step_aoi_vals) / len(step_aoi_vals) if step_aoi_vals else 0.0
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
        remaining = [n for n in self.env.sensors if n.id not in self.visited]

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
                self.rp_nodes = rp_nodes
                remaining = rp_nodes


        ux, uy, uz = self.uav.position()

        # Phase-4 Hierarchical Semantic Routing:
        # Score each cluster by urgency (priority × AoI) discounted by UAV distance,
        # then route to the highest-urgency cluster first.
        if len(self.active_centroids) > 0 and len(self.active_labels) == len(remaining):
            import numpy as np
            uav_vec = np.array([ux, uy, uz])

            # Build per-cluster urgency score: mean_priority × (1 + mean_aoi/MAX_AOI_LIMIT)
            # discounted by proximity to UAV.
            cluster_scores = {}
            for idx, centroid in enumerate(self.active_centroids):
                if np.all(centroid == 0):
                    continue  # skip empty noise cluster
                members = [
                    remaining[i] for i, label in enumerate(self.active_labels)
                    if label == idx
                ]
                if not members:
                    continue
                mean_priority = sum(n.priority for n in members) / len(members)
                mean_aoi      = sum(getattr(n, "aoi_timer", 0.0) for n in members) / len(members)
                urgency       = mean_priority * (1.0 + mean_aoi / max(Config.MAX_AOI_LIMIT, 1))
                dist          = float(np.linalg.norm(uav_vec - centroid))
                # Higher urgency / closer clusters get higher scores
                cluster_scores[idx] = urgency / (1.0 + dist)

            if cluster_scores:
                best_cluster_idx = max(cluster_scores, key=cluster_scores.get)
                priority_subgroup = [
                    remaining[i] for i, label in enumerate(self.active_labels)
                    if label == best_cluster_idx
                ]
                # Fallback if DBSCAN noise filtered out all valid subgroup members
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

        # Stage 3: SCA Hover Position Refinement (Gap 5 — Zheng & Liu, IEEE TVT 2025 §III-E)
        # Optimises the exact (x, y, z) hover coordinate above each node to minimise
        # inter-waypoint travel cost while respecting obstacle altitude constraints.
        if Config.ENABLE_SCA_HOVER and self.target_queue:
            hover_coords = HoverOptimizer.apply_sequence(
                uav_start=(ux, uy, uz),
                nodes=self.target_queue,
                obstacles=self.env.obstacles,
            )
            self._hover_positions = {
                node.id: pos
                for node, pos in zip(self.target_queue, hover_coords)
            }
            print(f"[SCA Hover] Refined {len(hover_coords)} hover positions (α={Config.SCA_STEP_SIZE}, tol={Config.SCA_CONVERGENCE_TOL})")
        else:
            self._hover_positions = {}

        # Write back cache so trap-escape replans reuse this plan without re-running GA
        self._rp_cache_key = current_key
        self._cached_queue = self.target_queue[:]

        self.current_target = None



    # ---------------------------------------------------------
    # Motion: Target Selection, Service, and Physics Delegation
    # ---------------------------------------------------------

    def _move_one_step(self):

        if not self.current_target:
            if not self.target_queue:
                return
            self.current_target = self.target_queue.pop(0)

        # Use SCA-refined hover position if available; otherwise fall back to node position
        target_pos = self._hover_positions.get(
            self.current_target.id, self.current_target.position()
        )
        current_pos = self.uav.position()

        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        dz = target_pos[2] - current_pos[2]
        distance = math.sqrt(dx**2 + dy**2 + dz**2)

        dt = float(Config.TIME_STEP)

        # -------------------------------------------------------------
        # Buffer-Aware Dynamic Service Time (DST-BA): 'Chord-Fly' Check
        # -------------------------------------------------------------
        data_collected = BufferAwareManager.process_data_collection(
            self.uav.position(), self.current_target, dt, self.env,
            active_node_id=self.current_target.id
        )

        if data_collected > 0:
            self.collected_data_mbits += data_collected
            _rate = CommunicationEngine.achievable_data_rate(
                self.current_target.position(), self.uav.position()
            )
            self.rate_log.append(_rate)

        if data_collected > 0 and distance > 5.0:
            print(f"[Chord-Fly] Node {self.current_target.id} at {distance:.1f}m | Collected {data_collected:.2f}Mb | Rem: {self.current_target.current_buffer:.2f}Mb")

        # Check if buffer drained via chord-fly
        if self.current_target.current_buffer <= 1e-3:
            self.visited.add(self.current_target.id)
            print(f"[Visited] Node {self.current_target.id} (Buffer completely drained via DST-BA)")
            self.current_target = None
            return

        # -------------------------------------------------------------
        # Target Reached: Analytical Service (Donipati et al., TNSM 2025)
        # -------------------------------------------------------------
        if distance < 1e-3:
            outcome = BufferAwareManager.execute_service(
                uav=self.uav,
                node=self.current_target,
                env=self.env,
                temporal=self.temporal,
            )

            if outcome['abandoned']:
                print(f"[Warning] Node {self.current_target.id} Data Rate is 0 Mbps (NLoS Blocked). Abandoning target.")
                self.visited.add(self.current_target.id)
                self.current_target = None
                return

            self.energy_consumed_total += outcome['energy_consumed']
            if outcome['data_collected'] > 0:
                self.collected_data_mbits += outcome['data_collected']
                self.rate_log.append(outcome['achievable_rate'])

            # Fast-forward other nodes' buffers and AoI by the service duration
            t_service = outcome['service_time_s']
            if t_service > 0:
                for sensor in self.env.sensors:
                    if sensor.id != self.current_target.id and sensor.id not in self.visited:
                        sensor.current_buffer = min(
                            sensor.buffer_capacity,
                            sensor.current_buffer + sensor.data_generation_rate * t_service
                        )
                        sensor.aoi_timer += t_service
                        if Config.ENABLE_AOI_EXPIRATION and sensor.aoi_timer >= Config.MAX_AOI_LIMIT:
                            sensor.max_aoi_timer = max(sensor.max_aoi_timer, sensor.aoi_timer)
                            sensor.current_buffer = 0.0
                            sensor.aoi_timer = 0.0

            nid = self.current_target.id
            residual = self.current_target.current_buffer
            if residual <= 1e-3:
                print(f"[Service] Node {nid}: τ*={outcome['service_time_s']:.1f}s, collected {outcome['data_collected']:.2f}Mb")
            else:
                print(f"[Service Timeout] Node {nid}: τ*={outcome['service_time_s']:.1f}s, residual {residual:.2f}Mb")

            self.visited.add(nid)
            self.current_target = None
            return

        # -------------------------------------------------------------
        # Physics-Based Movement (delegated to PhysicsEngine)
        # -------------------------------------------------------------
        move = PhysicsEngine.execute_movement(
            self.uav, target_pos, self.env, self.digital_twin, self.base_position
        )

        # Track energy and prediction error
        self.energy_consumed_total += move.energy_consumed
        if move.energy_consumed > 0:
            # Energy prediction error: stochastic wind/drag perturbation vs
            # deterministic model. Simulates real-world model mismatch.
            wind_factor = 1.0 + random.gauss(0.0, 0.05)  # ±5% noise
            actual_energy = move.energy_consumed * wind_factor
            error = abs(actual_energy - move.energy_consumed)
            self.energy_prediction_error_sum += error
            self.energy_prediction_samples += 1

        if move.collision:
            self.collision_count += 1
            print(f"[Warning] UAV trapped near ({current_pos[0]:.1f}, {current_pos[1]:.1f}). Forcing aggressive escape bounce.")
            self._trigger_replan(move.replan_reason)
            return

        if move.unsafe_return:
            self.unsafe_return_count += 1
            self._trigger_replan(move.replan_reason)
            self.temporal.active = False
            return

    # ---------------------------------------------------------
    # Terminal
    # ---------------------------------------------------------

    def _check_terminal_conditions(self):
        if not self.target_queue and not self.current_target:
            print("[Mission] All targets processed.")
            self.temporal.active = False
