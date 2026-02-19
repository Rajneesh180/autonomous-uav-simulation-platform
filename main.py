import time
import os
import random

from config.config import Config
from core.seed_manager import set_global_seed
from core.environment_model import Environment
from core.dataset_generator import generate_nodes
from core.obstacle_model import Obstacle
from metrics.metric_engine import MetricEngine
from metrics.logger import Logger
from core.energy_model import EnergyModel
from core.risk_zone_model import RiskZone
from visualization.plot_renderer import PlotRenderer
from core.temporal_engine import TemporalEngine
from core.dataset_generator import spawn_single_node
from core.run_manager import RunManager
from clustering.cluster_manager import ClusterManager
from core.stability_monitor import StabilityMonitor


def main():
    print("=== Autonomous UAV Simulation Platform ===")

    # -------- Seed --------
    if Config.RANDOMIZE_SEED:
        active_seed = int(time.time())
        set_global_seed(active_seed)
        print(f"[SeedManager] Random seed set to {active_seed}")
    else:
        active_seed = Config.RANDOM_SEED
        set_global_seed(active_seed)
        print(f"[SeedManager] Global seed set to {active_seed}")

    # -------- Run Manager --------
    run_manager = RunManager()
    print(f"[RunManager] Run ID: {run_manager.get_run_id()}")

    cluster_manager = ClusterManager()

    stability_monitor = StabilityMonitor()

    # -------- Environment --------
    env = Environment(Config.MAP_WIDTH, Config.MAP_HEIGHT)
    env.dataset_mode = Config.DATASET_MODE

    # -------- Temporal Engine --------
    temporal = TemporalEngine(Config.TIME_STEP, Config.MAX_TIME_STEPS)
    env.temporal_engine = temporal

    # -------- Node Generation --------
    nodes = generate_nodes(
        Config.DATASET_MODE,
        Config.NODE_COUNT,
        Config.MAP_WIDTH,
        Config.MAP_HEIGHT,
        active_seed,
    )

    for node in nodes:
        env.add_node(node)

    # -------- Obstacles --------
    if Config.ENABLE_OBSTACLES:
        env.add_obstacle(Obstacle(200, 200, 350, 350))
        env.add_obstacle(Obstacle(500, 100, 650, 250))

    # -------- Risk Zones --------
    if Config.ENABLE_RISK_ZONES:
        env.add_risk_zone(RiskZone(100, 400, 300, 550, multiplier=1.8))

    print("Environment Summary:", env.summary())

    # -------- Energy Simulation --------
    print("\n--- Temporal Simulation ---")

    while temporal.tick():
        env.reset_change_flag()
        print(f"[Time Step] {temporal.current_step}")

        # -------- Environment Updates --------
        env.update_risk_zones(temporal.current_step)
        env.update_obstacles()

        # -------- Dynamic Node Removal --------
        if (
            Config.ENABLE_NODE_REMOVAL
            and temporal.current_step % Config.NODE_REMOVAL_INTERVAL == 0
        ):
            if random.random() < Config.NODE_REMOVAL_PROBABILITY:
                removed = env.remove_random_node(Config.MIN_NODE_FLOOR)
                if removed:
                    env.mark_changed()
                    print("[Dynamic] Node Removed")

        # -------- Dynamic Node Spawn --------
        if (
            Config.ENABLE_DYNAMIC_NODES
            and temporal.current_step % Config.DYNAMIC_NODE_INTERVAL == 0
            and len(env.nodes) < Config.NODE_COUNT + Config.MAX_DYNAMIC_NODES
        ):
            new_id = len(env.nodes)
            new_node = spawn_single_node(
                Config.MAP_WIDTH,
                Config.MAP_HEIGHT,
                new_id,
                env,
            )

            if new_node:
                env.add_node(new_node)
                env.mark_changed()
                print(f"[Dynamic] Node Spawned: {new_id}")
            else:
                print("[Dynamic] Spawn Skipped (Obstacle Collision)")

        # -------- Reclustering Trigger --------
        current_nodes = env.get_node_count()
        if cluster_manager.should_recluster(current_nodes):
            print("[Cluster] Recomputing clusters...")
            cluster_manager.mark_clustered(current_nodes)

        # -------- Replan Trigger --------
        if env.environment_changed:
            print("[Temporal] Environment mutated during simulation")
            temporal.trigger_replan("environment_changed")
            env.reset_change_flag()

        if temporal.replan_required:
            print(f"[Replan Triggered] Reason: {temporal.replan_reason}")
            temporal.reset_replan()

        # -------- Frame Capture --------
        PlotRenderer.render_environment_frame(
            env,
            run_manager.get_frames_path(),
            temporal.current_step,
        )

    # Existing energy simulation logic goes here
    print("\n--- Energy Simulation ---")

    if not env.nodes:
        print("No nodes generated. Exiting.")
        return
    uav = env.nodes[0]

    center = (env.width // 2, env.height // 2)
    base_position = env.get_safe_start(center)
    uav.x, uav.y = base_position

    visited = 0
    attempted = 0
    energy_consumed_total = 0
    abort_reason = None
    return_triggered = False

    # NEW COUNTERS
    collision_count = 0
    unsafe_return_count = 0

    # -------- Temporal History Trackers --------
    visited_history = []
    battery_history = []
    replan_history = []

    frame_counter = 10000
    for target in env.nodes[1:]:

        attempted += 1

        # -------- Collision Check --------
        if env.has_collision(uav.position(), target.position()):
            collision_count += 1
            temporal.trigger_replan("obstacle_blocked")
            print(f"[Replan Triggered] Reason: {temporal.replan_reason}")
            print(f"Path blocked by obstacle to Node {target.id}")
            continue

        # -------- Threshold Check --------
        if EnergyModel.should_return(uav):
            abort_reason = "threshold"
            return_triggered = True
            print("Battery threshold reached. Returning to base.")
            break

        distance = MetricEngine.euclidean_distance(uav.position(), target.position())

        # -------- Risk Multiplier --------
        risk_mult = env.risk_multiplier(target.position())
        distance *= risk_mult

        # -------- Feasibility Check --------
        if not EnergyModel.can_travel(uav, distance):
            abort_reason = "insufficient_energy"
            print("Cannot reach next node safely.")
            break

        # -------- Consume Energy --------
        energy_needed = EnergyModel.energy_for_distance(uav, distance)
        EnergyModel.consume(uav, energy_needed)
        energy_consumed_total += energy_needed

        # -------- Return Safety --------
        if not EnergyModel.can_return_to_base(uav, target.position(), base_position):
            unsafe_return_count += 1
            abort_reason = "unsafe_return"
            temporal.trigger_replan("energy_risk")
            print(f"[Replan Triggered] Reason: {temporal.replan_reason}")
            print("Return-to-base unsafe. Mission halted.")
            break

        # -------- Motion Interpolation --------
        steps = 10
        tx, ty = target.position()
        dx = (tx - uav.x) / steps
        dy = (ty - uav.y) / steps

        motion_blocked = False

        for s in range(steps):
            next_x = uav.x + dx
            next_y = uav.y + dy

            # ----- Mid-Motion Collision Guard -----
            if env.point_in_obstacle((next_x, next_y)):
                collision_count += 1
                temporal.trigger_replan("mid_motion_collision")
                print("[Replan Triggered] Reason: mid_motion_collision")
                break

            uav.x = next_x
            uav.y = next_y

            # ----- Frame Capture -----
            frame_counter += 1
            PlotRenderer.render_environment_frame(
                env, run_manager.get_frames_path(), frame_counter
            )

        # ----- Snap Exact Target (float drift fix) -----
        uav.x, uav.y = tx, ty

        visited += 1

        print(
            f"Visited Node {target.id} | "
            f"Battery Left: {round(uav.current_battery, 2)}"
        )

        # -------- Temporal Tracking --------
        visited_history.append(visited)
        battery_history.append(uav.current_battery)
        replan_history.append(temporal.replan_count)

    # -------- MetricEngine Derived Metrics --------
    mission_completion = MetricEngine.mission_completion(visited, attempted)

    energy_efficiency = MetricEngine.energy_efficiency(energy_consumed_total, visited)

    abort_flag = MetricEngine.abort_flag(abort_reason)
    return_flag = MetricEngine.return_flag(return_triggered)

    coverage_ratio = MetricEngine.coverage_ratio(visited, len(env.nodes) - 1)

    constraint_flag = MetricEngine.constraint_violation_flag(
        collision_count, unsafe_return_count
    )

    print(f"Total Visited: {visited}")
    print(f"Energy Consumed: {round(energy_consumed_total, 2)}")
    print(f"Mission Completion %: {mission_completion}")
    print(f"Energy Efficiency: {energy_efficiency}")
    print(f"Coverage Ratio: {coverage_ratio}")
    print(f"Constraint Violations: {constraint_flag}")
    print(f"Replan Count: {temporal.replan_count}")
    print(f"Recluster Count: {cluster_manager.get_recluster_count()}")

    # -------- Metrics Test --------
    timer_start = MetricEngine.start_timer()

    sample_positions = [node.position() for node in env.nodes[:10]]
    sample_path_length = MetricEngine.path_length(sample_positions)

    runtime = MetricEngine.end_timer(timer_start)

    print(f"Sample Path Length: {round(sample_path_length, 2)}")
    print(f"Runtime: {runtime}s")

    # -------- Temporal Metrics --------
    initial_nodes = Config.NODE_COUNT
    final_nodes = env.get_node_count()

    adaptation_latency = temporal.replan_count / max(1, temporal.current_step)
    node_churn_rate = abs(final_nodes - initial_nodes) / max(1, initial_nodes)
    path_stability_index = 1 / (1 + temporal.replan_count)
    energy_prediction_error = energy_consumed_total / max(1, visited)

    print(f"Adaptation Latency: {round(adaptation_latency, 3)}")
    print(f"Node Churn Rate: {round(node_churn_rate, 3)}")
    print(f"Path Stability Index: {round(path_stability_index, 3)}")
    print(f"Energy Prediction Error: {round(energy_prediction_error, 3)}")

    stability_monitor.record(
        temporal.replan_count, node_churn_rate, energy_prediction_error
    )

    stability = stability_monitor.stability_score()
    print(f"Stability Score: {stability}")

    # -------- Logging --------
    if Config.ENABLE_LOGGING:
        payload = {
            "timestamp": Logger.timestamp(),
            "node_count": env.get_node_count(),
            "sample_path_length": round(sample_path_length, 2),
            "runtime": runtime,
            "energy_consumed": round(energy_consumed_total, 2),
            "visited_nodes": visited,
            "attempted_nodes": attempted,
            "mission_completion_pct": mission_completion,
            "energy_efficiency": energy_efficiency,
            "coverage_ratio": coverage_ratio,
            "constraint_flag": constraint_flag,
            "abort_flag": abort_flag,
            "return_flag": return_flag,
            "replan_count": temporal.replan_count,
            "recluster_count": cluster_manager.get_recluster_count(),
            # -------- NEW TEMPORAL METRICS --------
            "adaptation_latency": round(adaptation_latency, 3),
            "node_churn_rate": round(node_churn_rate, 3),
            "path_stability_index": round(path_stability_index, 3),
            "energy_prediction_error": round(energy_prediction_error, 3),
            "stability_score": stability,
        }

        log_dir = run_manager.get_logs_path()

        Logger.log_json(os.path.join(log_dir, "run_log.json"), payload)

        Logger.log_csv(
            os.path.join(log_dir, "run_metrics.csv"),
            list(payload.keys()),
            list(payload.values()),
        )

        print(f"Logs written to {run_manager.get_logs_path()}")

    # -------- Visualization --------
    if Config.ENABLE_VISUALS:
        PlotRenderer.render_environment(env, run_manager.get_figures_path())
        PlotRenderer.render_energy_plots(
            visited, energy_consumed_total, run_manager.get_plots_path()
        )
        PlotRenderer.render_metrics_snapshot(
            mission_completion, energy_consumed_total, run_manager.get_plots_path()
        )
        PlotRenderer.render_time_series(
            visited_history,
            battery_history,
            replan_history,
            run_manager.get_plots_path(),
        )


if __name__ == "__main__":
    main()
