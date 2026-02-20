import os
import time
import json

from config.config import Config
from core.seed_manager import set_global_seed
from core.environment_model import Environment
from core.dataset_generator import generate_nodes
from core.obstacle_model import Obstacle
from core.risk_zone_model import RiskZone
from core.temporal_engine import TemporalEngine
from core.run_manager import RunManager
from core.mission_controller import MissionController
from core.stability_monitor import StabilityMonitor
from core.clustering.cluster_manager import ClusterManager
from visualization.plot_renderer import PlotRenderer
from metrics.metric_engine import MetricEngine


def run_simulation(verbose=True, render=True, seed_override=None):

    # ---------------------------------------------------------
    # Seed Control
    # ---------------------------------------------------------
    if seed_override is not None:
        active_seed = seed_override
    else:
        active_seed = int(time.time()) if Config.RANDOMIZE_SEED else Config.RANDOM_SEED

    set_global_seed(active_seed)

    Config.apply_hostility_profile()

    # ---------------------------------------------------------
    # Run Manager (creates results/runs/<run_id>/...)
    # ---------------------------------------------------------
    run_manager = RunManager()
    run_id = run_manager.get_run_id()

    # ---------------------------------------------------------
    # Core Systems
    # ---------------------------------------------------------
    cluster_manager = ClusterManager()
    stability_monitor = StabilityMonitor()

    env = Environment(Config.MAP_WIDTH, Config.MAP_HEIGHT)
    env.dataset_mode = Config.DATASET_MODE

    temporal = TemporalEngine(Config.TIME_STEP, Config.MAX_TIME_STEPS)
    env.temporal_engine = temporal

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    nodes = generate_nodes(
        Config.DATASET_MODE,
        Config.NODE_COUNT,
        Config.MAP_WIDTH,
        Config.MAP_HEIGHT,
        active_seed,
    )

    for node in nodes:
        env.add_node(node)

    # ---------------------------------------------------------
    # Environment Features
    # ---------------------------------------------------------
    if Config.ENABLE_OBSTACLES:
        env.add_obstacle(Obstacle(200, 200, 350, 350))
        env.add_obstacle(Obstacle(500, 100, 650, 250))

    if Config.ENABLE_RISK_ZONES:
        env.add_risk_zone(RiskZone(100, 400, 300, 550, multiplier=1.8))

    # ---------------------------------------------------------
    # Mission Controller
    # ---------------------------------------------------------
    mission = MissionController(
        env,
        temporal,
        run_manager=run_manager,
        render=render,
    )

    step_counter = 0

    while mission.is_active():
        mission.step()
        step_counter += 1

    # ---------------------------------------------------------
    # Aggregate Results
    # ---------------------------------------------------------
    results = {
        "run_id": run_id,
        "seed": active_seed,
        "steps": step_counter,
        "final_battery": mission.uav.current_battery,
        "visited": len(mission.visited),
        "replans": temporal.replan_count,
        "collisions": mission.collision_count,
        "unsafe_return": mission.unsafe_return_count,
        "event_count": mission.event_count,
        "event_timestamps": mission.event_timestamps,
        "replan_timestamps": mission.replan_timestamps,
        "energy_prediction_error": (
            mission.energy_prediction_error_sum / mission.energy_prediction_samples
            if mission.energy_prediction_samples > 0
            else 0.0
        ),
    }

    # Semantic Evaluations
    if Config.ENABLE_SEMANTIC_CLUSTERING:
        visited_nodes_objs = [n for n in env.nodes if n.id in mission.visited]
        semantic_scores = MetricEngine.compute_semantic_metrics(
            visited_nodes=visited_nodes_objs,
            all_nodes=env.nodes[1:], 
            active_labels=mission.active_labels
        )
        results.update(semantic_scores)

    # ---------------------------------------------------------
    # Persist Run Summary
    # ---------------------------------------------------------
    logs_path = run_manager.get_path("logs")

    summary_file = os.path.join(logs_path, "run_summary.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=4)

    # ---------------------------------------------------------
    # Persist Config Snapshot (critical for reproducibility)
    # ---------------------------------------------------------
    config_snapshot = {
        key: getattr(Config, key) for key in dir(Config) if key.isupper()
    }

    config_file = os.path.join(logs_path, "config_snapshot.json")
    with open(config_file, "w") as f:
        json.dump(config_snapshot, f, indent=4)

    # ---------------------------------------------------------
    # Final Visual Metric Artifact Generation
    # ---------------------------------------------------------
    visuals_path = run_manager.get_path("plots")
    
    PlotRenderer.render_environment(env, visuals_path)
    
    PlotRenderer.render_energy_plots(
        visited=len(mission.visited),
        energy_consumed=mission.energy_consumed_total,
        save_dir=visuals_path
    )
    
    PlotRenderer.render_time_series(
        visited_hist=mission.visited_history,
        battery_hist=mission.battery_history,
        replan_hist=mission.replan_history,
        save_dir=visuals_path
    )

    return results
