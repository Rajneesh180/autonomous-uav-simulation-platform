import os
import time
import json

from config.config import Config
from core.seed_manager import set_global_seed
from core.models.environment_model import Environment
from core.dataset_generator import generate_nodes
from core.models.obstacle_model import Obstacle
from core.models.risk_zone_model import RiskZone
from core.temporal_engine import TemporalEngine
from core.run_manager import RunManager
from core.mission_controller import MissionController
from core.stability_monitor import StabilityMonitor
from core.clustering.cluster_manager import ClusterManager
from core.telemetry_logger import TelemetryLogger
from visualization.plot_renderer import PlotRenderer
from visualization.animation_builder import AnimationBuilder
from metrics.metric_engine import MetricEngine
from metrics.auto_logger import IEEEDocLogger


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
    frames_path = run_manager.get_path("frames")
    telemetry = TelemetryLogger(run_manager.get_path("telemetry"))
    keyframe_interval = max(1, Config.FRAME_SUBSAMPLE_INTERVAL)

    while mission.is_active():
        mission.step()
        telemetry.log_step(step_counter, mission)

        # Render frames only when explicitly requested (single-run mode)
        if render and (step_counter % keyframe_interval == 0):
            PlotRenderer.render_environment_frame(
                env, frames_path, step_counter, mission=mission
            )

        step_counter += 1

    # Flush telemetry and save node state snapshot
    telemetry.save_node_state(mission)
    telemetry.close()

    # ---------------------------------------------------------
    # Aggregate Results
    # ---------------------------------------------------------
    results = {
        "run_id": run_id,
        "seed": active_seed,
        "steps": step_counter,
        "replans": temporal.replan_count,
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

    # ---- IEEE-aligned Comprehensive MetricsDashboard ----
    # Bug Fix: total_collected_mbits = all payload offloaded + any remaining in-flight payload
    total_data_collected = (
        getattr(mission, "total_uplinked_mbits", 0.0) + mission.collected_data_mbits
    )
    dashboard = MetricEngine.compute_full_dashboard(
        mission=mission,
        env=env,
        temporal=temporal,
        time_step=float(Config.TIME_STEP),
        collected_data_mbits=total_data_collected,
        rate_log=mission.rate_log,
    )

    results.update(dashboard)

    # ---- Stability Metrics ----
    stability = MetricEngine.compute_stability_metrics(results)
    results.update(stability)

    # ---- Semantic Evaluations ----
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
    # Gate heavy artifacts behind render flag (batch runs skip this)
    # ---------------------------------------------------------
    if render:
        visuals_path = run_manager.get_path("plots")

        # Legacy plots
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

        # ---- v0.5 IEEE-Quality Post-Run Plots ----
        PlotRenderer.render_radar_chart(results, visuals_path)

        PlotRenderer.render_node_energy_heatmap(
            nodes=env.nodes[1:],
            env_width=env.width,
            env_height=env.height,
            save_dir=visuals_path
        )

        PlotRenderer.render_trajectory_summary(
            env=env,
            visited_ids=mission.visited,
            save_dir=visuals_path
        )

        PlotRenderer.render_dashboard_panel(
            results=results,
            battery_hist=mission.battery_history,
            visited_hist=mission.visited_history,
            save_dir=visuals_path
        )

        PlotRenderer.render_3d_trajectory(env=env, save_dir=visuals_path)

        # ---- v0.5.2 Advanced Plots ----
        PlotRenderer.render_trajectory_heatmap(env=env, save_dir=visuals_path)

        PlotRenderer.render_aoi_timeline(
            aoi_history=mission.aoi_history,
            save_dir=visuals_path
        )

        PlotRenderer.render_battery_with_replans(
            battery_hist=mission.battery_history,
            replan_steps=mission.replan_timestamps,
            save_dir=visuals_path
        )

        # ---- MP4 Animation ----
        AnimationBuilder.build_mp4(
            frames_dir=frames_path,
            output_dir=run_manager.get_path("animations"),
            fps=4,
            max_frames=200,
        )

        # ---- IEEE Experiment Report ----
        IEEEDocLogger.generate_experiment_doc(
            results=results,
            metrics=stability,
            run_id=run_id,
            reports_dir=run_manager.get_path("reports"),
        )

        # ---- Post-Run Artifact Manifest ----
        print(f"\n{'='*55}")
        print(f"  Run Artifacts — {run_id}")
        print(f"{'='*55}")
        base = run_manager.base_path
        artifact_count = 0
        for root, dirs, files in os.walk(base):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, base)
                size_kb = os.path.getsize(fpath) / 1024
                print(f"  {rel:50s} {size_kb:8.1f} KB")
                artifact_count += 1
        print(f"{'='*55}")
        print(f"  Total: {artifact_count} artifacts in {base}")
        print(f"{'='*55}\n")

    return results
