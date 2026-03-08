import argparse
import json
import os

from config.config import Config
from config.feature_toggles import FeatureToggles
from core.batch_runner import BatchRunner
from core.simulation_runner import run_simulation
from metrics.auto_logger import IEEEDocLogger
from metrics.metric_engine import MetricEngine


def run_single(render: bool = True):
    results = run_simulation(verbose=True, render=render)
    metrics = MetricEngine.compute_stability_metrics(results)

    print("\nMission execution completed.")
    print(f"Run ID: {results['run_id']}")
    print(f"Total simulation steps executed: {results['steps']}")
    print(f"Final battery: {round(results['final_battery_J'], 2)}")
    print(f"path_stability_index: {round(metrics['path_stability_index'], 4)}")
    print(f"node_churn_impact: {round(metrics['node_churn_impact'], 4)}")
    print(f"energy_prediction_error: {round(metrics['energy_prediction_error'], 4)}")
    print(f"Visited nodes: {results['nodes_visited']}")
    print(f"Replan count: {results['replans']}")
    print(f"Collision count: {results['collision_rate']}")
    
    if "priority_satisfaction_percent" in results:
        print("\n--- Semantic Intelligence Metrics ---")
        print(f"Priority Satisfaction: {results['priority_satisfaction_percent']}%")
        print(f"Semantic Purity Index: {results['semantic_purity_index']}")
    print(f"Unsafe return count: {results['unsafe_return']}")

    print("\n--- Stability Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {round(v, 4)}")

    # Persist per-run stability metrics
    metrics_path = os.path.join(
        "visualization",
        "runs",
        results["run_id"],
        "logs",
        "stability_metrics.json",
    )

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Generate automated IEEE report
    IEEEDocLogger.generate_experiment_doc(results, metrics, results["run_id"])


def run_batch():
    runner = BatchRunner(runs=10)
    aggregated = runner.execute()

    runner.save(aggregated)

    print("\n--- Batch Aggregated Metrics ---")
    for metric, stats in aggregated.items():
        print(metric, stats)


def main():
    parser = argparse.ArgumentParser(description="Autonomous UAV Simulation Platform")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["single", "batch"], 
        default="single",
        help="Execution mode: 'single' for a detailed run, 'batch' for statistical aggregation."
    )
    parser.add_argument("--preset", type=str, choices=["simple", "full"], default=None,
                        help="Apply a parameter preset: 'simple' (fast demo) or 'full' (research).")
    parser.add_argument(
        "--render", 
        action="store_true", 
        help="Enable Matplotlib/Pygame telemetry visualizer."
    )
    parser.add_argument(
        "--render-mode", 
        type=str, 
        choices=["2D", "3D", "both"], 
        default="2D",
        help="Visualisation projection: '2D' top-down, '3D' perspective, or 'both'."
    )
    # --- Toggle flags (true/false strings) ---
    _toggle = lambda name, default, hlp: parser.add_argument(
        f"--{name}", type=str, choices=["true","false","True","False"], default=default, help=hlp)
    _toggle("obstacles", "true", "Toggle obstacle presence.")
    _toggle("moving_obstacles", "false", "Toggle obstacle animation.")
    _toggle("risk_zones", None, "Toggle risk zones.")
    _toggle("energy", None, "Toggle UAV energy model.")
    _toggle("bs_uplink", None, "Toggle base-station uplink model.")
    _toggle("tdma", None, "Toggle TDMA scheduling.")
    _toggle("ga", None, "Toggle GA sequence optimizer.")
    _toggle("clustering", None, "Toggle semantic clustering.")
    _toggle("rendezvous", None, "Toggle rendezvous-point selection.")
    _toggle("sca", None, "Toggle SCA hover optimizer.")
    _toggle("sensing", None, "Toggle probabilistic sensing.")
    _toggle("dynamic_nodes", None, "Toggle dynamic node join/leave.")
    _toggle("predictive_avoidance", None, "Toggle predictive avoidance.")
    
    args = parser.parse_args()

    # 1. Apply preset (if given via CLI or use Config default)
    if args.preset:
        Config.PRESET = args.preset
    Config.apply_preset()

    # 2. Apply CLI toggle overrides on top of preset
    FeatureToggles.apply_overrides(args)
    FeatureToggles.sync_to_config()

    # 3. Hostility + validation
    Config.apply_hostility_profile()
    Config.validate()

    print("=== Autonomous UAV Simulation Platform ===")
    print(f"    Preset: {Config.PRESET}  |  Nodes: {Config.NODE_COUNT}  |  Steps: {Config.MAX_TIME_STEPS}")

    if args.mode == "single":
        run_single(render=True)
        if args.render or True: # Render is currently hardcoded to True in run_single above
            print("\n[Dashboard] Simulation complete! The dynamic dashboard is still active.")
            input("Press [ENTER] to exit the simulation framework...")
    elif args.mode == "batch":
        print(f"[Warning] GUI rendering is implicitly disabled during high-speed batch executions.")
        run_batch()

if __name__ == "__main__":
    main()
