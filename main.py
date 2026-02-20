import os
import json
import argparse

from core.simulation_runner import run_simulation
from metrics.metric_engine import MetricEngine
from core.batch_runner import BatchRunner


def run_single(render: bool = False):
    results = run_simulation(verbose=True, render=render)
    metrics = MetricEngine.compute_stability_metrics(results)

    print("\nMission execution completed.")
    print(f"Run ID: {results['run_id']}")
    print(f"Total simulation steps executed: {results['steps']}")
    print(f"Final battery: {round(results['final_battery'], 2)}")
    print(f"path_stability_index: {round(metrics['path_stability_index'], 4)}")
    print(f"node_churn_impact: {round(metrics['node_churn_impact'], 4)}")
    print(f"energy_prediction_error: {round(metrics['energy_prediction_error'], 4)}")
    print(f"Visited nodes: {results['visited']}")
    print(f"Replan count: {results['replans']}")
    print(f"Collision count: {results['collisions']}")
    
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
        "results",
        "runs",
        results["run_id"],
        "logs",
        "stability_metrics.json",
    )

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


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
    parser.add_argument(
        "--render", 
        action="store_true", 
        help="Enable Matplotlib/Pygame telemetry visualizer."
    )
    
    args = parser.parse_args()

    print("=== Autonomous UAV Simulation Platform ===")

    if args.mode == "single":
        run_single(render=args.render)
    elif args.mode == "batch":
        print(f"[Warning] GUI rendering is implicitly disabled during high-speed batch executions.")
        run_batch()

if __name__ == "__main__":
    main()
