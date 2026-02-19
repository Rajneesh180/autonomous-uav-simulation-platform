import os
import json

from core.simulation_runner import run_simulation
from metrics.metric_engine import MetricEngine
from core.batch_runner import BatchRunner


def run_single():
    results = run_simulation(verbose=True)
    metrics = MetricEngine.compute_stability_metrics(results)

    print("\nMission execution completed.")
    print(f"Run ID: {results['run_id']}")
    print(f"Total simulation steps executed: {results['steps']}")
    print(f"Final battery: {round(results['final_battery'], 2)}")
    print(f"Visited nodes: {results['visited']}")
    print(f"Replan count: {results['replans']}")
    print(f"Collision count: {results['collisions']}")
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
    print("=== Autonomous UAV Simulation Platform ===")

    MODE = "batch"  # change to "batch" when needed

    if MODE == "single":
        run_single()
    elif MODE == "batch":
        run_batch()
    else:
        raise ValueError("Invalid MODE. Use 'single' or 'batch'.")


if __name__ == "__main__":
    main()
