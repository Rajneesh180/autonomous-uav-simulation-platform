import json
import os
import statistics
from math import sqrt

from config.config import Config
from core.simulation_runner import run_simulation
from metrics.metric_engine import MetricEngine


class BatchRunner:
    """
    Executes multiple independent simulation runs and aggregates
    stability metrics statistically.
    """

    def __init__(self, runs=10):
        self.runs = runs
        self.results = []
        self.metrics = []

    # ---------------------------------------------------------
    # Core Execution
    # ---------------------------------------------------------

    def execute(self):
        for i in range(self.runs):
            print(f"\n=== Batch Run {i+1}/{self.runs} ===")

            run_result = run_simulation(
                verbose=False,
                render=False,
                seed_override=Config.RANDOM_SEED + i,
            )
            stability = MetricEngine.compute_stability_metrics(run_result)

            self.results.append(run_result)
            self.metrics.append(stability)

        return self._aggregate()

    # ---------------------------------------------------------
    # Statistical Aggregation
    # ---------------------------------------------------------

    def _aggregate(self):
        aggregated = {}

        metric_keys = self.metrics[0].keys()

        for key in metric_keys:
            values = [m[key] for m in self.metrics]

            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0

            # 95% confidence interval (normal approx)
            ci95 = 1.96 * (std / sqrt(len(values))) if len(values) > 1 else 0.0

            aggregated[key] = {
                "mean": round(mean, 6),
                "std": round(std, 6),
                "ci95": round(ci95, 6),
            }

        return aggregated

    # ---------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------

    def save(self, aggregated_metrics, output_path="results/aggregated"):
        os.makedirs(output_path, exist_ok=True)

        filepath = os.path.join(output_path, "batch_summary.json")

        with open(filepath, "w") as f:
            json.dump(aggregated_metrics, f, indent=4)

        print(f"\nBatch summary saved → {filepath}")
