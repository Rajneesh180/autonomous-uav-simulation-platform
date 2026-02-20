import json
import os
import statistics
from math import sqrt

import pandas as pd

from config.config import Config
from core.simulation_runner import run_simulation
from metrics.metric_engine import MetricEngine
from visualization.batch_plotter import BatchPlotter


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
        # Execute runs
        for i in range(self.runs):
            print(f"\n=== Batch Run {i+1}/{self.runs} ===")

            run_result = run_simulation(
                verbose=False,
                render=False,
                seed_override=Config.RANDOM_SEED + i,
            )
            stability = MetricEngine.compute_stability_metrics(run_result)
            
            # Combine run variables for flat DataFrame representation
            combined = run_result.copy()
            combined.update(stability)
            combined['mode'] = Config.DATASET_MODE
            combined['hostility_level'] = Config.HOSTILITY_LEVEL

            self.results.append(combined)
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
        
        # Phase 4: DataFrame Plotting (IEEE Stats)
        df = pd.DataFrame(self.results)
        plot_dir = os.path.join(output_path, "plots")
        
        print("\n[BatchRunner] Generating standard statistical figures...")
        BatchPlotter.render_stability_boxplots(df, plot_dir)
        BatchPlotter.render_semantic_correlation_heatmap(df, plot_dir)
        BatchPlotter.render_efficiency_pareto(df, plot_dir)
        print(f"[BatchRunner] Batch graphics finalized in: {plot_dir}")
