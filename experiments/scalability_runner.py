"""
Scalability Runner
==================
Measures how simulation performance scales with respect to key
problem-size parameters: number of nodes, map dimensions, and
obstacle count.

Usage:
    python -m experiments.scalability_runner
    python -m experiments.scalability_runner --param node_count --values 20,50,100
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.config import Config
from core.batch_runner import BatchRunner


# ── Parameter Sweep Definitions ──────────────────────────────────────
# Maps parameter label → (Config attribute, list of values to sweep)
DEFAULT_SWEEPS: Dict[str, tuple] = {
    "node_count": ("NODE_COUNT",    [10, 20, 40, 60, 80, 100]),
    "map_size":   ("MAP_WIDTH",     [200, 400, 600, 800, 1000]),
}


# ── Core Runner ──────────────────────────────────────────────────────

def run_scalability(
    param: str = "node_count",
    values: List[int | float] | None = None,
    runs_per_point: int = 3,
    output_dir: str = "results/scalability",
) -> Dict[str, dict]:
    """
    Sweep a single Config parameter across *values* and record batch
    metrics at each point.

    Returns
    -------
    dict  mapping  str(value) → aggregated metric dict
    """
    if param not in DEFAULT_SWEEPS:
        raise ValueError(f"Unknown parameter '{param}'. Choose from {list(DEFAULT_SWEEPS)}")

    attr_name, default_values = DEFAULT_SWEEPS[param]
    values = values or default_values
    os.makedirs(output_dir, exist_ok=True)

    original_value = getattr(Config, attr_name)
    results: Dict[str, dict] = {}

    for v in values:
        print(f"\n{'=' * 60}")
        print(f"  SCALABILITY: {attr_name} = {v}")
        print(f"{'=' * 60}")

        setattr(Config, attr_name, v)

        # For map_size sweeps, keep height consistent
        if attr_name == "MAP_WIDTH":
            setattr(Config, "MAP_HEIGHT", v)

        aggregated = BatchRunner(runs=runs_per_point).execute()
        results[str(v)] = aggregated

    # Restore
    setattr(Config, attr_name, original_value)
    if attr_name == "MAP_WIDTH":
        setattr(Config, "MAP_HEIGHT", original_value)

    out_path = os.path.join(output_dir, f"scalability_{param}.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[ScalabilityRunner] Results saved → {out_path}")

    _generate_scalability_plots(results, param, output_dir)

    return results


def _generate_scalability_plots(results: Dict[str, dict], param: str, output_dir: str) -> None:
    """
    Generate publication-quality line plots from a completed scalability sweep.
    Saves PNG + PDF to *output_dir*/plots/.
    """
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    x_vals  = sorted(results.keys(), key=lambda k: float(k))
    x_float = [float(v) for v in x_vals]

    xlabel_map = {
        "node_count": "Number of IoT Nodes",
        "map_size":   "Map Dimension (m)",
    }
    xlabel = xlabel_map.get(param, param.replace("_", " ").title())

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 12, "axes.titlesize": 13,
        "axes.grid": True, "grid.alpha": 0.3,
        "figure.dpi": 150,
    })

    metrics_to_plot = [
        ("coverage_ratio_percent",      "Coverage Ratio (%)",
         "#1565C0", "Coverage vs " + xlabel),
        ("replan_frequency",            "Replan Frequency (replans/step)",
         "#C62828", "Replan Frequency vs " + xlabel),
        ("path_stability_index",        "Path Stability Index",
         "#2E7D32", "Path Stability vs " + xlabel),
    ]

    for metric_key, ylabel, colour, title in metrics_to_plot:
        y_mean, y_err = [], []
        for v in x_vals:
            entry = results[v].get(metric_key, {})
            y_mean.append(entry.get("mean", 0.0))
            y_err.append(entry.get("ci95", 0.0))

        if not any(y_mean):
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(x_float, y_mean, marker="o", color=colour, linewidth=1.8, markersize=5)
        ax.fill_between(
            x_float,
            [m - e for m, e in zip(y_mean, y_err)],
            [m + e for m, e in zip(y_mean, y_err)],
            alpha=0.15, color=colour,
        )
        ax.errorbar(x_float, y_mean, yerr=y_err, fmt="none",
                    ecolor=colour, elinewidth=1, capsize=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x_float)

        fname = f"scalability_{param}_{metric_key}"
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(plot_dir, f"{fname}.{ext}"),
                        format=ext, bbox_inches="tight")
        plt.close(fig)
        print(f"[ScalabilityRunner] Plot saved → {plot_dir}/{fname}.png")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="UAV Simulation Scalability Runner")
    parser.add_argument("--param", type=str, default="node_count",
                        choices=list(DEFAULT_SWEEPS.keys()))
    parser.add_argument("--values", type=str, default=None,
                        help="Comma-separated values to sweep, e.g. 20,50,100")
    parser.add_argument("--runs", type=int, default=3, help="Runs per point")
    parser.add_argument("--outdir", type=str, default="results/scalability")
    args = parser.parse_args()

    values = [int(v) for v in args.values.split(",")] if args.values else None
    run_scalability(param=args.param, values=values,
                    runs_per_point=args.runs, output_dir=args.outdir)


if __name__ == "__main__":
    main()
