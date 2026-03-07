"""
Ablation Runner
===============
Systematically toggles individual pipeline components on/off and measures
their contribution to overall mission performance.

Usage:
    python -m experiments.ablation_runner           # all ablations
    python -m experiments.ablation_runner --factor obstacles
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config.config import Config
from config.feature_toggles import FeatureToggles
from core.batch_runner import BatchRunner
from metrics.metric_engine import MetricEngine


# ── Ablation Factor Registry ────────────────────────────────────────
# Each entry maps a readable label → (attribute_owner, attribute_name, off_value)
FACTOR_REGISTRY: Dict[str, tuple] = {
    "obstacles":            (FeatureToggles, "ENABLE_OBSTACLES", False),
    "moving_obstacles":     (FeatureToggles, "MOVING_OBSTACLES", False),
    "semantic_clustering":  (FeatureToggles, "ENABLE_SEMANTIC_CLUSTERING", False),
}


def _snapshot_toggles() -> Dict[str, object]:
    """Capture current toggle state so it can be restored after each ablation."""
    return {name: getattr(owner, attr) for name, (owner, attr, _) in FACTOR_REGISTRY.items()}


def _restore_toggles(snapshot: Dict[str, object]) -> None:
    for name, value in snapshot.items():
        owner, attr, _ = FACTOR_REGISTRY[name]
        setattr(owner, attr, value)


# ── Core Runner ─────────────────────────────────────────────────────

def run_ablation(
    factors: List[str] | None = None,
    runs_per_condition: int = 5,
    output_dir: str = "results/ablation",
) -> Dict[str, dict]:
    """
    Run ablation study for each *factor*.

    Returns
    -------
    dict  mapping  factor_name → {baseline: {...}, ablated: {...}, delta: {...}}
    """
    factors = factors or list(FACTOR_REGISTRY.keys())
    os.makedirs(output_dir, exist_ok=True)

    snapshot = _snapshot_toggles()
    results: Dict[str, dict] = {}

    # 1) Baseline
    print("\n" + "=" * 60)
    print("  BASELINE (all features enabled)")
    print("=" * 60)
    _restore_toggles(snapshot)
    baseline = BatchRunner(runs=runs_per_condition).execute()

    # 2) Ablations
    for factor in factors:
        if factor not in FACTOR_REGISTRY:
            print(f"[WARN] Unknown factor '{factor}', skipping.")
            continue

        owner, attr, off_value = FACTOR_REGISTRY[factor]

        print(f"\n{'=' * 60}")
        print(f"  ABLATION: {factor} = {off_value}")
        print(f"{'=' * 60}")

        _restore_toggles(snapshot)
        setattr(owner, attr, off_value)
        ablated = BatchRunner(runs=runs_per_condition).execute()

        delta = _compute_delta(baseline, ablated)
        results[factor] = {
            "baseline": baseline,
            "ablated": ablated,
            "delta": delta,
        }

    # Restore original state
    _restore_toggles(snapshot)

    # Persist
    out_path = os.path.join(output_dir, "ablation_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[AblationRunner] Results saved → {out_path}")

    _generate_ablation_plots(results, output_dir)

    return results


def _generate_ablation_plots(results: Dict[str, dict], output_dir: str) -> None:
    """
    Generate a grouped bar chart showing the relative (%) delta of each
    ablation factor across key performance metrics.

    For each factor, a bar is shown per metric.  Positive delta = feature
    removal *improved* the metric; negative = it *degraded* it.
    """
    if not results:
        return

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    METRICS = [
        ("coverage_ratio_percent",   "Coverage\nRatio (%)"),
        ("path_stability_index",     "Path\nStability"),
        ("replan_frequency",         "Replan\nFrequency"),
    ]

    factors     = list(results.keys())
    n_factors   = len(factors)
    n_metrics   = len(METRICS)
    bar_width   = 0.20
    x           = np.arange(n_factors)

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 12, "axes.titlesize": 13,
        "axes.grid": True, "grid.alpha": 0.3,
        "figure.dpi": 150,
    })

    fig, ax = plt.subplots(figsize=(max(7, 2.5 * n_factors), 5))

    palette = ["#1565C0", "#4CAF50", "#E53935"]

    for m_idx, (metric_key, metric_label) in enumerate(METRICS):
        deltas = []
        for factor in factors:
            d_val = results[factor].get("delta", {}).get(metric_key, 0.0)
            deltas.append(float(d_val) if d_val is not None else 0.0)

        colours = [palette[m_idx] if d >= 0 else "#FF8F00" for d in deltas]
        offset  = (m_idx - n_metrics / 2 + 0.5) * bar_width
        bars    = ax.bar(x + offset, deltas, bar_width,
                         color=colours, alpha=0.8, label=metric_label)

        for bar, val in zip(bars, deltas):
            if val != 0.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.5 if val >= 0 else -1.5),
                    f"{val:+.1f}%",
                    ha="center", va="bottom", fontsize=8,
                )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in factors], fontsize=10)
    ax.set_ylabel("Relative Δ vs Baseline (%)")
    ax.set_title("Ablation Study: Feature Contribution (% Change)",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(plot_dir, f"ablation_delta_bar.{ext}"),
                    format=ext, bbox_inches="tight")
    plt.close(fig)
    print(f"[AblationRunner] Delta bar chart saved → {plot_dir}/ablation_delta_bar.png")


def _compute_delta(baseline: dict, ablated: dict) -> dict:
    """Compute metric-wise *relative change* between baseline and ablated."""
    delta = {}
    for key in baseline:
        b_mean = baseline[key]["mean"]
        a_mean = ablated[key]["mean"]
        if b_mean != 0:
            delta[key] = round((a_mean - b_mean) / abs(b_mean) * 100, 2)
        else:
            delta[key] = round(a_mean * 100, 2)
    return delta


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="UAV Simulation Ablation Runner")
    parser.add_argument("--factor", type=str, help="Single factor to ablate")
    parser.add_argument("--runs", type=int, default=5, help="Runs per condition")
    parser.add_argument("--outdir", type=str, default="results/ablation")
    args = parser.parse_args()

    factors = [args.factor] if args.factor else None
    run_ablation(factors=factors, runs_per_condition=args.runs, output_dir=args.outdir)


if __name__ == "__main__":
    main()
