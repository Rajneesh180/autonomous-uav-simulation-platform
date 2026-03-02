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

    return results


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
