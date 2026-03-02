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

from config.config import Config
from core.batch_runner import BatchRunner


# ── Parameter Sweep Definitions ──────────────────────────────────────
# Maps parameter label → (Config attribute, list of values to sweep)
DEFAULT_SWEEPS: Dict[str, tuple] = {
    "node_count": ("NUM_NODES",     [10, 20, 40, 60, 80, 100]),
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

    return results


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
