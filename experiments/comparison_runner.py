"""
Algorithm Comparison Framework
===============================
Runs the proposed DST-BA algorithm against 4 baseline strategies on
an identical environment (same seed, same nodes, same obstacles) and
produces a head-to-head comparison table + grouped bar chart.

Baselines implemented:
  1. Nearest-Neighbour (NN) — greedy closest-node-first
  2. Random-Walk (RW) — visit nodes in random permutation order
  3. Fixed-Sweep — systematic lawnmower sweep (left→right, bottom→top)
  4. Cluster-First Route-Second (CFRS) — KMeans + intra-cluster NN

Usage:
    python -m experiments.comparison_runner [--seed 42]
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np

from config.config import Config
from config.feature_toggles import FeatureToggles
from core.seed_manager import set_global_seed
from core.models.environment_model import Environment
from core.models.node_model import UAVState, SensorNode
from core.dataset_generator import generate_nodes
from core.models.obstacle_model import Obstacle
from core.models.risk_zone_model import RiskZone
from core.models.energy_model import EnergyModel
from core.temporal_engine import TemporalEngine
from core.run_manager import RunManager
from core.mission_controller import MissionController
from visualization.plot_renderer import PlotRenderer


# ═════════════════════════════════════════════════════════════
#  Helper: build a fresh environment from a seed
# ═════════════════════════════════════════════════════════════
def _build_env(seed: int):
    """Create a deterministic environment identical across algorithms."""
    set_global_seed(seed)
    Config.apply_hostility_profile()

    env = Environment(Config.MAP_WIDTH, Config.MAP_HEIGHT)
    env.dataset_mode = Config.DATASET_MODE

    temporal = TemporalEngine(Config.TIME_STEP, Config.MAX_TIME_STEPS)
    env.temporal_engine = temporal

    nodes = generate_nodes(
        Config.DATASET_MODE,
        Config.NODE_COUNT,
        Config.MAP_WIDTH,
        Config.MAP_HEIGHT,
        seed,
    )
    for node in nodes:
        env.add_node(node)

    if Config.ENABLE_OBSTACLES:
        _rng = random.Random(seed + Config.OBSTACLE_SEED_OFFSET)
        margin = 60
        for _ in range(Config.OBSTACLE_COUNT):
            x1 = _rng.randint(margin, Config.MAP_WIDTH - margin - 120)
            y1 = _rng.randint(margin, Config.MAP_HEIGHT - margin - 100)
            w  = _rng.randint(80, 160)
            h  = _rng.randint(80, 140)
            env.add_obstacle(Obstacle(x1, y1, x1 + w, y1 + h))

    if Config.ENABLE_RISK_ZONES:
        env.add_risk_zone(RiskZone(100, 400, 300, 550, multiplier=1.8))

    return env, temporal


# ═════════════════════════════════════════════════════════════
#  Shared lightweight simulation loop for baselines
# ═════════════════════════════════════════════════════════════
def _run_baseline(env, temporal, visit_order, label):
    """
    Fly the UAV along *visit_order* (list of SensorNode),
    collecting data at each stop.  Includes realistic hover energy
    for data collection (buffer_drain_time * hover_power) so the
    energy comparison with DST-BA is fair.

    Returns a results dict comparable to DST-BA output.
    """
    uav = env.uav
    center = (env.width // 2, env.height // 2)
    safe = env.get_safe_start(center)
    uav.x, uav.y = safe
    uav.z = getattr(Config, "UAV_ALTITUDE", 50.0)
    uav.current_battery = Config.BATTERY_CAPACITY

    step_size = getattr(Config, "UAV_STEP_SIZE", 10.0)
    max_steps = Config.MAX_TIME_STEPS
    dt = float(Config.TIME_STEP)

    # Pre-compute hover power (v=0) used during data collection
    hover_power = EnergyModel.propulsion_power(0.0)

    visited = set()
    energy_total = 0.0
    collected_data = 0.0
    step = 0
    battery_hist = []
    visited_hist = []
    queue = list(visit_order)

    # Pre-fill node buffers (in DST-BA, data accumulates over time;
    # for baseline fairness, assume all data is ready for collection)
    for node in env.sensors:
        node.current_buffer = node.buffer_capacity

    while step < max_steps and uav.current_battery > 0 and queue:
        target = queue[0]

        # Move towards target
        dx = target.x - uav.x
        dy = target.y - uav.y
        dz = (getattr(target, "z", 0)) - uav.z
        dist = math.sqrt(dx**2 + dy**2 + dz**2)

        if dist > step_size:
            ratio = step_size / dist
            uav.x += dx * ratio
            uav.y += dy * ratio
            uav.z += dz * ratio
        else:
            uav.x, uav.y = target.x, target.y

        # Record UAV trail
        if not hasattr(env, "uav_trail"):
            env.uav_trail = []
        env.uav_trail.append((uav.x, uav.y, uav.z))

        # Energy consumption — propulsion for this step
        fly_energy = EnergyModel.energy_for_distance(uav, min(dist, step_size))
        uav.current_battery -= fly_energy
        energy_total += fly_energy

        # Check if arrived at node
        arrive_dist = math.sqrt(
            (uav.x - target.x)**2 + (uav.y - target.y)**2
        )
        if arrive_dist < 15.0 and target.id not in visited:
            # Hover to drain buffer — realistic collection energy
            # time = buffer / data_rate;  energy = hover_power * time
            data_rate = getattr(target, "data_generation_rate", 1.0)
            drain_time = target.current_buffer / max(data_rate, 0.1)
            hover_e = hover_power * drain_time
            uav.current_battery -= hover_e
            energy_total += hover_e

            collected_data += target.current_buffer
            target.current_buffer = 0.0
            visited.add(target.id)
            queue.pop(0)

        battery_hist.append(uav.current_battery)
        visited_hist.append(len(visited))
        step += 1

    total_nodes = len(env.sensors)
    coverage = (len(visited) / total_nodes * 100) if total_nodes > 0 else 0
    epn = energy_total / max(len(visited), 1)
    throughput = collected_data / max(energy_total, 1e-6)  # Mbits per Joule

    return {
        "algorithm": label,
        "nodes_visited": len(visited),
        "total_nodes": total_nodes,
        "coverage_ratio_percent": round(coverage, 2),
        "energy_consumed_J": round(energy_total, 2),
        "energy_per_node_J": round(epn, 2),
        "data_collected_mbits": round(collected_data, 2),
        "throughput_mbits_per_J": round(throughput, 6),
        "data_collection_rate_percent": round(
            collected_data / max(sum(n.buffer_capacity for n in env.sensors), 1e-6) * 100, 2
        ),
        "steps": step,
        "battery_history": battery_hist,
        "visited_history": visited_hist,
    }


# ═════════════════════════════════════════════════════════════
#  Baseline Strategies
# ═════════════════════════════════════════════════════════════

def _nearest_neighbour_order(nodes, start_x, start_y):
    """Greedy nearest-neighbour tour."""
    remaining = list(nodes)
    order = []
    cx, cy = start_x, start_y
    while remaining:
        best = min(remaining, key=lambda n: (n.x - cx)**2 + (n.y - cy)**2)
        order.append(best)
        cx, cy = best.x, best.y
        remaining.remove(best)
    return order


def _random_walk_order(nodes, seed):
    """Random permutation of nodes."""
    rng = random.Random(seed)
    order = list(nodes)
    rng.shuffle(order)
    return order


def _sweep_order(nodes):
    """Lawnmower sweep: sort by Y then alternating X direction."""
    # Divide into horizontal bands
    ys = [n.y for n in nodes]
    band_height = (max(ys) - min(ys)) / max(int(len(nodes)**0.5), 2)
    bands = {}
    for n in nodes:
        band_idx = int((n.y - min(ys)) / max(band_height, 1))
        bands.setdefault(band_idx, []).append(n)

    order = []
    for i, band_idx in enumerate(sorted(bands.keys())):
        band = sorted(bands[band_idx], key=lambda n: n.x,
                       reverse=(i % 2 == 1))
        order.extend(band)
    return order


def _cluster_first_order(nodes, start_x, start_y, seed, k=4):
    """KMeans clustering → intra-cluster NN → inter-cluster NN."""
    from sklearn.cluster import KMeans

    if len(nodes) < k:
        return _nearest_neighbour_order(nodes, start_x, start_y)

    coords = np.array([[n.x, n.y] for n in nodes])
    km = KMeans(n_clusters=k, random_state=seed, n_init=5).fit(coords)
    labels = km.labels_
    centroids = km.cluster_centers_

    # Order clusters by NN from start
    cx, cy = start_x, start_y
    cluster_order = []
    remaining_clusters = list(range(k))
    while remaining_clusters:
        best_c = min(remaining_clusters,
                     key=lambda c: (centroids[c][0] - cx)**2 +
                                   (centroids[c][1] - cy)**2)
        cluster_order.append(best_c)
        cx, cy = centroids[best_c]
        remaining_clusters.remove(best_c)

    # Within each cluster, NN order
    order = []
    px, py = start_x, start_y
    for c in cluster_order:
        cluster_nodes = [nodes[i] for i in range(len(nodes)) if labels[i] == c]
        sub_order = _nearest_neighbour_order(cluster_nodes, px, py)
        if sub_order:
            order.extend(sub_order)
            px, py = sub_order[-1].x, sub_order[-1].y
    return order


# ═════════════════════════════════════════════════════════════
#  Run DST-BA (our proposed approach) through normal pipeline
# ═════════════════════════════════════════════════════════════

def _run_dst_ba(seed):
    """Run the full DST-BA simulation and return comparable results."""
    from core.simulation_runner import run_simulation
    results = run_simulation(verbose=False, render=False, seed_override=seed)
    # Normalise keys to match baseline format
    results["algorithm"] = "DST-BA (Proposed)"
    results["energy_consumed_J"] = results.get(
        "energy_consumed_total_J",
        results.get("energy_consumed_J", 0),
    )
    if "energy_per_node_J" not in results:
        results["energy_per_node_J"] = (
            results["energy_consumed_J"] / max(results.get("nodes_visited", 1), 1)
        )
    # Data throughput
    collected = results.get("total_collected_mbits", 0)
    results["data_collected_mbits"] = round(collected, 2)
    results["throughput_mbits_per_J"] = round(
        collected / max(results["energy_consumed_J"], 1e-6), 6
    )
    return results


# ═════════════════════════════════════════════════════════════
#  Comparison Orchestrator
# ═════════════════════════════════════════════════════════════

def run_comparison(seed=42, save_dir=None):
    """Run all algorithms on an identical environment, return results."""
    print(f"\n{'='*60}")
    print(f"  ALGORITHM COMPARISON — Seed {seed}")
    print(f"{'='*60}\n")

    all_results = []

    # 1. DST-BA (proposed)
    print("[1/5] Running DST-BA (Proposed)...")
    dst_ba = _run_dst_ba(seed)
    all_results.append(dst_ba)
    print(f"       → Visited: {dst_ba.get('nodes_visited')}/{dst_ba.get('total_nodes')}  "
          f"Steps: {dst_ba.get('steps')}")

    # 2-5: Baselines
    baselines = [
        ("Nearest-Neighbour", _nearest_neighbour_order),
        ("Random-Walk", None),
        ("Fixed-Sweep", _sweep_order),
        ("Cluster-First", None),
    ]

    for idx, (label, order_fn) in enumerate(baselines, start=2):
        print(f"[{idx}/5] Running {label}...")
        env, temporal = _build_env(seed)
        nodes = list(env.sensors)

        # Initialise UAV start position (same logic as simulation_runner)
        center = (env.width // 2, env.height // 2)
        safe = env.get_safe_start(center)
        env.uav.x, env.uav.y = safe
        sx, sy = safe

        if label == "Nearest-Neighbour":
            order = _nearest_neighbour_order(nodes, sx, sy)
        elif label == "Random-Walk":
            order = _random_walk_order(nodes, seed)
        elif label == "Fixed-Sweep":
            order = _sweep_order(nodes)
        elif label == "Cluster-First":
            order = _cluster_first_order(nodes, sx, sy, seed)
        else:
            order = nodes

        result = _run_baseline(env, temporal, order, label)
        all_results.append(result)
        print(f"       → Visited: {result['nodes_visited']}/{result['total_nodes']}  "
              f"Steps: {result['steps']}")

    # ── Generate Comparison Artefacts ──
    if save_dir is None:
        rm = RunManager()
        save_dir = rm.get_path("plots")

    os.makedirs(save_dir, exist_ok=True)

    # Table
    _print_comparison_table(all_results)
    _save_comparison_table(all_results, save_dir)

    # Bar chart
    _render_comparison_chart(all_results, save_dir)

    # per-metric bar charts
    _render_per_metric_charts(all_results, save_dir)

    print(f"\n  Comparison artefacts saved to: {save_dir}\n")
    return all_results


# ═════════════════════════════════════════════════════════════
#  Output Formatters
# ═════════════════════════════════════════════════════════════

def _print_comparison_table(results):
    """Console table."""
    header = (f"{'Algorithm':<22} {'Visited':>8} {'Cov%':>6} "
              f"{'Energy(J)':>10} {'E/Node':>8} {'Data(Mb)':>9} "
              f"{'Mb/J':>8} {'Steps':>6}")
    print(f"\n  {header}")
    print(f"  {'─' * len(header)}")
    for r in results:
        print(f"  {r['algorithm']:<22} "
              f"{r.get('nodes_visited',0):>8} "
              f"{r.get('coverage_ratio_percent',0):>6.1f} "
              f"{r.get('energy_consumed_J',0):>10.0f} "
              f"{r.get('energy_per_node_J',0):>8.1f} "
              f"{r.get('data_collected_mbits',0):>9.1f} "
              f"{r.get('throughput_mbits_per_J',0):>8.4f} "
              f"{r.get('steps',0):>6}")
    print()


def _save_comparison_table(results, save_dir):
    """Save as JSON + LaTeX."""
    clean = []
    for r in results:
        row = {k: v for k, v in r.items()
               if k not in ("battery_history", "visited_history",
                            "event_timestamps", "replan_timestamps",
                            "rate_log")}
        clean.append(row)
    with open(os.path.join(save_dir, "comparison_table.json"), "w") as f:
        json.dump(clean, f, indent=2, default=str)

    # LaTeX table for thesis
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Performance Comparison of UAV Path Planning Algorithms}",
        r"\label{tab:algorithm-comparison}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Algorithm} & \textbf{Coverage (\%)} & \textbf{Energy (kJ)} & "
        r"\textbf{E/Node (kJ)} & \textbf{Data (Mb)} & \textbf{Steps} \\",
        r"\midrule",
    ]

    best_energy = min(r["energy_consumed_J"] for r in results)
    best_epn = min(r["energy_per_node_J"] for r in results)

    for r in results:
        algo = r["algorithm"].replace("_", r"\_")
        cov = r.get("coverage_ratio_percent", 0)
        energy_kj = r.get("energy_consumed_J", 0) / 1000
        epn_kj = r.get("energy_per_node_J", 0) / 1000
        data = r.get("data_collected_mbits", 0)
        steps = r.get("steps", 0)

        # Bold the best values
        e_str = f"\\textbf{{{energy_kj:.1f}}}" if r["energy_consumed_J"] == best_energy else f"{energy_kj:.1f}"
        epn_str = f"\\textbf{{{epn_kj:.1f}}}" if r["energy_per_node_J"] == best_epn else f"{epn_kj:.1f}"

        lines.append(
            f"  {algo} & {cov:.1f} & {e_str} & {epn_str} & {data:.1f} & {steps} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(os.path.join(save_dir, "comparison_table.tex"), "w") as f:
        f.write("\n".join(lines))


def _render_comparison_chart(results, save_dir):
    """Grouped bar chart of key metrics — all algorithms side-by-side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PlotRenderer._set_style()

    metrics = [
        ("Coverage (%)", "coverage_ratio_percent"),
        ("Nodes Visited", "nodes_visited"),
        ("Energy/Node (J)", "energy_per_node_J"),
        ("Data Collected (Mb)", "data_collected_mbits"),
        ("Steps", "steps"),
    ]
    algorithms = [r["algorithm"] for r in results]
    n_alg = len(algorithms)
    n_met = len(metrics)
    x = np.arange(n_met)
    width = 0.8 / n_alg

    colours = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A", "#C62828"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, r in enumerate(results):
        vals = [r.get(m[1], 0) for m in metrics]
        offset = (i - n_alg / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=r["algorithm"],
                       color=colours[i % len(colours)], alpha=0.88,
                       edgecolor="black", linewidth=0.3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=10)
    ax.set_title("Algorithm Comparison — Key Performance Metrics",
                  fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.2)

    plt.tight_layout()
    PlotRenderer._save_dual(fig, save_dir, "algorithm_comparison")


def _render_per_metric_charts(results, save_dir):
    """Individual bar chart per metric for thesis figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PlotRenderer._set_style()

    per_metric = [
        ("Coverage Ratio (%)", "coverage_ratio_percent", "#1565C0"),
        ("Nodes Visited", "nodes_visited", "#2E7D32"),
        ("Energy per Node (J)", "energy_per_node_J", "#E65100"),
        ("Total Energy (J)", "energy_consumed_J", "#C62828"),
        ("Data Collected (Mbits)", "data_collected_mbits", "#00838F"),
        ("Throughput (Mbits/J)", "throughput_mbits_per_J", "#4527A0"),
        ("Steps to Completion", "steps", "#6A1B9A"),
    ]

    algorithms = [r["algorithm"] for r in results]
    x = np.arange(len(algorithms))

    for title, key, colour in per_metric:
        vals = [r.get(key, 0) for r in results]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(x, vals, width=0.5, color=colour, alpha=0.85,
                       edgecolor="black", linewidth=0.4)

        # Highlight best (max for coverage/visited, min for energy/steps)
        if "energy" in key.lower() or "steps" in key.lower():
            best_idx = int(np.argmin(vals)) if any(v > 0 for v in vals) else 0
        else:
            best_idx = int(np.argmax(vals))
        bars[best_idx].set_edgecolor("#1B5E20")
        bars[best_idx].set_linewidth(2.5)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, fontsize=8, rotation=15, ha="right")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(title.split("(")[0].strip())

        plt.tight_layout()
        safe_name = key.replace("_", "-")
        PlotRenderer._save_dual(fig, save_dir, f"compare_{safe_name}")


# ═════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algorithm Comparison")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--preset", choices=["simple", "full"],
                        default="simple")
    args = parser.parse_args()

    Config.PRESET = args.preset
    Config.apply_preset()
    FeatureToggles.RENDER_MODE = "3D"

    run_comparison(seed=args.seed)
