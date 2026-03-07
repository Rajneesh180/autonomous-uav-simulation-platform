# Phase-4 Completion Report
**UAV Swarm Simulation Platform — IoT-Based Environment with Semantic Clustering**

---

## 1. Overview

Phase-4 finalises the core autonomous-planning pipeline for UAV-assisted IoT data collection in dynamic environments. All components developed in earlier phases have been integrated, hardened, and validated through scalability sweeps and ablation studies. No reinforcement learning elements are included; the system operates entirely on classical planning algorithms with reactive adaptation.

---

## 2. Architecture Summary

The simulation follows a four-stage pre-flight planning pipeline followed by a discrete-step execution loop with online replanning.

```
IoT Environment (node positions, priorities, AoI, obstacles)
        │
        ▼
┌────────────────────────┐
│  RendezvousSelector    │  Cluster nodes → RP compression points
└────────────┬───────────┘
             │  rendezvous_points, member_map
             ▼
┌────────────────────────┐
│  PCAGLSRouter          │  Compute Euclidean travelling-salesman order
└────────────┬───────────┘
             │  initial_route
             ▼
┌────────────────────────┐
│  GASequenceOptimizer   │  Refine visiting sequence (energy + time windows)
└────────────┬───────────┘
             │  optimised_sequence
             ▼
┌────────────────────────┐
│  HoverOptimizer        │  Set hover altitude & dwell time per waypoint
└────────────┬───────────┘
             │
             ▼
     Discrete-step execution (MissionController)
       ├── DigitalTwinMap  (motion scoring, obstacle risk)
       ├── CommunicationModel (buffer, data-rate estimation)
       └── StabilityMonitor (replan trigger, hysteresis)
```

---

## 3. Modules Implemented / Extended

### 3.1 Configuration (`config/config.py`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `OBSTACLE_COUNT` | 3 | Number of seeded obstacles generated per run |
| `OBSTACLE_SEED_OFFSET` | 7 | RNG offset added to `RANDOM_SEED` for obstacle placement |
| `GA_ENERGY_WEIGHT` | 0.003 | Weight of the altitude-penalty term in GA fitness |

### 3.2 Simulation Runner (`core/simulation_runner.py`)

- **Seeded obstacle placement**: Hardcoded obstacle positions replaced with a seeded loop that generates `Config.OBSTACLE_COUNT` obstacles using `random.Random(seed + Config.OBSTACLE_SEED_OFFSET)`. This makes experiments reproducible while varying obstacle layout across random-seed sweeps.
- **Phase-4 visualisation calls**: Six new `PlotRenderer` methods are invoked after each simulation run when `render=True` is active.

### 3.3 GA Sequence Optimizer (`path/ga_sequence_optimizer.py`)

The fitness function now incorporates a 3-D altitude-aware energy cost term:

$$f(\sigma) = \frac{1}{1 + d_\text{horiz}(\sigma) + w_\text{tw} \cdot v(\sigma) + w_e \cdot \sum_i |z_i - z_{i-1}|}$$

where:
- $d_\text{horiz}$ — total horizontal travel distance along the route
- $v(\sigma)$ — total time-window violation (seconds past deadline)
- $w_\text{tw}$ — time-window penalty weight (`GA_TW_PENALTY`)
- $w_e$ — energy weight (`GA_ENERGY_WEIGHT = 0.003`)
- $|z_i - z_{i-1}|$ — absolute vertical displacement between consecutive waypoints

This penalises unnecessary altitude changes, producing flatter mission profiles and reducing energy expenditure.

### 3.4 Mission Controller (`core/mission_controller.py`)

**Urgency-stratified inter-cluster routing**  
The cluster selection logic was replaced with a priority-and-recency score:

$$\text{score}(c) = \frac{\bar{p}_c \cdot \left(1 + \dfrac{\overline{\text{AoI}}_c}{\text{MAX\_AOI\_LIMIT}}\right)}{1 + \|\mathbf{u} - \mathbf{c}\|}$$

where $\bar{p}_c$ is the mean node priority in cluster $c$, $\overline{\text{AoI}}_c$ is the mean AoI timer, $\mathbf{u}$ is the current UAV position, and $\mathbf{c}$ is the cluster centroid. The UAV consistently navigates to the most urgent, data-stale cluster first.

**New per-step telemetry attributes**
- `aoi_mean_history` — mean AoI (seconds) across all active nodes at each simulation step
- `collected_data_history` — cumulative data collected (Mbits) at each simulation step

### 3.5 Scalability Runner (`experiments/scalability_runner.py`)

- **Bug fixed**: Config attribute swept was `"NUM_NODES"` (non-existent); corrected to `"NODE_COUNT"`.
- **Publication plots**: After each sweep, `_generate_scalability_plots()` generates line charts with 95% CI shading for:
  - Coverage Ratio (%) vs sweep parameter
  - Replan Frequency vs sweep parameter
  - Path Stability Index vs sweep parameter
  - Saved as both PNG (300 dpi) and PDF to `results/experiments/scalability/plots/`

### 3.6 Ablation Runner (`experiments/ablation_runner.py`)

- **Delta bar chart**: `_generate_ablation_plots()` generates a grouped bar chart showing the relative % change (Δ) in each metric when each pipeline component is disabled. Bars are coloured by metric; negative deltas (component removal degraded performance) are highlighted in amber. Saved to `results/ablation/plots/ablation_delta_bar.png|pdf`.

---

## 4. Visualisation Outputs

### Existing methods (pre-Phase-4)

| Method | Output |
|--------|--------|
| `render_map()` | Static environment + UAV trail |
| `render_trajectory()` | 3-D trajectory with altitude colour mapping |
| `render_aoi_timeline()` | Per-node AoI heat timeline |
| `render_battery_with_replans()` | Battery level annotated with replan events |
| `render_speed_heatmap()` | Speed vs position density |
| `render_dashboard_panel()` | 4-panel summary dashboard |
| `render_run_comparison()` | Multi-run metric comparison |

### New Phase-4 methods

| Method | Description |
|--------|-------------|
| `render_semantic_clustering()` | Geographic overlay of cluster assignments with convex-hull boundaries and centroid star markers |
| `render_clustering_pca_space()` | 2-D PCA scatter plot of node feature embeddings coloured by cluster label |
| `render_routing_pipeline()` | 3-panel figure: raw nodes → RP-compressed graph → GA-optimised visiting sequence |
| `render_rendezvous_compression()` | Before/after two-panel showing RP compression ratio annotated on the figure |
| `render_communication_quality()` | Data-rate vs distance with 1/d² reference + buffer occupancy histogram |
| `render_mission_progress_combined()` | 4-panel 2×2 grid (visited nodes, battery, collected data, mean AoI) sharing a step x-axis with replan event markers |

---

## 5. Experiment Capabilities

### Scalability Study

```bash
python -m experiments.scalability_runner --param node_count --values 20 40 60 80 100
```

Sweeps any Config numeric attribute over specified values, running N seeds per value. Exports JSON statistics and publication-quality line plots.

### Ablation Study

```bash
python -m experiments.ablation_runner --factor semantic_clustering
python -m experiments.ablation_runner          # all three factors
```

Three ablation factors registered:

| Factor | Toggle disabled | Measures contribution of |
|--------|----------------|--------------------------|
| `obstacles` | `FeatureToggles.ENABLE_OBSTACLES = False` | Static obstacle avoidance |
| `moving_obstacles` | `FeatureToggles.MOVING_OBSTACLES = False` | Dynamic obstacle handling |
| `semantic_clustering` | `FeatureToggles.ENABLE_SEMANTIC_CLUSTERING = False` | Cluster-guided routing |

---

## 6. Results Directory Structure

```
results/
├── aggregated/              # BatchRunner aggregate JSON files
├── runs/                    # Per-run SimulationResult JSON + plots
│   └── <run_id>/
│       ├── metrics.json
│       └── plots/
├── experiments/
│   └── scalability/
│       ├── scalability_node_count.json
│       └── plots/
│           ├── scalability_node_count_coverage_ratio_percent.{png,pdf}
│           ├── scalability_node_count_replan_frequency.{png,pdf}
│           └── scalability_node_count_path_stability_index.{png,pdf}
└── ablation/
    ├── ablation_results.json
    └── plots/
        └── ablation_delta_bar.{png,pdf}
```

---

## 7. Known Limitations

1. **Single UAV**: The platform models one UAV. Multi-agent coordination is deferred to a future phase.
2. **Simplified communication model**: Data rate uses an inverse-distance heuristic; fading, interference, and multi-path effects are not modelled.
3. **Ground-truth obstacle planning**: `_recompute_plan()` uses `env.obstacles` directly. The digital twin is used for motion scoring and risk assessment in `_move_one_step()`, but obstacle-avoidance routing still relies on ground truth.
4. **2-D hover optimisation**: The `HoverOptimizer` optimises dwell time; altitude is set by a fixed policy (`Config.UAV_ALTITUDE`) rather than per-waypoint optimisation.
5. **No RL, no MARL**: All policies are deterministic or evolutionary. No trained policy networks are included in Phase-4.

---

## 8. Dependency Summary

All dependencies are listed in `requirements.txt`. No new external libraries were introduced in Phase-4 beyond those already present.

Key runtime dependencies:
- `numpy`, `scipy` — numerical and spatial operations
- `matplotlib` — all plotting (Agg backend for headless operation)
- `scikit-learn` — semantic clustering, PCA feature reduction

---

*Report generated at Phase-4 completion checkpoint.*
