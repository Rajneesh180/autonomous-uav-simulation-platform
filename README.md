# Autonomous UAV Simulation Platform — Phase 1

> **IEEE-Aligned IoT Data Collection via UAV with Rendezvous Points, Genetic Algorithm, TDMA, and SCA Hover Optimization**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Phase](https://img.shields.io/badge/Phase-1%20%28Classical%29-green.svg)]()
[![License](https://img.shields.io/badge/License-Academic-orange.svg)]()

---

## Overview

A discrete-event simulation platform for **autonomous UAV data collection** from ground IoT sensor networks. The UAV navigates a 2D/3D environment, collects buffered data from IoT nodes via TDMA-scheduled communication, avoids obstacles, and periodically uplinks collected payload to a base station.

The platform implements **10 core algorithmic gaps** identified from 4 IEEE reference papers (2022–2025), covering rendezvous point selection, genetic algorithm path optimization, probabilistic sensing, SCA hover optimization, and base station uplink with data-age constraints.

## Architecture

```
Phase 1/
├── config/                 # Config.py + FeatureToggles
├── core/                   # Simulation engine
│   ├── mission_controller.py   # Central UAV control loop
│   ├── energy_model.py         # Propulsion + battery physics
│   ├── environment_model.py    # Map, nodes, obstacles
│   ├── obstacle_model.py       # Rectangular + Gaussian obstacles
│   ├── rendezvous_selector.py  # Gap 1: RP node compression
│   ├── communication.py        # Shannon rate, buffer fill, TX energy
│   ├── buffer_aware_manager.py # DST-BA data collection
│   ├── base_station_uplink.py  # Gap 10: BS uplink + AoI constraint
│   ├── agent_centric_transform.py  # Gap 6: coord transform for RL
│   ├── simulation_runner.py    # Entry point for simulation
│   ├── telemetry_logger.py     # Per-step CSV telemetry
│   └── clustering/             # Semantic node clustering
├── path/                   # Path planning algorithms
│   ├── pca_gls.py              # PCA + Guided Local Search solver
│   ├── ga_sequence_optimizer.py # Gap 4: GA visiting sequence
│   └── hover_optimizer.py      # Gap 9: SCA hover position
├── metrics/                # IEEE metrics dashboard
│   └── metric_engine.py        # SR, DR, CR, AoI, network lifetime
├── visualization/          # Rendering & plotting
│   ├── plot_renderer.py        # 2D/3D frame renders + IEEE plots
│   ├── interactive_dashboard.py # Live matplotlib dashboard
│   ├── animation_builder.py    # GIF trajectory animation
│   ├── batch_plotter.py        # Batch-run comparative plots
│   └── runs/                   # Per-run artifact directories
├── docs/                   # Documentation & experiment reports
│   ├── auto_logger.py          # IEEE experiment report generator
│   ├── theory/                 # Mathematical derivations per gap
│   └── experiments/            # Auto-generated experiment MDs
└── tests/                  # Unit tests
```

## Quick Start

### Prerequisites
```bash
pip install matplotlib numpy scipy pillow seaborn pandas
```

### Run a Single Simulation
```bash
# Headless (no per-frame PNG rendering — fast ~30s)
MPLBACKEND=Agg python3 main.py --mode single

# With live interactive dashboard (GUI)
python3 main.py --mode single --render

# With per-frame PNG export + GIF animation
python3 main.py --mode single --render
```

### Run a Batch Experiment
```bash
python3 main.py --mode batch
```

### CLI Options
| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `single` | `single` or `batch` |
| `--render` | `False` | Enable per-frame PNG export + GIF |
| `--dimensions` | `2D` | `2D` or `3D` environment |

## Output Structure

Every simulation run generates a self-contained artifact directory:

```
visualization/runs/<YYYY-MM-DD_HH-MM-SS>/
├── logs/
│   ├── run_summary.json          # All IEEE metrics (JSON)
│   └── config_snapshot.json      # Frozen config for reproducibility
├── telemetry/
│   ├── step_telemetry.csv        # Per-step UAV state (13 columns)
│   └── node_state.csv            # End-of-mission node snapshot
├── plots/
│   ├── radar_chart.png/pdf       # 6-KPI spider chart
│   ├── dashboard_panel.png/pdf   # 2×3 multi-panel IEEE figure
│   ├── trajectory_summary.png/pdf # Final trajectory map
│   ├── node_energy_heatmap.png/pdf # IoT residual battery map
│   ├── trajectory_3d_*.png/pdf   # 3D views (isometric, top, side)
│   ├── battery_over_time.png     # Battery discharge curve
│   ├── visited_over_time.png     # Node visit progression
│   └── ...                       # 25 total artifacts
├── animations/
│   └── trajectory.gif            # Animated flight path (if --render)
└── reports/
    └── experiment_report.md      # IEEE experiment documentation
```

## Implemented IEEE Gaps

| # | Feature | Paper | Module |
|---|---------|-------|--------|
| 1 | Rendezvous Point Selection | Donipati et al. (TNSM 2025) | `rendezvous_selector.py` |
| 2 | IoT Node TX Energy + Network Lifetime | Donipati et al. (TNSM 2025) | `communication.py` |
| 3 | Multi-Trial Probabilistic Sensing | Wang et al. (IoT 2022) | `buffer_aware_manager.py` |
| 4 | GA Visiting Sequence Optimizer | Chen et al. (IoT 2025) | `ga_sequence_optimizer.py` |
| 5 | 3D Gaussian Obstacle Height Model | Zheng & Liu (TVT 2025) | `obstacle_model.py` |
| 6 | Agent-Centric Coordinate Transform | Chen et al. (IoT 2025) | `agent_centric_transform.py` |
| 7 | TDMA Single-Node Scheduling | Donipati et al. (TNSM 2025) | `buffer_aware_manager.py` |
| 8 | Comprehensive Metrics Dashboard | Wang / Chen / Donipati | `metric_engine.py` |
| 9 | SCA-Inspired Hover Optimizer | Zheng & Liu (TVT 2025) | `hover_optimizer.py` |
| 10 | BS Uplink + Data-Age Constraint | Zheng & Liu (TVT 2025) | `base_station_uplink.py` |

## Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| DR (%) | Data collected / total available | → 100% |
| Coverage (%) | Nodes visited / total nodes | → 100% |
| Avg AoI (s) | Mean peak data staleness | → 0 |
| Network Lifetime | Mean residual IoT battery | → 1.0 |
| Path Stability | 1 − (replans/steps) | → 1.0 |
| Priority Satisfaction (%) | High-priority nodes serviced | → 100% |

## References

1. **Wang et al.** — "Deep Reinforcement Learning for UAV Data Collection" (IEEE IoT Journal, 2022)
2. **Donipati et al.** — "DST-BA: Buffer-Aware Scheduling for UAV IoT" (IEEE TNSM, 2025)
3. **Zheng & Liu** — "3D Trajectory Optimization for ISAC-UAV" (IEEE TVT, 2025)
4. **Chen et al.** — "TD3+ISAC+Digital Twin for UAV IoT" (IEEE IoT Journal, 2025)

---

*Autonomous UAV Simulation Platform — Phase 1 Classical Algorithms — BTP Major Project*
