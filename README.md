# Autonomous UAV Simulation Platform — Phase 1

> **Autonomous Data Collection from Ground IoT Sensor Networks using Intelligent UAV Trajectory Planning**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Phase](https://img.shields.io/badge/Phase-1%20%28Classical%29-green.svg)]()

---

## Overview

A discrete-event simulation engine for **autonomous UAV data collection** from ground IoT sensor networks. The UAV plans an energy-efficient trajectory through a 2D/3D environment, collects time-sensitive buffered data from IoT nodes using scheduled communication, avoids obstacles, and periodically returns to a base station to uplink collected payload.

**Core Capabilities:**
- Rendezvous point compression to reduce path complexity
- Genetic algorithm + PCA-GLS hybrid path optimiser
- Buffer-aware dynamic service-time data collection
- TDMA-scheduled multi-node communication
- SCA-inspired hover position refinement
- Base station uplink with data-age freshness constraints
- Agent-centric coordinate transforms (RL-ready observation space)
- 3D Gaussian obstacle height modelling
- Per-step telemetry recording + post-run analytics pipeline
- 25 auto-generated artifacts per run (plots, CSVs, GIFs, reports)

## Architecture

```
Phase 1/
├── config/                     # Parameters & feature toggles
├── core/                       # Simulation engine
│   ├── mission_controller.py       # UAV control loop & decision logic
│   ├── energy_model.py             # Propulsion physics & battery model
│   ├── environment_model.py        # Map, nodes, obstacles
│   ├── rendezvous_selector.py      # Greedy RP node compression
│   ├── communication.py            # Shannon rate, buffer fill, TX energy
│   ├── buffer_aware_manager.py     # Dynamic service-time data collection
│   ├── base_station_uplink.py      # Payload offload & data-age tracking
│   ├── agent_centric_transform.py  # Coordinate transform for RL observation
│   ├── simulation_runner.py        # Entry point & artifact pipeline
│   └── telemetry_logger.py         # Per-step CSV telemetry
├── path/                       # Path planning algorithms
│   ├── pca_gls.py                  # PCA + Guided Local Search solver
│   ├── ga_sequence_optimizer.py    # Genetic algorithm visiting sequence
│   └── hover_optimizer.py          # SCA hover position optimiser
├── metrics/                    # Performance analytics
│   └── metric_engine.py            # Mission KPIs (coverage, DR, AoI, etc.)
├── visualization/              # Rendering & plotting
│   ├── plot_renderer.py            # Frame renders + post-run IEEE plots
│   ├── interactive_dashboard.py    # Live matplotlib dashboard
│   ├── animation_builder.py        # GIF trajectory animation
│   ├── batch_plotter.py            # Batch-run comparative analysis
│   └── runs/                       # Per-run artifact storage
├── docs/                       # Documentation & reports
│   ├── auto_logger.py              # Experiment report generator
│   ├── theory/                     # Mathematical derivations
│   └── experiments/                # Auto-generated experiment logs
└── tests/                      # Unit tests
```

## Quick Start

```bash
# Install dependencies
pip install matplotlib numpy scipy pillow seaborn pandas

# Run a single simulation (headless, ~30s)
MPLBACKEND=Agg python3 main.py --mode single

# Run with live interactive dashboard (GUI)
python3 main.py --mode single --render

# Batch experiment across multiple seeds
python3 main.py --mode batch
```

## Output Structure

Every simulation run produces a self-contained artifact directory:

```
visualization/runs/<timestamp>/
├── logs/           run_summary.json, config_snapshot.json
├── telemetry/      step_telemetry.csv (per-step UAV state), node_state.csv
├── frames/         Per-step PNG keyframes (every 20th step)
├── plots/          11 PNG + 6 PDF post-run analysis figures
├── animations/     trajectory.gif (auto-stitched from frames)
└── reports/        experiment_report.md
```

**Generated Plots:** radar chart, 2×3 dashboard panel, trajectory summary, node energy heatmap, 3D trajectory (isometric/top/side views), battery discharge curve, visited progression.

## Performance Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Data Collection Rate (%) | Data collected / total available | → 100% |
| Coverage (%) | Nodes visited / total nodes | → 100% |
| Average AoI (s) | Mean peak data staleness | → 0 |
| Network Lifetime | Mean residual IoT battery fraction | → 1.0 |
| Path Stability | 1 − (replans / steps) | → 1.0 |
| Priority Satisfaction (%) | High-priority nodes serviced | → 100% |

## Research Foundation

This platform builds upon and extends concepts from recent research in UAV-assisted IoT data collection:

- Wang et al., "Deep Reinforcement Learning for UAV Data Collection" (2022)
- Donipati et al., "Buffer-Aware Scheduling for UAV IoT Networks" (2025)
- Zheng & Liu, "3D Trajectory Optimization for ISAC-UAV" (2025)
- Chen et al., "TD3 with Digital Twin for UAV IoT" (2025)

---

*Phase 1 — Classical Algorithms | BTP Major Project*
