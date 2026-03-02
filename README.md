# Autonomous UAV Simulation Platform

> **Autonomous Data Collection from Ground IoT Sensor Networks using Intelligent UAV Trajectory Planning**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/Tests-73%20passing-brightgreen.svg)]()
[![Phase](https://img.shields.io/badge/Phase-1%20%28Classical%29-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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
├── main.py                         # CLI entry point (single / batch)
├── pyproject.toml                  # Project metadata, deps, tool config
├── Makefile                        # Dev workflow shortcuts
├── config/
│   ├── config.py                       # Simulation parameters
│   └── feature_toggles.py             # Runtime feature flags
├── core/                           # Simulation engine
│   ├── mission_controller.py           # UAV control loop & decision logic
│   ├── simulation_runner.py            # Run orchestration & artifact pipeline
│   ├── batch_runner.py                 # Multi-seed statistical runner
│   ├── temporal_engine.py              # Time-step advancement
│   ├── rendezvous_selector.py          # Greedy RP node compression
│   ├── dataset_generator.py            # Node layout generators
│   ├── seed_manager.py                 # Reproducible RNG control
│   ├── run_manager.py                  # Artifact directory management
│   ├── stability_monitor.py            # Online stability tracking
│   ├── telemetry_logger.py             # Per-step CSV telemetry
│   ├── models/                     # Domain models
│   │   ├── energy_model.py                 # Propulsion physics & battery
│   │   ├── environment_model.py            # Map, nodes, obstacles
│   │   ├── node_model.py                   # IoT node representation
│   │   ├── obstacle_model.py               # 3D Gaussian obstacle model
│   │   └── risk_zone_model.py              # Dynamic risk zones
│   ├── comms/                      # Communication subsystem
│   │   ├── communication.py                # Shannon rate, TX energy
│   │   ├── buffer_aware_manager.py         # Dynamic service-time collection
│   │   └── base_station_uplink.py          # Payload offload & data-age
│   ├── sensing/                    # Observation & mapping
│   │   ├── agent_centric_transform.py      # RL-ready coordinate transform
│   │   └── digital_twin_map.py             # Digital twin state mirror
│   └── clustering/                 # Semantic intelligence
│       ├── cluster_manager.py              # Cluster lifecycle manager
│       ├── semantic_clusterer.py           # Multi-feature clustering
│       └── feature_scaler.py               # Feature normalization
├── path/                           # Path planning algorithms
│   ├── pca_gls_router.py              # PCA + Guided Local Search solver
│   ├── ga_sequence_optimizer.py        # Genetic algorithm visit sequence
│   └── hover_optimizer.py              # SCA hover position optimiser
├── metrics/                        # Performance analytics
│   ├── metric_engine.py                # Mission KPIs (coverage, DR, AoI)
│   ├── auto_logger.py                  # IEEE experiment report generator
│   └── latex_exporter.py               # LaTeX table export
├── visualization/                  # Rendering & plotting
│   ├── plot_renderer.py                # Frame renders + IEEE plots
│   ├── interactive_dashboard.py        # Live matplotlib dashboard
│   ├── animation_builder.py            # GIF trajectory animation
│   ├── batch_plotter.py                # Batch comparative analysis
│   └── runs/                           # Per-run artifact storage
├── experiments/                    # Experiment harnesses
│   ├── ablation_runner.py              # Feature ablation studies
│   └── scalability_runner.py           # Node-count scaling tests
├── results/                        # Simulation outputs
│   ├── aggregated/                     # Cross-run statistical summaries
│   └── runs/                           # Per-run raw data (latest 10)
├── tests/                          # Pytest test suite (73 tests)
│   ├── conftest.py                     # Shared fixtures
│   ├── test_energy_model.py
│   ├── test_node_model.py
│   ├── test_communication.py
│   ├── test_clustering.py
│   └── test_path_planning.py
├── docs/                           # Documentation
│   ├── architecture/                   # System design docs
│   ├── theory/                         # Mathematical derivations
│   ├── phases/                         # Phase summaries & notes
│   ├── experiments/                    # Auto-generated experiment logs
│   ├── changelog/                      # Upgrade roadmaps & changelogs
│   └── roadmap/                        # Release notes & phase progression
└── scripts/
    └── research/                       # Archived exploratory scripts
```

## Quick Start

```bash
# Clone & setup
git clone https://github.com/Rajneesh180/autonomous-uav-simulation-platform.git
cd autonomous-uav-simulation-platform

# Install dependencies
pip install -r requirements.txt

# Run a single simulation (headless)
make run

# Run with live GUI dashboard
make run-gui

# 10-seed batch experiment
make batch

# Run the test suite
make test
```

## Output Structure

Every simulation run produces a self-contained artifact directory:

```
visualization/runs/<timestamp>/
├── logs/           run_summary.json, config_snapshot.json
├── telemetry/      step_telemetry.csv, node_state.csv
├── frames/         Per-step PNG keyframes
├── plots/          11 PNG + 6 PDF analysis figures
├── animations/     trajectory.gif
└── reports/        experiment_report.md
```

## Performance Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Data Collection Rate (%) | Data collected / total available | → 100% |
| Coverage (%) | Nodes visited / total nodes | → 100% |
| Average AoI (s) | Mean peak data staleness | → 0 |
| Network Lifetime | Mean residual IoT battery fraction | → 1.0 |
| Path Stability | 1 − (replans / steps) | → 1.0 |
| Priority Satisfaction (%) | High-priority nodes serviced | → 100% |

## Development

```bash
make install-dev   # Install dev tooling (pytest, ruff)
make test          # Run 73 unit tests
make lint          # Static analysis
make clean         # Remove caches
```

## Research Foundation

This platform builds upon and extends concepts from recent research in UAV-assisted IoT data collection:

- Wang et al., "Deep Reinforcement Learning for UAV Data Collection" (2022)
- Donipati et al., "Buffer-Aware Scheduling for UAV IoT Networks" (2025)
- Zheng & Liu, "3D Trajectory Optimization for ISAC-UAV" (2025)
- Chen et al., "TD3 with Digital Twin for UAV IoT" (2025)

---

*Phase 1 — Classical Algorithms | BTP Major Project*
