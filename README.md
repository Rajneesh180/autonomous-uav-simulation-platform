<div align="center">
  
# 🛸 Autonomous UAV Simulation & Data Collection Platform
  
**A Research-Grade, Deterministic, 3D Time-Constrained Trajectory Planning & Dynamic Service Time Simulator.**

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://opensource.org/licenses/GPL-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

## 📌 Abstract
This platform realistically simulates the flight, navigation, and decision-making capabilities of Unmanned Aerial Vehicles (UAVs) in highly dynamic 3D IoT environments. It implements **Buffer-Aware Data Collection (DST-BA)** and **Path Cheapest Arc with Guided Local Search (PCA-GLS)** algorithms to handle complex routing constraints.

The system incorporates rigorous aerodynamics, Rician Fading Channel probabilistic constraints, and predictive obstacle avoidance designed for evaluating high-end routing meta-heuristics and multi-agent systems.

## 🔬 Core Architecture & Mathematical Foundations

### 1. 3D Kinematics and Propulsion Energy Model
- **Rotary-Wing Propulsion Power Model:** Incorporates mass, moment of inertia, and rotor disc area to generate precise power requirements over varying linear and angular velocities.
- **Continuous Trajectory Discretization:** Processes spatial obstacles modeled with continuous elevation functions.

### 2. Time-Constrained Dynamic Service Time & Buffer Optimization
- **Dual-State Servicing:** The UAV dynamically transitions between continuous flight ("chord-flying") over nodes and stationary hovering depending on instantaneous IoT node buffer volume and Shannon-rated transmission latency.
- **Rician Fading:** Calculates probabilistic Line-of-Sight (LoS) links dynamically modified by elevation angles and scattering profiles.

### 3. Anticipatory Route Optimization
- **Meta-Heuristics:** Employs Guided Local Search algorithms incorporating strict time-window boundaries, ensuring global optimality for tardiness and probability-of-success metrics instead of myopic greedy routing.

## 🛠 System Architecture

The simulation decouples physics, routing, and communication layers to allow seamless integration of advanced ML controllers and metric aggregation:
- `core/kinematics.py`: Governs continuous 3D motion restrictions.
- `core/communication.py`: Assesses fading margins and deterministic buffer depletion.
- `path/trajectory_optimizer.py`: Handles exact Time Window bounds using PCA-GLS.
- `core/mission_controller.py`: Centralized orchestrator dispatching adaptation triggers.

## 📈 Visual Artifacts & Persistence

Simulation telemetry is completely deterministic and preserved to generate reproducible heatmaps, trajectory reconstructions, and stability metrics.

*(Heatmaps, temporal volatility plots, and layout diagrams are dynamically synthesized and injected into the `experiments/runs/` workspace directory upon completion of execution cycles.)*
