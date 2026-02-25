"""
IEEE Experiment Report Generator
=================================
Automatically generates a comprehensive IEEE-formatted Markdown document
for every simulation run, capturing all parameters, gap-specific metrics,
feature toggle states, and embedding post-run visualisations.

Output: docs/experiments/experiment_<run_id>.md
        AND visualization/runs/<run_id>/reports/experiment_report.md (copy)
"""

import os
import datetime
import json
from config.config import Config
from config.feature_toggles import FeatureToggles


class IEEEDocLogger:
    """
    Generates publication-quality Markdown experiment documentation
    aligned with IEEE Transactions formatting conventions.
    """

    @staticmethod
    def generate_experiment_doc(results: dict, metrics: dict, run_id: str,
                                 reports_dir: str = None) -> str:
        """
        Generate the full experiment report.

        Parameters
        ----------
        results     : merged results dict from compute_full_dashboard
        metrics     : stability metrics dict
        run_id      : unique run identifier (timestamp)
        reports_dir : optional path to run's reports/ dir for a local copy

        Returns
        -------
        str : path to the generated report
        """
        docs_dir = os.path.join("docs", "experiments")
        os.makedirs(docs_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plots_rel = f"../../visualization/runs/{run_id}/plots"

        doc = f"""# Simulation Experiment Report — {run_id}

> **Generated:** {timestamp}
> **Seed:** {results.get('seed', 'N/A')} | **Steps:** {results.get('steps', 0)} | **Dimensions:** {FeatureToggles.DIMENSIONS}

---

## 1. Feature Toggle Configuration

| Feature | Status |
|---------|--------|
| Obstacle Collisions | `{FeatureToggles.ENABLE_OBSTACLES}` |
| Moving Obstacles | `{FeatureToggles.MOVING_OBSTACLES}` |
| Rendezvous Point Selection (Gap 1) | `{Config.ENABLE_RENDEZVOUS_SELECTION}` |
| Node Energy Depletion (Gap 2) | `{Config.ENABLE_NODE_ENERGY_DRAIN}` |
| Multi-Trial Sensing (Gap 3) | `{Config.ENABLE_PROBABILISTIC_SENSING}` |
| GA Sequence Optimizer (Gap 4) | `{Config.ENABLE_GA_SEQUENCE}` |
| 3D Gaussian Obstacles (Gap 5) | `True` |
| Agent-Centric Transform (Gap 6) | `True` |
| TDMA Scheduling (Gap 7) | `{Config.ENABLE_TDMA_SCHEDULING}` |
| BS Uplink Model (Gap 10) | `{Config.ENABLE_BS_UPLINK_MODEL}` |
| SCA Hover Optimizer (Gap 9) | `{Config.ENABLE_SCA_HOVER}` |
| Semantic Clustering | `{Config.ENABLE_SEMANTIC_CLUSTERING}` |

## 2. Simulation Parameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Max Time Steps | $T_{{max}}$ | {Config.MAX_TIME_STEPS} |
| UAV Step Size | $v$ | {Config.UAV_STEP_SIZE} m/step |
| Time Step | $\\Delta t$ | {Config.TIME_STEP} s |
| Battery Capacity | $E_{{max}}$ | {Config.BATTERY_CAPACITY} J |
| Carrier Frequency | $f_c$ | {Config.CARRIER_FREQUENCY} Hz |
| Bandwidth | $B$ | {Config.BANDWIDTH} Hz |
| Node Count | $N$ | {Config.NODE_COUNT} |
| Map Dimensions | — | {Config.MAP_WIDTH} × {Config.MAP_HEIGHT} m |
| RP Coverage Radius | $R_{{max}}$ | {Config.RP_COVERAGE_RADIUS} m |
| RP Obstacle Buffer | — | {Config.RP_OBSTACLE_BUFFER} m |
| GA Population | — | {Config.GA_POPULATION_SIZE} |
| GA Generations | — | {Config.GA_MAX_GENERATIONS} |
| BS Data-Age Limit | $T_{{limit}}$ | {Config.BS_DATA_AGE_LIMIT} steps |
| Node TX Battery | $E_{{node}}$ | {Config.NODE_BATTERY_CAPACITY_J} J |

## 3. IEEE-Aligned Performance Metrics

### 3.1 Core KPIs

| Metric | Symbol | Value |
|--------|--------|-------|
| Mission Success | SR | `{results.get('mission_success', False)}` |
| Data Collection Rate | DR | {results.get('data_collection_rate_percent', 0):.4f}% |
| Coverage Ratio | CR | {results.get('coverage_ratio_percent', 0):.2f}% |
| Collision Rate | — | {results.get('collision_rate', 0):.6f} |
| Mission Time | $T_{{total}}$ | {results.get('mission_completion_time_s', 0):.1f} s |
| Average AoI | $\\bar{{A}}$ | {results.get('average_aoi_s', 0):.4f} s |
| Avg Achievable Rate | $\\bar{{R}}_c$ | {results.get('average_achievable_rate_mbps', 0):.6f} Mbps |
| Network Lifetime | $L_{{net}}$ | {results.get('network_lifetime_residual', 0):.6f} |

### 3.2 Efficiency & Coverage

| Metric | Value |
|--------|-------|
| Nodes Visited | {results.get('nodes_visited', 0)} / {results.get('total_nodes', 0)} |
| Total Data Collected | {results.get('total_collected_mbits', 0):.4f} Mbits |
| Total Available Data | {results.get('total_available_mbits', 0):.4f} Mbits |
| Energy Consumed | {results.get('energy_consumed_total_J', 0):.4f} J |
| Energy per Node | {results.get('energy_per_node_J', 0):.2f} J/node |
| Final Battery | {results.get('final_battery_J', 0):.2f} J |
| Deadline Violations | {results.get('deadline_violations', 0)} |

### 3.3 Stability Metrics

| Metric | Value | Ideal |
|--------|-------|-------|
| Replan Frequency | {metrics.get('replan_frequency', 0):.6f} | → 0 |
| Path Stability Index | {metrics.get('path_stability_index', 0):.6f} | → 1 |
| Adaptation Latency | {metrics.get('adaptation_latency', 0):.4f} steps | → 0 |
| Node Churn Impact | {metrics.get('node_churn_impact', 0):.6f} | → 0 |
| Energy Prediction Error | {metrics.get('energy_prediction_error', 0):.6f} | → 0 |

"""

        # Semantic metrics section
        if "priority_satisfaction_percent" in results:
            doc += f"""### 3.4 Semantic Intelligence

| Metric | Value |
|--------|-------|
| Priority Satisfaction | {results.get('priority_satisfaction_percent', 0)}% |
| Semantic Purity Index | {results.get('semantic_purity_index', 0)} |

"""

        # Embedded visualisations
        doc += f"""## 4. Generated Visualisations

| Plot | Description |
|------|-------------|
| ![Radar Chart]({plots_rel}/radar_chart.png) | Mission Performance Radar |
| ![Dashboard]({plots_rel}/dashboard_panel.png) | 2×3 IEEE Dashboard Panel |
| ![Trajectory]({plots_rel}/trajectory_summary.png) | UAV Trajectory Summary |
| ![Energy Map]({plots_rel}/node_energy_heatmap.png) | IoT Node Residual Energy |
| ![3D View]({plots_rel}/trajectory_3d_isometric.png) | 3D Isometric Trajectory |

## 5. Reproducibility

- **Run ID:** `{run_id}`
- **Seed:** `{results.get('seed', 'N/A')}`
- **Config Snapshot:** `visualization/runs/{run_id}/logs/config_snapshot.json`
- **Telemetry CSV:** `visualization/runs/{run_id}/telemetry/step_telemetry.csv`
- **Animation:** `visualization/runs/{run_id}/animations/trajectory.gif`

---
*Auto-generated by IEEEDocLogger — Autonomous UAV Simulation Platform v0.5*
"""

        # Write to docs/experiments/
        filename = os.path.join(docs_dir, f"experiment_{run_id}.md")
        with open(filename, "w") as f:
            f.write(doc)

        # Copy to run's reports/ directory if available
        if reports_dir:
            os.makedirs(reports_dir, exist_ok=True)
            report_copy = os.path.join(reports_dir, "experiment_report.md")
            with open(report_copy, "w") as f:
                f.write(doc)

        print(f"[IEEE Docs] Generated experiment report → {filename}")
        return filename
