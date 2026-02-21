import os
import datetime
import json
from config.config import Config
from config.feature_toggles import FeatureToggles


class IEEEDocLogger:
    """
    Automatically generates IEEE-formatted Markdown documentation for every simulation run,
    detailing the exact parameters, physics models, and algorithms active during the experiment.
    """

    @staticmethod
    def generate_experiment_doc(results, metrics, run_id):
        docs_dir = os.path.join("docs", "experiments")
        os.makedirs(docs_dir, exist_ok=True)
        
        filename = os.path.join(docs_dir, f"experiment_{run_id}.md")
        
        doc_content = f"""# Simulation Experiment Track: {run_id}
> **Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This document automatically logs the environmental constants, feature structures, and final algorithmic outputs of a single simulation experiment to ensure perfect reproducibility for IEEE research papers.

## 1. Architectural Configuration
The following core subsystems were active during this execution:
- **Spatial Dimensions:** {FeatureToggles.DIMENSIONS}
- **Obstacle Collisions Enabled:** {FeatureToggles.ENABLE_OBSTACLES}
- **Moving Obstacles (Temporal Markhov):** {FeatureToggles.MOVING_OBSTACLES}
- **Visual Rendering Export:** {FeatureToggles.ENABLE_VISUALS}

## 2. Theoretical Parameters 
The mathematical backbone of the simulation engine was initialized with the following values:
- **Max Time Steps ($T_{{max}}$):** {Config.MAX_TIME_STEPS}
- **UAV Velocity Vector Scale ($v$):** {Config.UAV_STEP_SIZE} units/step
- **Prediction Horizon (Lookahead):** {Config.PREDICTION_HORIZON} steps
- **Carrier Frequency ($f_c$):** {Config.CARRIER_FREQUENCY} Hz
- **Node Initial Data Rate:** {Config.DEFAULT_DATA_RATE_MBPS} Mbps
- **Battery Capacity Bound:** {Config.BATTERY_CAPACITY} J

## 3. Heuristic Results (PCA-GLS)
Final execution trace of the Path Cheapest Arc (PCA) + Guided Local Search algorithms.
- **Total Steps Executed:** {results.get('steps', 0)}
- **Nodes Visited Successfully:** {results.get('visited', 'N/A')}
- **Replans Triggered by Volatility:** {results.get('replans', 0)}
- **Hard Collisions Avoided/Encountered:** {results.get('collisions', 0)}
- **Unsafe Returns (Energy Deficit):** {results.get('unsafe_return', 0)}
- **Remaining Energy Yield:** {results.get('final_battery', 0.0):.2f} J

## 4. Stability & Semantic Metrics
- **Path Stability Index:** {metrics.get('path_stability_index', 0.0):.4f} (Ideal: 1.0)
- **Node Churn Impact:** {metrics.get('node_churn_impact', 0.0):.4f} (Lower = Less Volatile)
- **Energy Prediction Error:** {metrics.get('energy_prediction_error', 0.0):.4f} 
"""

        if "priority_satisfaction_percent" in results:
            doc_content += f"""
### Phase 4 Semantic Values
- **Semantic Purity Index:** {results.get('semantic_purity_index', 0.0)}
- **Priority Satisfaction Rate:** {results.get('priority_satisfaction_percent', 0.0)}%
"""

        with open(filename, 'w') as f:
            f.write(doc_content)
            
        print(f"[IEEE Docs] Automatically generated experiment report at {filename}")
