# Feature Toggles Architecture

The Autonomous UAV Simulation Platform relies on a highly modular engine that isolates rendering, mathematical dimensions, and constraint evaluations into an overarching configuration layer. This allows rapid testing of discrete components (e.g., testing PCA-GLS routing independently from obstacle avoidance).

## 1. The `FeatureToggles` Singleton
The `config.feature_toggles.FeatureToggles` class acts as the central source of truth for all major conditional operations. 

It intercepts standard Command Line Interfaces (CLI) arguments parsed by `argparse` in `main.py` and mutates the environment variables globally before any class instances are instantiated.

### 1.1 Dimensional Mutability (`--dimensions`)
*   **2D (Default):** The simulation locks the UAV `z` coordinate to `0.0` or a fixed altitude. Path planning heuristics exclusively evaluate Euclidean distance in the XY-plane ($\sqrt{dx^2 + dy^2}$).
*   **3D Mode:** Unlocks the `z` axis. The `MissionController` transitions trajectory computation to spherical coordinates (evaluating discrete pitch angles alongside yaw). Node scattering engines inject random scalar altitudes up to 30.0 meters. The `<CommunicationEngine>` engages Rician Fading probabilities based on explicit elevation angles ($E_{\theta}$). The rendering engine (`PlotRenderer`) initiates `mpl_toolkits.mplot3d` canvases and extrudes 2D risk-zone patches into `Poly3DCollection` rectangular prisms.

### 1.2 Obstacle Physics Toggle (`--obstacles`, `--moving_obstacles`)
Separating physics models from the collision detection engine allows researchers to baseline the PCA-GLS algorithms against an empty vacuum vs. a highly constricted urban environment.

*   `--obstacles false`: Disables the obstacle bounding box collision checkers. Evaluates pure network throughput and travel efficiency without path detours.
*   `--moving_obstacles false`: Freezes obstacles in place upon initialization, allowing the Time-Constrained routing heuristics to evaluate against a static obstacle map rather than a non-stationary Markov process.

## 2. Integration with `Config` Defaults
The `FeatureToggles` layer dynamically injects values into the traditional constant-based `config.py` execution structure. This ensures legacy code referring to `Config.ENABLE_OBSTACLES` seamlessly honors the new dynamic CLI parameters without requiring expansive refactoring of the dependency tree.
