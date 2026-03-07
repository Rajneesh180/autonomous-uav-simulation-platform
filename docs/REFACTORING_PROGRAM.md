# Structured Refactoring Program
## UAV-Based IoT Data Collection Simulator — Phase 1 → Phase 5 Bridge

**Document Type**: Principal Architect → Implementation Engineer Handoff  
**Implementation Target**: Claude Sonnet  
**Date**: 7 March 2026  
**Repository**: `Rajneesh180/autonomous-uav-simulation-platform` (branch: `main`)

---

## Table of Contents

1. [Task 1 — Critical Fix Order](#task-1--critical-fix-order)
2. [Task 2 — Stage Definitions](#task-2--stage-definitions)
3. [Task 3 — Verification Procedures](#task-3--verification-procedures)
4. [Task 4 — Git Workflow](#task-4--git-workflow)
5. [Task 5 — Sonnet Implementation Prompts](#task-5--sonnet-implementation-prompts)
6. [Task 6 — Final Simulator Architecture](#task-6--final-simulator-architecture)
7. [Task 7 — Phase-5 RL Foundation](#task-7--phase-5-rl-foundation)
8. [Task 8 — Research Platform Requirements](#task-8--research-platform-requirements)

---

# Task 1 — Critical Fix Order

## 1.1 Minimum Defect Set for Phase-5 RL Readiness

The following defects from the architectural audit **must** be corrected before any RL agent can be trained against this simulator. Each defect is mapped to its blocking consequence for reinforcement learning.

| Defect ID | Description | Why It Blocks RL |
|-----------|-------------|------------------|
| D2 | `Node` class shared by UAV and IoT sensors | Gymnasium `observation_space` cannot be defined; UAV state and sensor state are semantically conflated |
| D3 | Step-based hover loop instead of analytical service time | Reward signal is distorted; service time is an artifact of step count, not physics |
| D6 | Discrete step counter with no variable time advancement | RL episodes have inconsistent temporal semantics; cannot model variable-duration actions |
| D1 | MissionController God object (~600 lines) | Cannot isolate `reset()`, `step(action)`, observation extraction, or reward computation |
| D4 | Matplotlib rendering inside simulation loop | Training at 10,000+ episodes is impossible; each step incurs 10-100× overhead |
| D5 | 2D collision detection in a 3D simulation | Collision reward is meaningless; false positives when UAV flies above obstacles |

## 1.2 Dependency-Ordered Refactoring Plan

The stages below are ordered by **dependency** — each stage depends only on the completion of prior stages. No stage may be reordered without breaking downstream assumptions.

```
Stage 1 — Entity Model Separation (D2)
   │   Splits Node into UAVState + SensorNode + BaseStation
   │   Required by: everything downstream (all modules reference Node)
   │
Stage 2 — Visualization Decoupling (D4)
   │   Removes matplotlib from the simulation tick loop
   │   Required by: Stage 4 (clean step() interface), all batch/training runs
   │
Stage 3 — Continuous Simulation Clock (D6)
   │   Replaces TemporalEngine step counter with variable-Δt ContinuousClock
   │   Required by: Stage 4 (analytical service time needs variable advancement)
   │
Stage 4 — Analytical Service Model (D3)
   │   Replaces hover loop with τ* = D_i(t) / R_i(t) analytical formula
   │   Required by: Stage 5 (correct physics in decomposed controller)
   │
Stage 5 — MissionController Decomposition (D1)
   │   Splits God object into PhysicsEngine, ObservationBuilder, RewardFunction
   │   Required by: Stage 6 (clean Gymnasium wrapper)
   │
Stage 6 — True 3D Collision Detection (D5)
       Adds z-axis comparison to collision checks
       Required by: RL collision reward signal
```

## 1.3 Why This Order

**Stage 1 first**: Every subsequent module needs to know the difference between UAV and sensor. If we decompose the controller (Stage 5) before splitting Node, we propagate the conflation into new modules and must refactor them again.

**Stage 2 before Stage 4**: The analytical service model changes the semantics of `step()`. We cannot verify timing correctness if every step is throttled by `plt.pause(0.001)`. Decoupling visualization first ensures we can run rapid verification cycles.

**Stage 3 before Stage 4**: The analytical service time formula produces a continuous Δt value (e.g., 12.7 seconds). Without a ContinuousClock that can advance by arbitrary amounts, this value has nowhere to go — the simulator can only advance by integer steps.

**Stage 4 before Stage 5**: The MissionController decomposition (Stage 5) must extract a clean `ServiceModel`. If we decompose first and then fix the service time, we must edit the newly created module — wasting effort and creating regression risk.

**Stage 5 before Stage 6**: 3D collision detection requires understanding how the physics engine moves the UAV. The PhysicsEngine extracted in Stage 5 is the correct place for the 3D collision check.

---

# Task 2 — Stage Definitions

## Stage 1 — Entity Model Separation

### Objective

Split the monolithic `Node` dataclass into three distinct entity types that reflect the physical reality of the simulation: a UAV agent, ground-mounted IoT sensor nodes, and a base station.

### Modules Affected

| Module | Change |
|--------|--------|
| `core/models/node_model.py` | Refactored: split into `UAVState`, `SensorNode` (kept in same file for backward compat); add `BaseStation` |
| `core/mission_controller.py` | Update: `self.uav` typed as `UAVState`; all `env.nodes[1:]` patterns replaced with explicit `env.sensors` |
| `core/models/environment_model.py` | Update: add `self.sensors: List[SensorNode]`, `self.uav: UAVState`, `self.base_station: BaseStation`; remove `self.nodes` list (or keep as deprecated alias) |
| `core/simulation_runner.py` | Update: create `UAVState` and `SensorNode` separately; pass to `Environment` |
| `core/dataset_generator.py` | Update: `generate_nodes()` returns `List[SensorNode]`; separate UAV creation |
| `core/comms/buffer_aware_manager.py` | Update: parameter types from `Node` to `SensorNode` |
| `core/comms/communication.py` | Update: accepts position tuples (interface unchanged) |
| `core/comms/base_station_uplink.py` | Update: use `BaseStation` entity |
| `core/clustering/semantic_clusterer.py` | Update: operates on `List[SensorNode]` |
| `core/rendezvous_selector.py` | Update: operates on `List[SensorNode]` |
| `path/pca_gls_router.py` | Update: operates on `List[SensorNode]` |
| `path/ga_sequence_optimizer.py` | Update: operates on `List[SensorNode]` |
| `path/hover_optimizer.py` | Minimal change: uses position tuples |
| `metrics/metric_engine.py` | Update: separate UAV metrics from sensor metrics |
| `visualization/interactive_dashboard.py` | Update: render UAV and sensors separately |
| `visualization/plot_renderer.py` | Update: render UAV and sensors separately |
| `core/telemetry_logger.py` | Update: log UAVState fields and SensorNode fields separately |
| `tests/*` | Update: all test fixtures |

### Architectural Change

**Current**: One `Node` dataclass with ~30 fields. UAV is `env.nodes[0]`. Sensor fields (buffer, data_rate, aoi) and UAV fields (battery, velocity, yaw/pitch) coexist in every instance.

**After**: Three dataclasses:

```
UAVState:
    id: int
    x, y, z: float
    yaw, pitch: float
    vx, vy, vz: float
    battery_capacity: float
    current_battery: float
    payload_buffer_mbits: float  # data collected, awaiting BS uplink
    energy_per_meter: float
    hover_cost: float
    return_threshold: float
    
    def position() -> Tuple[float, float, float]
    def reset_battery()

SensorNode:
    id: int
    x, y, z: float  (z = 0 for ground sensors)
    priority: int
    risk: float
    signal_strength: float
    deadline: Optional[float]
    reliability: float
    buffer_capacity: float
    current_buffer: float
    data_generation_rate: float
    aoi_timer: float
    max_aoi_timer: float
    time_window_start: float
    time_window_end: float
    node_battery_J: float
    tx_energy_consumed_J: float
    
    def position() -> Tuple[float, float, float]

BaseStation:
    x, y, z: float
    uplink_bandwidth: float
    
    def position() -> Tuple[float, float, float]
```

**Environment** gains:
```
self.uav: UAVState
self.sensors: List[SensorNode]
self.base_station: BaseStation
```

The old `self.nodes` list is removed. All code that does `env.nodes[1:]` now uses `env.sensors`. All code that does `env.nodes[0]` now uses `env.uav`.

### Expected Behavior After Change

- `python3 main.py` completes without errors.
- All tests pass (after fixture updates).
- Telemetry CSV columns are identical (field names unchanged).
- Visualization shows UAV and sensors as before.
- Metrics JSON output is identical within floating-point tolerance.
- The UAV is no longer in any sensor list; no `[1:]` slicing exists anywhere.

---

## Stage 2 — Visualization Decoupling

### Objective

Remove all rendering calls from the simulation tick loop. Rendering must never execute during `MissionController.step()`. Instead, rendering is performed:
- Post-hoc from saved state, OR
- Via an explicit callback registered outside the simulation engine.

### Modules Affected

| Module | Change |
|--------|--------|
| `core/mission_controller.py` | Remove: `InteractiveDashboard` import, instantiation, and `render()` call inside `step()`; remove `PlotRenderer.render_environment_frame()` call inside `step()` |
| `core/simulation_runner.py` | Update: move frame rendering into the `while` loop in `run_simulation()`, gated by `render` flag; render is called **after** `mission.step()`, not inside it |
| `visualization/interactive_dashboard.py` | No change to internal logic; it is simply no longer called from within the physics tick |

### Architectural Change

**Current**: `MissionController.step()` contains:
```python
if self.render_enabled:
    self.interactive_dash.render(...)  # matplotlib plt.pause(0.001)

# ... physics ...

if self.render_enabled and ...:
    PlotRenderer.render_environment_frame(...)
```

**After**: `MissionController.step()` contains **zero** rendering calls. The `render_enabled` flag and `interactive_dash` attribute are removed from `MissionController`.

`simulation_runner.py` becomes the sole rendering orchestrator:
```
while mission.is_active():
    mission.step()
    telemetry.log_step(step_counter, mission)
    
    if render:
        # Interactive dashboard (optional, only in single-run mode)
        interactive_dash.render(mission.uav, ...)
        
        # Periodic frame saving
        if step_counter % keyframe_interval == 0:
            PlotRenderer.render_environment_frame(env, frames_path, step_counter, mission=mission)
    
    step_counter += 1
```

### Expected Behavior After Change

- `python3 main.py` still renders the interactive dashboard (render is just called from a different location).
- `python3 main.py --render` produces identical visual output.
- `python3 main.py --mode batch` runs significantly faster (no matplotlib overhead per step).
- `MissionController` has zero imports from `visualization/`.
- Frame files are still generated in `visualization/runs/<run_id>/frames/`.

---

## Stage 3 — Continuous Simulation Clock

### Objective

Replace `TemporalEngine` (integer step counter) with a `ContinuousClock` that supports variable time advancement — allowing flight segments, service durations, and sensing operations to advance the simulation clock by their exact physical durations.

### Modules Affected

| Module | Change |
|--------|--------|
| `core/temporal_engine.py` | Refactored: add `advance(dt: float)` method alongside the existing `tick()`; maintain backward compatibility by keeping `tick()` as `advance(self.time_step)` |
| `core/mission_controller.py` | Update: use `temporal.advance(dt)` where appropriate; `temporal.tick()` remains as the default 1-step advancement for now |
| `config/config.py` | Add: `TIME_STEP = 1.0` explicitly typed as float (currently implicit) |

### Architectural Change

**Current**: `TemporalEngine.tick()` does `self.current_step += self.time_step` (always integer 1). No way to advance by a fractional or larger amount.

**After**: `TemporalEngine` gains:
```
def advance(self, dt: float):
    """Advance simulation clock by an arbitrary positive amount."""
    self.current_time += dt
    self.current_step = int(self.current_time / self.time_step)  # for backward compat
    if self.current_time >= self.max_time:
        self.active = False

@property
def current_time(self) -> float:
    """Continuous simulation time in seconds."""
    return self._current_time

def tick(self):
    """Legacy: advance by one fixed time step."""
    self.advance(self.time_step)
    return self.active
```

The key addition is `current_time: float` which accumulates continuous time, decoupled from the integer `current_step`. Stage 4 will use `advance(tau_star)` to skip ahead by the analytical service time.

### Expected Behavior After Change

- `python3 main.py` produces identical output (all existing code still calls `tick()` which advances by 1.0s).
- `temporal.current_time` and `temporal.current_step` remain synchronized when only `tick()` is used.
- New `advance(dt)` method is available for Stage 4.
- All termination checks use `current_time >= max_time` instead of `current_step > max_steps`.
- Tests pass unchanged.

---

## Stage 4 — Analytical Service Model

### Objective

Replace the step-based hover loop in `MissionController._move_one_step()` with the Donipati et al. analytical service time formula: τ* = D_i(t) / R_i(t). When the UAV arrives at a node, it computes the exact service duration, advances the clock by that amount, and drains the buffer in one operation.

### Modules Affected

| Module | Change |
|--------|--------|
| `core/mission_controller.py` | Major refactor of `_move_one_step()`: remove the Center-Hover step loop; replace with single analytical service call |
| `core/comms/buffer_aware_manager.py` | Update: `execute_service()` method that computes τ*, drains buffer, advances clock, returns `ServiceOutcome` |
| `core/models/energy_model.py` | Update: `hover_energy(uav, duration_s)` must accept continuous duration, not just `dt` |
| `core/comms/communication.py` | No change (already provides `achievable_data_rate()` and `minimum_hover_time()`) |
| `config/config.py` | `MAX_HOVER_STEPS_PER_NODE` becomes `MAX_SERVICE_TIME_S = 30.0` (safety cap in seconds) |

### Architectural Change

**Current `_move_one_step()` Center-Hover logic**:
```python
if distance < 1e-3:
    # Hover in place, drain buffer one step at a time
    hover_e = EnergyModel.hover_energy(self.uav, dt)
    EnergyModel.consume(self.uav, hover_e)
    data_collected = BufferAwareManager.process_data_collection(...)
    self._hover_step_count[nid] += 1
    if buffer <= 0 or hover_count >= 30:
        mark_visited()
```

**After**:
```python
if distance < 1e-3:
    # Analytical service time (Donipati et al., TNSM 2025)
    outcome = BufferAwareManager.execute_service(
        uav=self.uav,
        node=self.current_target,
        env=self.env,
        temporal=self.temporal,
        energy_model=EnergyModel,
    )
    self.collected_data_mbits += outcome.data_collected
    self.energy_consumed_total += outcome.energy_consumed
    self.rate_log.append(outcome.achievable_rate)
    
    # Advance all other nodes' buffers and AoI by the service duration
    for sensor in self.env.sensors:
        if sensor.id != self.current_target.id and sensor.id not in self.visited:
            sensor.current_buffer = min(
                sensor.buffer_capacity,
                sensor.current_buffer + sensor.data_generation_rate * outcome.service_time_s
            )
            sensor.aoi_timer += outcome.service_time_s
    
    self.visited.add(self.current_target.id)
    self.current_target = None
```

**New `BufferAwareManager.execute_service()` method**:
```
Inputs: uav (UAVState), node (SensorNode), env, temporal (ContinuousClock), energy_model
Computation:
    R_i = CommunicationEngine.achievable_data_rate(node.position(), uav.position(), env)
    tau_star = node.current_buffer / R_i  (if R_i > 0, else abandon)
    T_sense = CommunicationEngine.minimum_hover_time(distance)
    T_total = min(tau_star + T_sense, MAX_SERVICE_TIME_S)
    E_hover = energy_model.hover_energy(uav, T_total)
    
    # Energy feasibility: reduce T_total if battery insufficient
    E_return = energy_model.energy_to_return(uav, base_station)
    if uav.current_battery < E_hover + E_return:
        T_total = affordable_duration(uav.current_battery - E_return, hover_power)
    
    data_collected = min(node.current_buffer, R_i * T_total)
    node.current_buffer -= data_collected
    uav.current_battery -= E_hover
    uav.payload_buffer_mbits += data_collected
    temporal.advance(T_total)
Outputs: ServiceOutcome(data_collected, service_time_s, energy_consumed, achievable_rate)
```

### Expected Behavior After Change

- `python3 main.py` completes. The number of simulation steps changes (potentially fewer iterations since service time is now a single clock advancement instead of 30 loop iterations).
- `coverage_ratio` may change — this is expected and correct.
- `mission_completion_time` reflects true service durations.
- `average_aoi` changes because AoI is now updated continuously.
- Telemetry CSV shows varying time gaps between entries (non-uniform Δt).
- The `MAX_HOVER_STEPS_PER_NODE` config entry and `_hover_step_count` dict are removed.

---

## Stage 5 — MissionController Decomposition

### Objective

Break the 600-line `MissionController` into focused modules with single responsibilities, preparing the interface needed for a Gymnasium `Env` wrapper.

### Modules Affected

| Module | Change |
|--------|--------|
| `core/mission_controller.py` | Major reduction: retains only orchestration logic (~150 lines). Delegates to extracted modules. |
| New: `core/physics_engine.py` | Extracted: UAV motion primitives, scoring, energy consumption, kinematic constraints |
| New: `core/observation_builder.py` | Extracted: state vector construction for RL agents (uses `AgentCentricTransform`) |
| New: `core/reward_function.py` | New: modular reward computation with tunable weights |
| `core/simulation_runner.py` | Minor update: passes new dependencies to `MissionController` constructor |

### Architectural Change

**Current**: `MissionController` owns everything — motion, scoring, energy, hovering, replanning, clustering, rendering, telemetry, terminal conditions.

**After**: `MissionController` becomes a thin orchestrator:

```
MissionController:
    __init__(env, temporal, physics, planner, service_model)
    
    step():
        1. temporal.tick()
        2. env.update_dynamics(temporal.dt)    # obstacles, risk zones, node spawn/removal
        3. if should_replan(): planner.replan()
        4. if energy_critical(): terminate()
        5. fill_buffers(temporal.dt)
        6. check_bs_uplink()
        7. physics.execute_movement(uav, target, env)  # flight + scoring
        8. if at_target(): service_model.execute_service(...)
        9. record_histories()
        10. check_terminal()
    
    is_active() → bool
```

**PhysicsEngine** (extracted from `_move_one_step()`):
```
PhysicsEngine:
    execute_movement(uav, target_pos, env, energy_model, dt):
        - Compute motion primitives (yaw/pitch offsets)
        - Score candidates (distance, energy, risk, obstacle clearance, goal progress)
        - Apply best move
        - Consume energy
        - Return MovementOutcome(new_pos, energy_consumed, collision)
    
    generate_motion_primitives(uav, target, dt) → List[Candidate]
    score_candidate(candidate, target, env) → float
```

**ObservationBuilder** (for RL — Stage 7 prerequisite):
```
ObservationBuilder:
    build(uav, target, sensors, obstacles, clock) → np.ndarray
        - Uses AgentCentricTransform.build_state_vector()
        - Appends K nearest obstacle features
        - Normalizes to [-1, 1]
```

**RewardFunction** (for RL — Stage 7 prerequisite):
```
RewardFunction:
    compute(world_state_before, action, world_state_after) → float
        - r_data: data collected this step / max_possible
        - r_energy: -energy_consumed / budget
        - r_aoi: -(sum AoI) / (N × T_max)
        - r_collision: -1.0 if collision else 0.0
        - r_deadline: -count(violations) / N
        - Weighted sum: α₁r₁ + α₂r₂ + α₃r₃ + α₄r₄ + α₅r₅
```

### Expected Behavior After Change

- `python3 main.py` produces identical output to pre-decomposition (behavioral equivalence).
- `core/mission_controller.py` is reduced from ~600 lines to ~150 lines.
- `core/physics_engine.py` contains all motion primitive logic (~200 lines).
- `core/observation_builder.py` is ready for RL integration (~50 lines).
- `core/reward_function.py` is ready for RL integration (~80 lines).
- All tests pass.
- No circular imports.

---

## Stage 6 — True 3D Collision Detection

### Objective

Fix the collision detection system to compare the UAV's z-coordinate against obstacle ceiling heights, eliminating false positive collisions when the UAV flies above obstacles.

### Modules Affected

| Module | Change |
|--------|--------|
| `core/models/environment_model.py` | Update: `has_collision()` and `point_in_obstacle()` accept z-coordinate and compare against `ObstacleHeightModel` |
| `core/models/obstacle_model.py` | Update: `Obstacle.contains_point(x, y, z)` checks 3D containment — returns `False` if `z > obstacle_ceiling + safety_margin` |
| `core/physics_engine.py` (from Stage 5) | Update: passes UAV z to collision check |
| `core/mission_controller.py` | Minor: passes z-coordinate to collision queries |

### Architectural Change

**Current**: `has_collision(start_pos, end_pos)` takes 2D tuples. `Obstacle.contains_point(x, y)` is a 2D bounding box test. The UAV at altitude 100m is considered colliding with a 20m obstacle if it passes over the XY footprint.

**After**: 
```
has_collision(start_pos_3d, end_pos_3d) → bool:
    for each sampled point (x, y, z) along the 3D path:
        for each obstacle:
            if obstacle.contains_point_2d(x, y):
                ceiling = ObstacleHeightModel.obstacle_height(x, y, obstacle) + SAFETY_MARGIN
                if z <= ceiling:
                    return True  # actual 3D collision
    return False

Obstacle.contains_point_3d(x, y, z) → bool:
    if not self.contains_point_2d(x, y):
        return False
    ceiling = self.height + SAFETY_MARGIN  # or Gaussian profile
    return z <= ceiling
```

### Expected Behavior After Change

- `python3 main.py` shows **zero or near-zero collision count** when UAV is flying at 100m altitude above 20m obstacles (currently reports false positives).
- `collision_rate` metric in the output JSON decreases.
- UAV trajectory may change slightly because the movement scoring no longer over-penalizes obstacle overflight.
- All tests pass (test fixtures updated to include z-coordinates).

---

# Task 3 — Verification Procedures

## Stage 1 Verification — Entity Model Separation

```
EXECUTE:
    cd "/Users/ramayan/Downloads/BTP-Major Project/Code/IOT Based/Phase 1"
    python3 main.py

VERIFY:
    1. Exit code is 0 (simulation completes without errors)
    2. Check visualization/runs/<latest_run_id>/:
       - logs/run_summary.json exists and contains valid metrics
       - logs/config_snapshot.json exists
       - telemetry/step_telemetry.csv exists with correct columns
       - telemetry/node_state.csv exists
       - plots/ directory contains environment.png, dashboard_panel.png
       - frames/ directory contains at least 1 PNG
    3. Grep codebase for "env.nodes[1:]" — must return ZERO matches
    4. Grep codebase for "env.nodes[0]" — must return ZERO matches
    5. Grep for "from core.models.node_model import Node" — only in node_model.py itself
       (all other imports should use UAVState or SensorNode)
    6. Run: python3 -m pytest tests/ -v
       All tests must pass.
    7. Compare run_summary.json metrics against pre-refactor baseline:
       - coverage_ratio_percent: within ±2% of baseline
       - total_energy_consumed: within ±5% of baseline
       - collision_count: identical
```

## Stage 2 Verification — Visualization Decoupling

```
EXECUTE:
    python3 main.py                    # single run with rendering
    python3 main.py --mode batch       # batch run (10 runs, no rendering)

VERIFY:
    1. Single run: exit code 0
    2. Single run: interactive matplotlib window appears during execution
    3. Single run: frames/ contains PNG files
    4. Single run: plots/ contains all expected plots
    5. Batch run: exit code 0
    6. Batch run: completes in < 60 seconds (was previously minutes with rendering)
    7. Grep for "import.*interactive_dashboard\|import.*plot_renderer" in 
       core/mission_controller.py — must return ZERO matches
    8. Grep for "plt\." in core/mission_controller.py — must return ZERO matches
    9. Grep for "render" in core/mission_controller.py — must return ZERO matches
       (MissionController no longer knows about rendering)
    10. Run: python3 -m pytest tests/ -v — all tests pass
```

## Stage 3 Verification — Continuous Simulation Clock

```
EXECUTE:
    python3 main.py

VERIFY:
    1. Exit code 0
    2. Add temporary assertion: at end of run_simulation(), verify
       temporal.current_time == temporal.current_step * Config.TIME_STEP
       (proves backward compatibility — tick() still advances by 1.0)
    3. Write a minimal test:
       clock = TemporalEngine(time_step=1, max_steps=100)
       clock.advance(5.7)
       assert abs(clock.current_time - 5.7) < 1e-9
       clock.advance(94.4)
       assert clock.active == False  # exceeded max_time = 100
    4. visualization/runs/<run_id>/logs/run_summary.json exists
    5. Metrics are within ±1% of Stage 2 baseline (no behavioral change)
    6. Run: python3 -m pytest tests/ -v — all tests pass
```

## Stage 4 Verification — Analytical Service Model

```
EXECUTE:
    python3 main.py

VERIFY:
    1. Exit code 0
    2. Console output shows "[Service] Node X: τ*=Y.Ys, collected Z.ZMb" 
       instead of "[Center-Hover] Node X | Hover #N"
    3. Grep for "hover_step_count" in codebase — ZERO matches
    4. Grep for "MAX_HOVER_STEPS_PER_NODE" in codebase — ZERO matches
    5. visualization/runs/<run_id>/logs/run_summary.json:
       - mission_completion_time may differ (expected)
       - coverage_ratio_percent should be ≥ previous (fewer timeouts)
       - average_aoi should decrease (continuous AoI advancement)
    6. telemetry/step_telemetry.csv:
       - Time column shows non-uniform increments (proof of variable Δt)
    7. Run: python3 -m pytest tests/ -v — all tests pass
    8. Specific test: create a SensorNode with buffer=10Mb, position at distance
       where rate=2Mbps. Service time should be 5.0s ± sensing overhead.
       Verify with: outcome = BufferAwareManager.execute_service(...)
       assert abs(outcome.service_time_s - 5.0) < T_sense_margin
```

## Stage 5 Verification — MissionController Decomposition

```
EXECUTE:
    python3 main.py
    python3 main.py --mode batch

VERIFY:
    1. Both exit code 0
    2. wc -l core/mission_controller.py — should be ≤ 200 lines
    3. core/physics_engine.py exists and contains motion primitive logic
    4. core/observation_builder.py exists
    5. core/reward_function.py exists
    6. Grep for "math.atan2\|math.cos\|math.sin" in core/mission_controller.py —
       ZERO matches (all trigonometry moved to physics_engine.py)
    7. Grep for "import.*visualization" in core/mission_controller.py — ZERO matches
    8. visualization/runs/<run_id>/logs/run_summary.json:
       - All metrics within ±1% of Stage 4 baseline (behavioral equivalence)
    9. Run: python3 -m pytest tests/ -v — all tests pass
    10. No circular imports: python3 -c "from core.physics_engine import PhysicsEngine"
        python3 -c "from core.observation_builder import ObservationBuilder"
        python3 -c "from core.reward_function import RewardFunction"
        All exit code 0.
```

## Stage 6 Verification — True 3D Collision Detection

```
EXECUTE:
    python3 main.py --obstacles

VERIFY:
    1. Exit code 0
    2. visualization/runs/<run_id>/logs/run_summary.json:
       - collision_count should be 0 or very low (UAV flies at 100m, obstacles are 20m)
       - Previous runs showed false positives; this must be resolved
    3. Write specific test:
       - Create obstacle at (100,100)→(200,200) with height 20m
       - UAV at position (150, 150, 100) — above obstacle at altitude 100m
       - env.has_collision((150,150,100), (150,150,100)) must return False
       - UAV at position (150, 150, 15) — below obstacle ceiling
       - env.has_collision((150,150,15), (150,150,15)) must return True
    4. Grep for "contains_point(x, y)" (2-arg version) — should be replaced
       by 3-arg version or explicit 3D check
    5. Run: python3 -m pytest tests/ -v — all tests pass
    6. Trajectory visualization shows UAV path is not deflected by obstacles
       it flies over (previously it would route around them due to false collisions)
```

---

# Task 4 — Git Workflow

## 4.1 Branch Strategy

All work happens on branch `main` (as per current repository convention). Each stage produces one or more commits directly to `main`.

## 4.2 Commit Protocol

For **every stage**, the implementation engineer must follow this exact sequence:

```
1. IMPLEMENT the stage changes
2. RUN the simulator: python3 main.py
3. RUN tests: python3 -m pytest tests/ -v (if tests exist)
4. VERIFY outputs against the stage-specific checklist (Task 3)
5. STAGE files: git add -A
6. COMMIT with conventional commit message:
   git commit -m "<type>(<scope>): <description>"
7. PUSH to remote:
   git push origin main
```

## 4.3 Commit Messages

| Stage | Commit Message |
|-------|---------------|
| 1 | `refactor(entity): split Node into UAVState, SensorNode, and BaseStation` |
| 2 | `refactor(viz): decouple rendering from simulation tick loop` |
| 3 | `refactor(temporal): introduce ContinuousClock with variable time advancement` |
| 4 | `fix(service): replace hover loop with analytical service time τ*=D/R` |
| 5 | `refactor(controller): decompose MissionController into PhysicsEngine, ObservationBuilder, RewardFunction` |
| 6 | `fix(collision): implement true 3D collision detection with altitude comparison` |

## 4.4 Prohibited Practices

The following are **explicitly forbidden**:

1. **Batching commits**: Do NOT commit Stages 1 and 2 together. Each stage = one commit minimum.
2. **Delayed pushing**: Do NOT wait until all stages are done to push. Push after each commit.
3. **Skipping verification**: Do NOT commit without running `python3 main.py` and confirming exit code 0.
4. **Force pushing**: Do NOT use `git push --force`.
5. **Bypassing hooks**: Do NOT use `--no-verify`.
6. **Amending pushed commits**: Do NOT use `git commit --amend` after pushing.

## 4.5 Baseline Snapshots

Before starting Stage 1, the engineer must:

```
python3 main.py
cp visualization/runs/<run_id>/logs/run_summary.json docs/metrics/baseline_pre_refactor.json
git add docs/metrics/baseline_pre_refactor.json
git commit -m "docs(metrics): save pre-refactor baseline metrics snapshot"
git push origin main
```

After each stage, save the metrics for comparison:

```
cp visualization/runs/<run_id>/logs/run_summary.json docs/metrics/baseline_stage_N.json
```

This creates a traceable metrics trail showing the impact of each refactoring stage.

---

# Task 5 — Sonnet Implementation Prompts

## Prompt 1 — Stage 1: Entity Model Separation

```
ROLE: You are an implementation engineer refactoring a UAV simulation platform.

TASK: Split the monolithic `Node` dataclass into three separate entity types.

SCOPE OF CHANGE:
- core/models/node_model.py — primary target
- core/models/environment_model.py — add typed containers
- core/mission_controller.py — update all Node references
- core/simulation_runner.py — update entity creation
- core/dataset_generator.py — return SensorNode list
- core/comms/buffer_aware_manager.py — update parameter types
- core/comms/base_station_uplink.py — use BaseStation
- core/clustering/semantic_clusterer.py — accept SensorNode
- core/rendezvous_selector.py — accept SensorNode
- path/pca_gls_router.py — accept SensorNode
- path/ga_sequence_optimizer.py — accept SensorNode
- metrics/metric_engine.py — separate UAV/sensor metrics
- visualization/interactive_dashboard.py — render separately
- visualization/plot_renderer.py — render separately
- core/telemetry_logger.py — update field references
- tests/* — update all fixtures

WHAT TO CREATE:

In core/models/node_model.py, define three dataclasses:

1. UAVState — fields: id, x, y, z, yaw, pitch, vx, vy, vz, 
   battery_capacity, current_battery, payload_buffer_mbits (new field, 
   replaces the informal tracking in MissionController.collected_data_mbits),
   energy_per_meter, hover_cost, return_threshold.
   Methods: position(), reset_battery()

2. SensorNode — fields: id, x, y, z, priority, risk, signal_strength, 
   deadline, reliability, buffer_capacity, current_buffer, 
   data_generation_rate, aoi_timer, max_aoi_timer, time_window_start, 
   time_window_end, node_battery_J, tx_energy_consumed_J.
   Methods: position()

3. BaseStation — fields: x, y, z (defaults from Config.BASE_STATION or 
   (MAP_WIDTH//2, MAP_HEIGHT//2, 0)).
   Methods: position()

Keep the old `Node` class as a deprecated alias if needed for 
backward compatibility during transition, but all new code must 
use the specific types.

In core/models/environment_model.py:
- Add: self.uav: UAVState (set during simulation setup)
- Add: self.sensors: List[SensorNode] (replaces self.nodes for ground nodes)
- Add: self.base_station: BaseStation
- Remove: self.nodes (or keep as deprecated computed property: 
  return [self.uav] + self.sensors)
- Method add_node() becomes add_sensor()
- All env.nodes[1:] patterns become env.sensors
- All env.nodes[0] patterns become env.uav

In core/simulation_runner.py:
- Create UAVState separately (not via generate_nodes)
- generate_nodes() returns List[SensorNode]
- Pass UAV and sensors to Environment separately

In core/mission_controller.py:
- self.uav typed as UAVState (not Node)
- Replace every env.nodes[1:] with env.sensors
- Replace every env.nodes[0] with env.uav

Apply the same pattern to ALL other files that import Node.

CONSTRAINTS:
- Do NOT change any algorithm logic
- Do NOT change config values
- Do NOT modify visualization rendering logic (only type references)
- Do NOT remove any existing functionality
- position() method must return Tuple[float, float, float] on all three types
- All existing tests must continue to pass after fixture updates
- The simulation must produce equivalent output

VERIFICATION:
After completing all changes:
1. Run: python3 main.py
   Confirm: exit code 0, no errors
2. Run: python3 -m pytest tests/ -v
   Confirm: all tests pass
3. Check: visualization/runs/<run_id>/logs/run_summary.json exists
4. Check: visualization/runs/<run_id>/telemetry/step_telemetry.csv exists
5. Grep: "env.nodes[1:]" across entire codebase → ZERO matches
6. Grep: "env.nodes[0]" across entire codebase → ZERO matches

GIT:
git add -A
git commit -m "refactor(entity): split Node into UAVState, SensorNode, and BaseStation"
git push origin main
```

---

## Prompt 2 — Stage 2: Visualization Decoupling

```
ROLE: You are an implementation engineer refactoring a UAV simulation platform.

TASK: Remove all rendering and visualization calls from MissionController.step(). 
Rendering must be orchestrated exclusively from simulation_runner.py.

SCOPE OF CHANGE:
- core/mission_controller.py — remove all visualization imports and calls
- core/simulation_runner.py — take ownership of rendering orchestration

MODULES NOT ALLOWED TO BE MODIFIED:
- visualization/interactive_dashboard.py (internal logic untouched)
- visualization/plot_renderer.py (internal logic untouched)
- Any path/, metrics/, or core/comms/ module

WHAT TO DO:

In core/mission_controller.py:
1. Remove the import of InteractiveDashboard
2. Remove the import of PlotRenderer
3. Remove the self.render_enabled attribute
4. Remove the self.interactive_dash attribute and its lazy initialization
5. Remove the interactive_dash.render() call inside step()
6. Remove the PlotRenderer.render_environment_frame() call inside step()
7. Remove the run_manager reference from __init__ if only used for frames 
   (keep if used for other purposes)
8. The constructor should no longer accept a render parameter

In core/simulation_runner.py:
1. Create InteractiveDashboard before the while loop (only if render=True)
2. Inside the while loop, AFTER mission.step() and telemetry.log_step():
   - Call interactive_dash.render(...) if render is True
   - Call PlotRenderer.render_environment_frame() at keyframe intervals if render
3. All render gating uses the local `render` parameter, NOT a flag on 
   MissionController

The interactive dashboard needs access to: env.uav, mission.current_target, 
temporal.current_step, base_position (from mission or config), 
active_centroids (from mission). These should be read from the mission 
object's public attributes — do NOT add methods just for rendering.

CONSTRAINTS:
- MissionController must have ZERO imports from visualization/
- MissionController must have ZERO references to plt, render, dashboard, frame
- Rendering behavior must be identical to current behavior when render=True
- Batch mode (render=False) must skip all rendering entirely
- Do NOT change any simulation logic, algorithms, or metrics

VERIFICATION:
1. Run: python3 main.py
   Confirm: exit code 0, interactive dashboard appears, frames generated
2. Run: python3 main.py --mode batch
   Confirm: exit code 0, completes in reasonable time
3. Grep: "visualization" in core/mission_controller.py → ZERO matches
4. Grep: "plt\." in core/mission_controller.py → ZERO matches
5. Grep: "render" in core/mission_controller.py → ZERO matches
6. Check: visualization/runs/<run_id>/frames/ contains PNGs
7. Check: visualization/runs/<run_id>/plots/ contains all expected plots
8. Run: python3 -m pytest tests/ -v — all tests pass

GIT:
git add -A
git commit -m "refactor(viz): decouple rendering from simulation tick loop"
git push origin main
```

---

## Prompt 3 — Stage 3: Continuous Simulation Clock

```
ROLE: You are an implementation engineer refactoring a UAV simulation platform.

TASK: Extend TemporalEngine to support variable time advancement via a new 
advance(dt) method, maintaining full backward compatibility with the existing 
tick() method.

SCOPE OF CHANGE:
- core/temporal_engine.py — primary target

MODULES NOT ALLOWED TO BE MODIFIED:
- core/mission_controller.py (no changes needed — still uses tick())
- Any visualization, path, metrics, or comms module

WHAT TO DO:

In core/temporal_engine.py:

1. Add a new float attribute: self._current_time = 0.0
   This tracks continuous simulation time in seconds.

2. Add a new float attribute: self.max_time = max_steps * time_step
   This is the continuous-time equivalent of max_steps.

3. Add method advance(dt: float):
   - self._current_time += dt
   - Update self.current_step = int(self._current_time / self.time_step)
     for backward compatibility with code that reads current_step
   - If self._current_time >= self.max_time: self.active = False
   - Return self.active

4. Add property current_time -> float:
   - Returns self._current_time

5. Modify existing tick() method:
   - Implement as: return self.advance(self.time_step)
   - This ensures tick() and advance() share the same code path
   - Remove the old manual step increment

6. Update reset():
   - self._current_time = 0.0
   - self.current_step = 0
   - self.active = True

7. Update termination check:
   - Use self._current_time >= self.max_time instead of 
     self.current_step > self.max_steps

CONSTRAINTS:
- tick() must continue to work exactly as before
- All existing code calling tick() must produce identical behavior
- No other files should need modification for this stage
- Do NOT change Config
- Do NOT add any new dependencies

VERIFICATION:
1. Run: python3 main.py
   Confirm: exit code 0, identical behavior to previous stage
2. Check: metrics in run_summary.json are within ±1% of Stage 2 baseline
3. Write and run a quick manual test in the terminal:
   python3 -c "
   from core.temporal_engine import TemporalEngine
   t = TemporalEngine(time_step=1, max_steps=100)
   t.advance(5.7)
   assert abs(t.current_time - 5.7) < 1e-9, f'Expected 5.7, got {t.current_time}'
   assert t.current_step == 5, f'Expected 5, got {t.current_step}'
   assert t.active == True
   t.advance(95.0)
   assert t.active == False, 'Should be inactive after exceeding max_time'
   print('ContinuousClock: all assertions passed')
   "
4. Run: python3 -m pytest tests/ -v — all tests pass

GIT:
git add -A
git commit -m "refactor(temporal): introduce ContinuousClock with variable time advancement"
git push origin main
```

---

## Prompt 4 — Stage 4: Analytical Service Model

```
ROLE: You are an implementation engineer refactoring a UAV simulation platform.

TASK: Replace the step-based hover loop in MissionController._move_one_step() 
with the Donipati et al. analytical service time formula: 
τ* = D_i(t) / R_i(t).

When the UAV arrives directly above a node (distance < threshold), 
compute the exact service duration analytically, advance the clock by 
that amount, and drain the buffer in a single operation.

SCOPE OF CHANGE:
- core/comms/buffer_aware_manager.py — add execute_service() method
- core/mission_controller.py — replace Center-Hover loop with single service call
- core/models/energy_model.py — ensure hover_energy() accepts continuous duration
- config/config.py — replace MAX_HOVER_STEPS_PER_NODE with MAX_SERVICE_TIME_S

MODULES NOT ALLOWED TO BE MODIFIED:
- core/comms/communication.py (already correct)
- path/* (routing untouched)
- visualization/* (untouched)

WHAT TO DO:

A) In core/comms/buffer_aware_manager.py, add a new method:

   execute_service(uav, node, env, temporal, energy_model, base_position):
   
   Steps:
   1. Compute R_i = CommunicationEngine.achievable_data_rate(
          node.position(), uav.position(), env)
      If R_i <= 0: return ServiceOutcome with 0 data (NLoS blocked)
   
   2. Compute analytical service time:
      tau_star = node.current_buffer / R_i   (seconds)
   
   3. Compute sensing overhead:
      dist = euclidean_distance(node.position(), uav.position())
      T_sense = CommunicationEngine.minimum_hover_time(dist)
   
   4. Total hover duration:
      T_total = tau_star + T_sense
      Cap at MAX_SERVICE_TIME_S for safety
   
   5. Energy feasibility:
      E_hover = energy_model.hover_energy(uav, T_total)
      E_return = energy_model.energy_for_return(uav, base_position)
      If uav.current_battery < E_hover + E_return * 1.15:
          Reduce T_total to the maximum affordable duration
   
   6. Execute:
      data_collected = min(node.current_buffer, R_i * T_total)
      node.current_buffer -= data_collected
      Consume hover energy from UAV
      temporal.advance(T_total)    ← uses the new ContinuousClock
   
   7. Return a result object (namedtuple or dataclass):
      ServiceOutcome(data_collected, service_time_s, energy_consumed, 
                     achievable_rate_mbps)

   Keep the existing process_data_collection() method for Chord-Fly 
   during transit (still step-based, which is correct for in-flight collection).

B) In core/mission_controller.py, replace the Center-Hover block:

   Current pattern (REMOVE):
   ```
   if distance < 1e-3:
       hover_e = EnergyModel.hover_energy(self.uav, dt)
       EnergyModel.consume(self.uav, hover_e)
       ... hover_strategy = BufferAwareManager.get_optimal_hover_strategy(...)
       ... data_collected = BufferAwareManager.process_data_collection(...)
       ... self._hover_step_count[nid] += 1
       ... if buffer <= 0 or hover_count >= MAX: mark visited
   ```
   
   Replace with:
   ```
   if distance < 1e-3:
       outcome = BufferAwareManager.execute_service(
           uav=self.uav, node=self.current_target,
           env=self.env, temporal=self.temporal,
           energy_model=EnergyModel, base_position=self.base_position
       )
       if outcome.data_collected > 0:
           self.collected_data_mbits += outcome.data_collected
           self.rate_log.append(outcome.achievable_rate_mbps)
       self.energy_consumed_total += outcome.energy_consumed
       
       # Advance all other sensors' buffers and AoI
       for sensor in self.env.sensors:
           if sensor.id != self.current_target.id and sensor.id not in self.visited:
               sensor.current_buffer = min(
                   sensor.buffer_capacity,
                   sensor.current_buffer + sensor.data_generation_rate * outcome.service_time_s
               )
               sensor.aoi_timer += outcome.service_time_s
       
       print(f"[Service] Node {self.current_target.id}: "
             f"τ*={outcome.service_time_s:.1f}s, "
             f"collected {outcome.data_collected:.2f}Mb")
       self.visited.add(self.current_target.id)
       self.current_target = None
       return
   ```

C) Remove from MissionController:
   - self._hover_step_count dict
   - All references to MAX_HOVER_STEPS_PER_NODE

D) In config/config.py:
   - Remove: MAX_HOVER_STEPS_PER_NODE = 30
   - Add: MAX_SERVICE_TIME_S = 30.0  (safety cap in seconds)

E) In core/models/energy_model.py:
   - Ensure hover_energy(node, duration) works with float duration 
     (currently uses dt=Config.TIME_STEP; make it accept any positive float)

CONSTRAINTS:
- Chord-Fly collection during transit (the code before the distance check) 
  is UNCHANGED — it correctly uses per-step collection
- Do NOT modify routing algorithms (PCA-GLS, GA, SCA)
- Do NOT modify clustering logic
- Do NOT modify communication.py
- The analytical formula τ* = D_i/R_i is the ONLY correct formula; do NOT 
  add heuristics or approximations
- Keep the NLoS blocked failsafe (R_i = 0 → abandon node)

VERIFICATION:
1. Run: python3 main.py
   Confirm: exit code 0
2. Console output shows "[Service] Node X: τ*=Y.Ys, collected Z.ZMb"
   NOT "[Center-Hover] Node X | Hover #N"
3. Grep: "_hover_step_count" in codebase → ZERO matches
4. Grep: "MAX_HOVER_STEPS_PER_NODE" in codebase → ZERO matches
5. Check run_summary.json:
   - coverage_ratio_percent: should be ≥ previous (fewer timeouts)
   - average_aoi: expect decrease
6. Check telemetry/step_telemetry.csv:
   - If a time column exists, verify non-uniform intervals
7. Run: python3 -m pytest tests/ -v — all tests pass

GIT:
git add -A
git commit -m "fix(service): replace hover loop with analytical service time τ*=D/R"
git push origin main
```

---

## Prompt 5 — Stage 5: MissionController Decomposition

```
ROLE: You are an implementation engineer refactoring a UAV simulation platform.

TASK: Decompose the MissionController (currently ~600 lines) into focused 
modules. Extract the motion primitive logic into PhysicsEngine, create an 
ObservationBuilder for RL state vectors, and create a RewardFunction module.

SCOPE OF CHANGE:
- core/mission_controller.py — reduce to orchestration only (~200 lines max)
- New file: core/physics_engine.py — motion primitives, scoring, energy check
- New file: core/observation_builder.py — RL state vector construction
- New file: core/reward_function.py — modular reward computation
- core/simulation_runner.py — pass new dependencies to MissionController
- tests/ — add basic tests for new modules

MODULES NOT ALLOWED TO BE MODIFIED:
- core/comms/* (communication and buffer logic untouched)
- path/* (routing algorithms untouched)
- visualization/* (untouched)
- config/* (untouched)

WHAT TO DO:

A) Create core/physics_engine.py:

   Extract from MissionController._move_one_step():
   - Motion primitive generation (yaw/pitch offsets, spherical-to-Cartesian)
   - Candidate scoring (distance-to-target, obstacle clearance, energy 
     feasibility, risk multiplier, goal progress)
   - Best-move selection
   - Energy consumption for movement
   - Obstacle height enforcement via ObstacleHeightModel
   - Digital Twin ISAC scan (obstacle routing decision)
   - Acceleration dynamics and kinematic constraints (max yaw/pitch rate)
   
   Interface:
   class PhysicsEngine:
       @staticmethod
       def execute_movement(uav, target_pos, env, energy_model, dt, 
                           digital_twin=None, hover_positions=None):
           """
           Compute and apply one movement step.
           Returns MovementOutcome(new_pos, energy_consumed, collision_detected)
           """
   
   Also extract helper methods:
   - _rectangle_clearance()
   - _predicted_clearance()
   - generate_motion_primitives()
   - score_candidate()

B) Create core/observation_builder.py:

   This module constructs the RL observation vector.
   It wraps AgentCentricTransform and adds obstacle features.
   
   class ObservationBuilder:
       @staticmethod
       def build(uav, target, sensors, obstacles, current_step, config=None):
           """
           Returns a normalized state vector for RL agents.
           Length: 7 (Wang et al. Table I) + K*3 (nearest obstacles)
           """
           Uses: core.sensing.agent_centric_transform.AgentCentricTransform
   
   This module is not called by MissionController in this stage.
   It is scaffolding for Phase 5 RL integration.

C) Create core/reward_function.py:

   class RewardFunction:
       # Default weights (tunable)
       ALPHA_DATA = 1.0
       ALPHA_ENERGY = 0.3
       ALPHA_AOI = 0.5
       ALPHA_COLLISION = 10.0
       ALPHA_DEADLINE = 0.5
       
       @staticmethod
       def compute(data_collected, energy_consumed, energy_budget,
                   avg_aoi, max_aoi, collision, deadline_violations,
                   num_nodes, max_time):
           """
           Returns scalar reward.
           r = α₁·r_data + α₂·r_energy + α₃·r_aoi + α₄·r_collision + α₅·r_deadline
           """
   
   This module is not called by MissionController in this stage.
   It is scaffolding for Phase 5 RL integration.

D) Refactor core/mission_controller.py:

   The step() method becomes:
   1. temporal.tick()
   2. Update environment dynamics (obstacles, risk zones, node spawn/removal)
   3. Periodic re-clustering
   4. Replan handling
   5. Energy threshold check
   6. Fill sensor buffers
   7. BS uplink urgency check
   8. If no current target: pop from queue
   9. Chord-Fly collection via BufferAwareManager.process_data_collection()
   10. Check if buffer drained (mark visited via Chord-Fly)
   11. If at target: BufferAwareManager.execute_service() (from Stage 4)
   12. Else: PhysicsEngine.execute_movement() ← NEW DELEGATION
   13. Record histories
   14. Check terminal conditions
   
   MissionController.__init__ should accept physics_engine as parameter.
   
   Remove from MissionController:
   - All motion primitive generation code
   - All candidate scoring code  
   - _rectangle_clearance(), _predicted_clearance()
   - All trigonometric calculations (math.sin, math.cos, math.atan2)

E) Update core/simulation_runner.py:
   - Import PhysicsEngine
   - Pass to MissionController constructor (or let MC instantiate internally)

CONSTRAINTS:
- BEHAVIORAL EQUIVALENCE: the simulation must produce identical metrics 
  (within ±1% floating-point tolerance) as the previous stage
- Do NOT change any algorithm logic during extraction
- Do NOT change config values
- Do NOT add new features or optimizations
- ObservationBuilder and RewardFunction are scaffolding — they must be 
  importable and have correct method signatures, but are NOT called by 
  the simulation yet

VERIFICATION:
1. Run: python3 main.py
   Confirm: exit code 0, same visual behavior
2. Run: python3 main.py --mode batch
   Confirm: exit code 0
3. wc -l core/mission_controller.py — should be ≤ 200 lines
4. File exists: core/physics_engine.py
5. File exists: core/observation_builder.py
6. File exists: core/reward_function.py
7. Grep: "math.atan2\|math.cos\|math.sin" in core/mission_controller.py 
   → ZERO matches
8. No circular imports:
   python3 -c "from core.physics_engine import PhysicsEngine; print('OK')"
   python3 -c "from core.observation_builder import ObservationBuilder; print('OK')"
   python3 -c "from core.reward_function import RewardFunction; print('OK')"
9. run_summary.json metrics within ±1% of Stage 4 baseline
10. Run: python3 -m pytest tests/ -v — all tests pass

GIT:
git add -A
git commit -m "refactor(controller): decompose MissionController into PhysicsEngine, ObservationBuilder, RewardFunction"
git push origin main
```

---

## Prompt 6 — Stage 6: True 3D Collision Detection

```
ROLE: You are an implementation engineer refactoring a UAV simulation platform.

TASK: Fix the collision detection system to perform true 3D checks. 
Currently, has_collision() and point_in_obstacle() operate in 2D only, 
causing false positive collisions when the UAV flies at altitude 100m 
above 20m obstacles.

SCOPE OF CHANGE:
- core/models/environment_model.py — update collision methods to accept 
  and check z-coordinate
- core/models/obstacle_model.py — add 3D containment check
- core/physics_engine.py (from Stage 5) — pass z-coordinate to collision checks

MODULES NOT ALLOWED TO BE MODIFIED:
- core/comms/* (untouched)
- path/* (untouched)
- visualization/* (untouched)
- config/* (untouched)

WHAT TO DO:

A) In core/models/obstacle_model.py:

   Modify Obstacle class:
   - Add a height attribute: self.height = height (default 20.0m, or 
     derive from existing Gaussian height model)
   - Add method contains_point_3d(x, y, z) → bool:
     If not within 2D bounding box: return False
     ceiling = ObstacleHeightModel.obstacle_height(x, y, self) + Config.VERTICAL_CLEARANCE
     (or a simpler: ceiling = self.height + safety_margin)
     return z <= ceiling
   
   Keep existing contains_point(x, y) for backward compatibility 
   (2D check only, used by ground-level operations like node placement).

B) In core/models/environment_model.py:

   Modify has_collision():
   - Change signature to accept 3D tuples:
     has_collision(start_pos, end_pos) where pos is (x, y, z)
   - Sample points along the 3D path
   - For each sample point (x, y, z):
     Check each obstacle with contains_point_3d(x, y, z)
   - Fall back gracefully if z is not provided (treat as 2D for 
     backward compatibility with any remaining 2D callers)
   
   Modify point_in_obstacle():
   - Add optional z parameter:
     point_in_obstacle(pos, z=None)
   - If z is provided: use 3D check
   - If z is None: use existing 2D check (backward compatible)

C) In core/physics_engine.py:

   When calling env.has_collision() or env.point_in_obstacle(), 
   pass the full 3D position (x, y, z) including the UAV's altitude.
   
   Specifically in the motion primitive scoring:
   - Replace: self.env.point_in_obstacle((new_x, new_y))
   - With: self.env.point_in_obstacle((new_x, new_y), z=new_z)
   
   And in path collision checking:
   - Replace: self.env.has_collision((x1, y1), (x2, y2))
   - With: self.env.has_collision((x1, y1, z1), (x2, y2, z2))

D) Add Config.COLLISION_VERTICAL_CLEARANCE = 5.0 (meters) — safety margin 
   above obstacle ceiling.

CONSTRAINTS:
- Do NOT change obstacle generation or placement logic
- Do NOT change the ObstacleHeightModel Gaussian profile
- Do NOT change routing algorithms
- 2D collision checks for ground operations (node placement in 
  dataset_generator.py, get_safe_start()) must continue to work
- The UAV at altitude 100m must NOT trigger collision with 20m obstacles

VERIFICATION:
1. Run: python3 main.py --obstacles
   Confirm: exit code 0
2. Check run_summary.json:
   - collision_count should be 0 or very low (UAV at 100m, obstacles at 20m)
3. Manual test:
   python3 -c "
   from core.models.obstacle_model import Obstacle
   from core.models.environment_model import Environment
   env = Environment(800, 600)
   obs = Obstacle(100, 100, 200, 200)  # 2D footprint
   env.add_obstacle(obs)
   # UAV at 100m altitude — should NOT collide
   result_above = env.has_collision((150, 150, 100), (150, 150, 100))
   print(f'Above obstacle (100m): collision={result_above}')  # expect False
   # UAV at 15m altitude — should collide (below 20m ceiling)
   result_below = env.has_collision((150, 150, 15), (150, 150, 15))
   print(f'Below ceiling (15m): collision={result_below}')    # expect True
   assert result_above == False, 'FALSE POSITIVE: UAV above obstacle detected as collision'
   assert result_below == True, 'FALSE NEGATIVE: UAV inside obstacle not detected'
   print('3D collision: all assertions passed')
   "
4. Run: python3 -m pytest tests/ -v — all tests pass

GIT:
git add -A
git commit -m "fix(collision): implement true 3D collision detection with altitude comparison"
git push origin main
```

---

# Task 6 — Final Simulator Architecture

After all six refactoring stages, the simulator has the following architecture:

## 6.1 Module Map

```
main.py
├── config/
│   ├── config.py              SimConfig (static class, ~120 parameters)
│   └── feature_toggles.py     CLI override controller
│
├── core/
│   ├── models/
│   │   ├── node_model.py      UAVState, SensorNode, BaseStation  [Stage 1]
│   │   ├── environment_model.py  Environment (uav, sensors, base_station,
│   │   │                          obstacles, risk_zones) + 3D collision  [Stage 6]
│   │   ├── energy_model.py    Rotary-wing propulsion energy model
│   │   ├── obstacle_model.py  Obstacle (3D containment), ObstacleHeightModel  [Stage 6]
│   │   └── risk_zone_model.py RiskZone with temporal fluctuation
│   │
│   ├── comms/
│   │   ├── communication.py       Rician fading, LoS probability, Shannon capacity
│   │   ├── buffer_aware_manager.py  DST-BA: execute_service(τ*=D/R), Chord-Fly  [Stage 4]
│   │   └── base_station_uplink.py   UAV↔BS uplink model
│   │
│   ├── clustering/
│   │   ├── cluster_manager.py     Recluster triggers, delegates to SemanticClusterer
│   │   ├── semantic_clusterer.py  PCA + KMeans/DBSCAN/GMM/auto
│   │   └── feature_scaler.py     MinMax / ZScore normalization
│   │
│   ├── sensing/
│   │   ├── digital_twin_map.py       ISAC local obstacle memory
│   │   └── agent_centric_transform.py  Body-frame coordinate transform (RL ready)
│   │
│   ├── mission_controller.py   Thin orchestrator (~200 lines)  [Stage 5]
│   ├── physics_engine.py       Motion primitives, scoring, movement  [Stage 5]
│   ├── observation_builder.py  RL state vector construction (scaffolding)  [Stage 5]
│   ├── reward_function.py      RL reward computation (scaffolding)  [Stage 5]
│   ├── temporal_engine.py      ContinuousClock with advance(dt)  [Stage 3]
│   ├── simulation_runner.py    Orchestrates full run + rendering  [Stage 2]
│   ├── batch_runner.py         Multi-run statistical aggregation
│   ├── dataset_generator.py    SensorNode generation
│   ├── seed_manager.py         Deterministic seeding
│   ├── run_manager.py          Filesystem artifact management
│   ├── stability_monitor.py    Windowed stability scoring
│   ├── telemetry_logger.py     Per-step CSV recording
│   └── rendezvous_selector.py  Donipati Alg. 1 RP selection
│
├── path/
│   ├── pca_gls_router.py         Path Cheapest Arc + Guided Local Search
│   ├── ga_sequence_optimizer.py   GA visiting order (Zheng Alg. 2)
│   └── hover_optimizer.py         SCA hover position refinement
│
├── metrics/
│   ├── metric_engine.py    IEEE-aligned metrics computation
│   ├── auto_logger.py      Automated IEEE doc logging
│   └── latex_exporter.py   LaTeX table generation
│
├── visualization/
│   ├── interactive_dashboard.py  Matplotlib live dashboard (called from sim runner)
│   ├── plot_renderer.py          Static plot generation
│   ├── animation_builder.py      MP4/GIF from frames
│   └── batch_plotter.py          Boxplots, heatmaps, Pareto
│
├── experiments/
│   ├── ablation_runner.py     Feature contribution analysis
│   └── scalability_runner.py  Node count scaling tests
│
└── tests/
    ├── conftest.py
    ├── test_clustering.py
    ├── test_communication.py
    ├── test_energy_model.py
    ├── test_node_model.py     → test_entity_model.py  [Stage 1]
    └── test_path_planning.py
```

## 6.2 Subsystem Interactions

### Environment Model
- **Owns**: `UAVState`, `List[SensorNode]`, `BaseStation`, obstacles, risk zones
- **Provides**: 3D collision detection, risk multiplier, safe start computation
- **Consumed by**: MissionController, PhysicsEngine, BufferAwareManager, all planners

### UAV Dynamics Model
- **Components**: `PhysicsEngine` (motion primitives), `EnergyModel` (propulsion), `UAVState` (state)
- **Provides**: Movement execution, energy feasibility, kinematic constraints
- **Consumed by**: MissionController (delegates movement to PhysicsEngine)

### Communication Model
- **Components**: `CommunicationEngine` (channel), `BufferAwareManager` (service), `BaseStationUplinkModel` (uplink)
- **Provides**: Achievable data rate, analytical service time, TDMA scheduling, BS uplink urgency
- **Consumed by**: MissionController (service and uplink), PhysicsEngine (for Chord-Fly data during transit)

### Service Model
- **Component**: `BufferAwareManager.execute_service()`
- **Provides**: Analytical service time computation (τ* = D/R), continuous clock advancement, buffer drain
- **Consumed by**: MissionController (when UAV arrives at target)

### Mission Planner
- **Components**: `RendezvousSelector`, `PCAGLSRouter`, `GASequenceOptimizer`, `HoverOptimizer`, `ClusterManager/SemanticClusterer`
- **Provides**: Target queue (ordered waypoints with hover positions)
- **Consumed by**: MissionController (replan trigger → fresh queue)

### Simulation Engine
- **Components**: `TemporalEngine` (clock), `simulation_runner.run_simulation()`, `MissionController`
- **Provides**: Full simulation execution, step orchestration, termination control
- **Consumed by**: `main.py`, `BatchRunner`, `AblationRunner`

### Experiment Framework
- **Components**: `BatchRunner`, `AblationRunner`, `ScalabilityRunner`
- **Provides**: Multi-run execution, statistical aggregation, ablation analysis
- **Consumed by**: CLI modes (`--mode batch`, `-m experiments.ablation_runner`)

### Visualization Layer
- **Components**: `InteractiveDashboard`, `PlotRenderer`, `AnimationBuilder`, `BatchPlotter`
- **Provides**: Live monitoring, post-run plots, animations
- **Consumed by**: `simulation_runner.py` (orchestrates rendering outside physics tick)
- **Never consumed by**: MissionController, PhysicsEngine, or any core module

## 6.3 Data Flow (Post-Refactor)

```
Config → SeedManager → generate_nodes() → List[SensorNode]
  ↓                                              ↓
UAVState ← manual creation               Environment(uav, sensors, base_station, obstacles)
  ↓                                              ↓
TemporalEngine(ContinuousClock)          MissionController(env, temporal, physics)
  ↓                                              ↓
                    ┌──── Simulation Loop ────────────────────┐
                    │  temporal.tick()                         │
                    │  env.update_dynamics(dt)                 │
                    │  Planner.maybe_replan()                  │
                    │  BufferAwareManager.fill_buffers(dt)     │
                    │  BS uplink check                         │
                    │  PhysicsEngine.execute_movement()        │
                    │    OR                                    │
                    │  BufferAwareManager.execute_service()    │
                    │    → temporal.advance(τ*)                │
                    │  Record: telemetry, histories            │
                    │  Check: terminal conditions              │
                    └──── After step() ──────────────────────┘
                                    ↓
                    Renderer.render() [from simulation_runner]
                                    ↓
                    MetricEngine.compute() → JSON
                    PlotRenderer.render_*() → PNGs
                    TelemetryLogger → CSV
```

---

# Task 7 — Phase-5 RL Foundation

## 7.1 Gymnasium Environment Interface

After the six refactoring stages, the simulator can be wrapped in a Gymnasium-compatible environment. This is the **next implementation target** after the refactoring program completes.

### State Representation

```
observation_space = gymnasium.spaces.Box(
    low=-1.0, high=1.0, 
    shape=(7 + K_OBS * 3,), 
    dtype=np.float32
)

Features [0:7] — Wang et al. (IEEE IoT 2022), Table I:
    [0] agent-centric target x  (p_ac_x / d_norm)        ∈ [-1, 1]
    [1] agent-centric target y  (p_ac_y / d_norm)        ∈ [-1, 1]
    [2] altitude difference     (Δz / H_max)              ∈ [-1, 1]
    [3] node buffer fraction    (buffer / capacity)        ∈ [0, 1]
    [4] node priority fraction  (priority / P_max)         ∈ [0, 1]
    [5] UAV battery fraction    (battery / capacity)       ∈ [0, 1]
    [6] elapsed time fraction   (t / T_max)                ∈ [0, 1]

Features [7:7+K*3] — K nearest obstacles in body frame:
    For each of K obstacles: (body_dx, body_dy, dz_to_ceiling) normalized

Source: ObservationBuilder.build() (created in Stage 5)
```

### Action Space

Two modes, selectable at environment construction:

**Mode A — Target Selection (Discrete)**:
```
action_space = gymnasium.spaces.Discrete(N_nodes + 1)

Action i ∈ [0, N-1]: select sensor node i as next target
Action N: return to base station

The classical planner (PCA-GLS + GA) handles the flight path between 
the current position and the selected target. The RL agent only decides 
the ordering.
```

**Mode B — Continuous Velocity Control (Box)**:
```
action_space = gymnasium.spaces.Box(
    low=np.array([-1.0, -1.0, -1.0]),
    high=np.array([1.0, 1.0, 1.0]),
    dtype=np.float32
)

Action (Δvx, Δvy, ΔH): continuous velocity command in body frame.
Decoded via AgentCentricTransform.agent_to_world().
PhysicsEngine applies kinematic constraints (max yaw rate, max acceleration).
```

### Reward Function

```
r_t = α₁·r_data + α₂·r_energy + α₃·r_aoi + α₄·r_collision + α₅·r_deadline

r_data     = data_collected_this_step / D_max          ∈ [0, 1]
r_energy   = -E_consumed_this_step / E_budget          ∈ [-1, 0]
r_aoi      = -(Σ_i AoI_i) / (N × T_max)              ∈ [-1, 0]
r_collision = -1.0 if collision_detected else 0.0       sparse
r_deadline  = -count(deadline_violations) / N           ∈ [-1, 0]

Default weights: α₁=1.0, α₂=0.3, α₃=0.5, α₄=10.0, α₅=0.5

Source: RewardFunction.compute() (created in Stage 5)
```

### Episode Termination

```
terminated = True if:
    - All sensor nodes have been visited (buffer fully drained) → success
    - UAV collision with obstacle (z < ceiling) → failure

truncated = True if:
    - clock.current_time >= T_max (time budget exhausted)
    - uav.current_battery <= 0 (energy depleted)
    - uav.current_battery < energy_to_return_to_base (safety trigger)
```

### Info Dictionary

```
info = {
    "nodes_visited": int,
    "total_data_collected_mbits": float,
    "total_energy_consumed_J": float,
    "average_aoi": float,
    "deadline_violations": int,
    "replan_count": int,
    "collision_count": int,
    "coverage_ratio": float,
    "mission_success": bool,
}
```

## 7.2 Supported RL Algorithms

| Algorithm | Paper | Action Space | Key Features |
|-----------|-------|-------------|--------------|
| D3QN | Wang et al. (IEEE IoT 2022) | Discrete (target selection) | Dueling architecture, double Q-learning, PER |
| TD3 | Chen et al. (TD3+ISAC+DT) | Continuous (velocity command) | Twin critics, delayed policy update, Gaussian noise |
| Hierarchical | Proposed novel contribution | D3QN (high-level target) + TD3 (low-level velocity) | Two-tier decision making, separate training |

## 7.3 Training Infrastructure Requirements

The following must be verifiable before RL training begins:

| Requirement | How to Verify |
|---|---|
| `env.reset()` returns valid observation | `obs, info = env.reset(); assert obs.shape == (7+K*3,)` |
| `env.step(action)` returns 5-tuple | `obs, reward, term, trunc, info = env.step(0)` |
| Episodes terminate correctly | Run 100 episodes; verify all terminate within T_max |
| Deterministic seeding | Same seed → identical trajectory |
| Vectorized envs work | `gymnasium.vector.SyncVectorEnv([make_env]*4)` |
| Reward is bounded | `assert -15 <= reward <= 2` across 1000 random steps |
| Observation is normalized | `assert np.all(np.abs(obs) <= 1.0 + 1e-6)` |

---

# Task 8 — Research Platform Requirements

## 8.1 Reproducibility

| Requirement | Status After Refactoring |
|---|---|
| Deterministic seeding | ✅ `SeedManager` controls numpy, random, torch seeds |
| Config snapshot per run | ✅ `config_snapshot.json` saved by `RunManager` |
| Telemetry recording | ✅ `TelemetryLogger` writes per-step CSV |
| Run isolation | ✅ Each run gets unique `visualization/runs/<run_id>/` directory |
| Seed override via CLI | ✅ `--seed` argument in main.py (if not present, must be added) |

## 8.2 Parameter Sweeps

The experiment framework must support:

```
Sweep Protocol:
    for seed in [42, 123, 256, 789, 1024]:
        for node_count in [10, 20, 50, 100]:
            for obstacle_mode in ["none", "static_5", "moving_5", "moving_10"]:
                run_simulation(seed=seed, config_overrides={
                    "NODE_COUNT": node_count,
                    "ENABLE_OBSTACLES": obstacle_mode != "none",
                    "OBSTACLE_COUNT": int(obstacle_mode.split("_")[1]) if "_" in obstacle_mode else 0,
                    "ENABLE_MOVING_OBSTACLES": "moving" in obstacle_mode,
                })
```

**Current support**: `BatchRunner` runs 10 identical seeds. `AblationRunner` toggles feature flags.

**Required addition**: A `SweepRunner` that accepts a parameter grid and produces a results matrix. This is a post-refactoring enhancement, not part of the 6 stages.

## 8.3 Statistical Evaluation

| Metric | Formula | Implementation |
|---|---|---|
| Mean ± 95% CI | $\bar{x} \pm 1.96 \cdot \frac{s}{\sqrt{n}}$ | `BatchRunner` computes mean/std/CI |
| Paired comparison | Welch's t-test, $p < 0.05$ | Must be added post-refactoring |
| Effect size | Cohen's $d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}$ | Must be added post-refactoring |
| Ablation delta | $\frac{x_{ablated} - x_{baseline}}{x_{baseline}} \times 100\%$ | `AblationRunner._compute_delta()` ✅ |

## 8.4 Baseline Algorithm Comparisons

The refactored platform must support running these baselines against the same scenarios:

| Baseline | Description | How to Run |
|---|---|---|
| Random Walk | UAV moves randomly, collects from nodes in range | Replace planner with random target selection |
| Greedy Nearest | Always fly to nearest unvisited node | Replace PCA-GLS queue with distance sort |
| Classical (PCA-GLS + GA) | Current system | Default mode |
| D3QN (Wang et al.) | RL target selection | Phase 5 agent (Discrete action space) |
| TD3 (Chen et al.) | RL velocity control | Phase 5 agent (Continuous action space) |
| Hierarchical RL | Proposed hybrid | Phase 5 composite agent |

## 8.5 Required Evaluation Metrics

The following metrics must be computed for every run and reportable in IEEE format:

| Metric | Symbol | Unit | Implementation |
|---|---|---|---|
| Data Collection Ratio | $\eta_{data}$ | % | `MetricEngine.data_collection_rate` ✅ |
| Mission Completion Time | $T_{mission}$ | s | `temporal.current_time` at termination |
| Average Age of Information | $\bar{A}$ | s | `MetricEngine.average_aoi` ✅ |
| Energy Efficiency | $\eta_{energy}$ | Mb/J | `collected_data / energy_consumed` |
| Coverage Ratio | $\eta_{cover}$ | % | `MetricEngine.coverage_ratio` ✅ |
| Collision Rate | $r_{coll}$ | count | `MetricEngine.collision_rate` ✅ |
| Deadline Satisfaction | $\eta_{deadline}$ | % | `MetricEngine.deadline_violations` ✅ |
| Network Lifetime | $T_{net}$ | s | `MetricEngine.network_lifetime` ✅ |
| Path Stability Index | $PSI$ | [0,1] | `MetricEngine.path_stability_index` ✅ |
| Replan Frequency | $f_{replan}$ | count/step | `MetricEngine.replan_frequency` ✅ |

## 8.6 Output Artifacts for Publication

Each simulation run must produce:

```
visualization/runs/<run_id>/
├── logs/
│   ├── run_summary.json        ← all metrics
│   └── config_snapshot.json    ← full config for reproducibility
├── telemetry/
│   ├── step_telemetry.csv      ← per-step UAV state
│   └── node_state.csv          ← final sensor states
├── frames/
│   └── *.png                   ← trajectory snapshots
├── plots/
│   ├── environment.png
│   ├── dashboard_panel.png
│   ├── radar_chart.png
│   ├── trajectory_heatmap.png
│   ├── aoi_timeline.png
│   ├── battery_with_replans.png
│   ├── node_energy_heatmap.png
│   ├── 3d_trajectory.png
│   └── semantic_clustering.png (if enabled)
├── animations/
│   └── mission.mp4 (if generated)
└── reports/
    └── ieee_report.md (auto-generated)
```

---

## Appendix A — Pre-Refactoring Baseline Capture

Before Stage 1 begins, the implementation engineer must run:

```bash
cd "/Users/ramayan/Downloads/BTP-Major Project/Code/IOT Based/Phase 1"
python3 main.py
# Copy the run_summary.json to a stable location
cp visualization/runs/$(ls -t visualization/runs/ | head -1)/logs/run_summary.json \
   docs/metrics/baseline_pre_refactor.json
git add docs/metrics/baseline_pre_refactor.json
git commit -m "docs(metrics): save pre-refactor baseline metrics snapshot"
git push origin main
```

This baseline is the reference for all "within ±N%" verification checks.

## Appendix B — Stage Dependency Graph

```
     ┌─────────┐
     │ Stage 1 │  Entity Model Separation
     │  (D2)   │
     └────┬────┘
          │
     ┌────▼────┐
     │ Stage 2 │  Visualization Decoupling
     │  (D4)   │
     └────┬────┘
          │
     ┌────▼────┐
     │ Stage 3 │  Continuous Simulation Clock
     │  (D6)   │
     └────┬────┘
          │
     ┌────▼────┐
     │ Stage 4 │  Analytical Service Model
     │  (D3)   │
     └────┬────┘
          │
     ┌────▼────┐
     │ Stage 5 │  MissionController Decomposition
     │  (D1)   │
     └────┬────┘
          │
     ┌────▼────┐
     │ Stage 6 │  True 3D Collision Detection
     │  (D5)   │
     └─────────┘
          │
          ▼
   Phase 5 RL Ready
```

## Appendix C — Risk Mitigation

| Risk | Mitigation |
|---|---|
| Stage 1 breaks everything (widespread Node references) | Use IDE rename refactoring; run tests after each file change; keep deprecated `Node` alias temporarily |
| Stage 4 changes metric values | Expected and correct; document the deltas in commit message |
| Stage 5 introduces circular imports | PhysicsEngine must NOT import MissionController; use dependency injection |
| Visualization breaks after Stage 2 | Test `python3 main.py` with `--render` immediately after Stage 2 |
| 3D collision breaks pathfinding in Stage 6 | UAV path should be less constrained (fewer false positives); monitor coverage_ratio for unexpected drops |

---

*End of Refactoring Program*
