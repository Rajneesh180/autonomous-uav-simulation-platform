# Phase-3: Dynamic Environment & Real-Time Adaptation Layer

## 1. Objective

Phase-3 transforms the UAV mission framework from a static constrained optimizer into a temporally adaptive autonomous system.  
The system is no longer evaluated solely on feasibility, but on its ability to maintain mission performance under environmental uncertainty.

---

## 2. Architectural Transition

Phase-2:
- Static nodes
- Static obstacles
- Single-shot optimization
- Deterministic energy trajectory

Phase-3:
- Time-step simulation engine
- Dynamic node churn (spawn/removal)
- Moving obstacles
- Event-triggered replanning
- Stability evaluation metrics

This phase introduces temporal intelligence without learning-based policies.

---

## 3. Discrete-Time System Model

Let time be discretized into steps:

t ∈ {0, 1, 2, ..., T}

At each time step:

1. Environment update: E(t)
2. Event detection
3. Replan trigger evaluation
4. UAV motion update
5. Energy state update
6. Metric logging

Mission execution loop:

while mission_active:
    update_environment()
    detect_events()
    if trigger:
        replan()
    move_uav()
    update_energy()

---

## 4. Event Model

Events introduced in Phase-3:

- Node Spawn
- Node Removal
- Collision Detection
- Risk Zone Violation

Each event may trigger replanning depending on system state.

---

## 5. Replanning Strategy

Current implementation:
- Full path recomputation upon trigger

Limitations:
- No incremental graph update
- No D* Lite or LPA*
- No bounded-time replanning

This is intentionally lightweight for Phase-3 stability analysis.

---

## 6. Stability Observations (Medium Hostility Profile)

Typical metrics observed:

- Replan Frequency ≈ 0.07
- Collision Rate ≈ 0.08
- Path Stability Index < 0 (volatile regime)
- Node Churn Impact > 1.5

Negative PSI indicates replanning intensity exceeds mission progression rate.

---

## 7. Limitations

- Replanning is global recomputation
- No uncertainty modeling in obstacle velocity
- Adaptation latency measurement not strictly causal
- Simplified UAV motion model
- No probabilistic risk propagation

---

## 8. Phase-3 Outcome

The system now:

- Reacts to environmental change
- Handles moving obstacles
- Performs real-time replanning
- Computes temporal stability metrics
- Supports batch-compatible evaluation

Phase-3 elevates the simulator from a constrained planner to a reactive autonomous agent prototype.
