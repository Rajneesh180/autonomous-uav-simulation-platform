# Temporal Simulation Engine

## Purpose

Introduce continuous time evolution into UAV mission execution.

---

## Time-Step Model

Discrete simulation:

t ∈ {0, 1, 2, ..., T}

State update:

S(t+1) = F(S(t), E(t))

Where:
S(t) = UAV state + environment state
E(t) = dynamic events at time t

---

## Event Handling

Triggers:
- Node spawn
- Node removal
- Collision
- Energy risk
- Obstacle interference

Replan is conditionally triggered with cooldown enforcement.

---

## Design Rationale

Avoid full recomputation at every step.
Prevent oscillatory instability.
Ensure deterministic reproducibility via seed control.
