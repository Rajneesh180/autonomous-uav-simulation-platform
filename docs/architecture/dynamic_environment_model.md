# Dynamic Environment Model

## Purpose

Model environmental uncertainty in UAV-based IoT data collection.

Phase-3 introduces dynamic environmental entities:

- Moving obstacles
- Node churn
- Risk fluctuation

---

## Environment State Representation

Environment(t) =
{
  Nodes(t),
  Obstacles(t),
  RiskZones(t)
}

---

## Node State

Node = {
  id,
  x,
  y,
  dynamic_attributes
}

Nodes may be added or removed over time.

---

## Obstacle State

Obstacle = {
  x1, y1,
  x2, y2,
  vx, vy
}

Movement is deterministic linear.

Future models may introduce:
- Random walk
- Acceleration
- Stochastic velocity noise

---

## Risk Zone Model

Risk multiplier applied to energy cost.

Energy_effective = distance × risk_multiplier

Risk(t) may vary per time step.

---

## Hostility Profiles

Hostility controls:

- Obstacle velocity scale
- Node spawn interval
- Node removal probability
- Prediction horizon
- Steering aggressiveness

This creates a robustness spectrum.

---

## Determinism & Reproducibility

Global random seed ensures identical runs for:

- Node spawn
- Node removal
- Obstacle motion
- Steering decision

Critical for research validation.
