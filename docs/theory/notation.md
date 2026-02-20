# Mathematical Notation

This document defines all symbols used throughout the system formulation.
Notation is designed to remain consistent through Phase-5 (Learning Layer).

---

## Time

t ∈ ℕ  
Discrete simulation time step.

T  
Total simulation horizon.

Δt  
Time resolution (default = 1).

---

## UAV State

P_u(t) ∈ ℝ²  
UAV position at time t.

B(t) ∈ ℝ⁺  
Battery energy remaining at time t.

V_u(t) ∈ ℝ²  
UAV velocity vector.

a(t)  
Control action (steering primitive selected at time t).

---

## Environment State

S(t)  
Full system state at time t.

S(t) = { P_u(t), B(t), N(t), O(t), R(t) }

---

## Nodes

N(t) = { n₁(t), n₂(t), …, n_k(t) }

Each node:

n_i(t) = (p_i, φ_i)

Where:
p_i ∈ ℝ²  → spatial coordinate  
φ_i       → feature vector (Phase-4 onward)

---

## Obstacles

O(t) = { o₁(t), o₂(t), … }

Each obstacle:

o_j(t) = (x₁, y₁, x₂, y₂, v_x, v_y)

Axis-aligned rectangular obstacle with velocity.

---

## Risk Field

R(t): ℝ² → ℝ⁺  
Spatial risk multiplier function.

ρ(p, t) = risk multiplier at position p and time t.

---

## Planning Variables

π(t)  
Planned path at time t.

𝒯(t)  
Replan trigger function.

---

## Metrics

RF  
Replan frequency.

CR  
Collision rate.

PSI  
Path stability index.

NCI  
Node churn impact.

AL  
Adaptation latency.

---

## Phase-4 Preview Symbols

F_i  
Feature vector for node i.

w  
Feature weight vector.

D(·)  
Weighted distance metric.

---

All subsequent formulations must strictly use this notation.
