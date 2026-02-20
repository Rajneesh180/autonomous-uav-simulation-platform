# System Formulation

This section formalizes the UAV simulation as a discrete-time dynamical system.

The formulation is phase-consistent and extends naturally to learning-based control (Phase-5).

---

# 1. System State

At time step t, the complete system state is defined as:

S(t) = { P_u(t), B(t), N(t), O(t), R(t) }

Where:

P_u(t) ∈ ℝ²      → UAV position  
B(t) ∈ ℝ⁺        → Remaining battery energy  
N(t)             → Active node set  
O(t)             → Obstacle set  
R(t)             → Risk field  

---

# 2. State Transition Model

System evolution is governed by:

S(t+1) = F(S(t), a(t), ξ(t))

Where:

a(t)     → Control action selected by planner  
ξ(t)     → Environmental stochastic events (node churn, obstacle motion)

The transition function F consists of:

1. UAV motion update  
2. Energy update  
3. Obstacle motion update  
4. Node set update  
5. Risk field update  

---

# 3. UAV Motion Model

Given control action a(t):

P_u(t+1) = P_u(t) + Δt · V_u(t)

Where velocity V_u(t) is determined by selected steering primitive.

Movement is bounded by:

‖V_u(t)‖ ≤ V_max

---

# 4. Energy Dynamics

Energy consumption for movement:

E_move(t) = c_e · d(t) · ρ(P_u(t), t)

Where:

c_e       → Energy per meter  
d(t)      → Distance traveled at step t  
ρ(p, t)   → Risk multiplier  

Battery update:

B(t+1) = B(t) - E_move(t) - E_hover(t)

Mission terminates if:

B(t) ≤ B_min

---

# 5. Obstacle Dynamics

Each obstacle:

o_j(t) = (x₁, y₁, x₂, y₂, v_x, v_y)

Linear motion model:

x₁(t+1) = x₁(t) + v_x  
x₂(t+1) = x₂(t) + v_x  
y₁(t+1) = y₁(t) + v_y  
y₂(t+1) = y₂(t) + v_y  

Velocity magnitude scaled by hostility profile.

---

# 6. Node Set Evolution

Node set evolves via:

N(t+1) = N(t)
         ∪ Spawn(t)
         \ Remove(t)

Spawn(t) triggered by interval condition.

Remove(t) triggered probabilistically with minimum floor constraint.

---

# 7. Risk Field Evolution

Risk multiplier:

ρ(p, t) = ρ_base(p) + δρ(t)

Where δρ(t) may vary temporally.

Currently deterministic, extensible to stochastic.

---

# 8. Replanning Trigger Function

Replanning occurs if:

𝒯(t) = 1

Where:

𝒯(t) = 1 if any of:

- Node set changed
- Collision detected
- Energy risk threshold exceeded
- Path invalidated by obstacle
- Environmental condition triggered

Replanning cooldown enforces:

t - t_last_replan ≥ τ

---

# 9. Control Objective (Phase-3)

Maximize mission coverage while maintaining feasibility:

Maximize:

|Visited Nodes|

Subject to:

B(t) > B_min  
Collision avoidance  
Dynamic feasibility  

No semantic weighting yet (introduced in Phase-4).

---

# 10. Determinism & Reproducibility

Given fixed seed:

S(0) deterministic  
ξ(t) deterministic  

Thus system trajectory is reproducible.

---

This formalization defines the simulator as a controlled stochastic dynamical system, forming the basis for semantic extension (Phase-4) and policy optimization (Phase-5).
