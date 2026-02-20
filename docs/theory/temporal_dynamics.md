# Temporal Dynamics & Replanning Mechanics

This section formalizes time evolution, event triggering, and replanning stability control.

---

# 1. Discrete-Time Model

Time is discrete:

t ∈ {0, 1, 2, ..., T}

Time increment:

Δt = 1

The system evolves sequentially:

S(t+1) = F(S(t), a(t), ξ(t))

---

# 2. Event Generation Model

Environmental events are defined as:

ξ(t) = { E_spawn(t), E_remove(t), E_collision(t), E_energy(t) }

Each event is binary:

E_i(t) ∈ {0, 1}

---

## 2.1 Node Spawn Event

E_spawn(t) = 1 if:

t mod τ_spawn = 0

Where:
τ_spawn = dynamic node interval

---

## 2.2 Node Removal Event

E_remove(t) = 1 with probability:

P_remove = p_r

Subject to:

|N(t)| > N_min

---

## 2.3 Collision Event

E_collision(t) = 1 if:

C(P_u(t), O(t)) < ε

Where:

C(·) = rectangle clearance function  
ε     = collision margin  

---

## 2.4 Energy Risk Event

E_energy(t) = 1 if:

B(t) - E_return_estimate(t) ≤ 0

---

# 3. Replanning Trigger Function

Define:

𝒯(t) = 1 if any E_i(t) = 1

Otherwise:

𝒯(t) = 0

---

# 4. Cooldown Enforcement

To prevent oscillatory instability:

Replanning allowed only if:

t - t_last_replan ≥ τ_cooldown

Where:

τ_cooldown = configurable cooldown interval

Thus effective trigger:

𝒯_eff(t) = 1 if:
    𝒯(t) = 1 AND
    (t - t_last_replan ≥ τ_cooldown)

---

# 5. Adaptive Stability Constraint

Frequent replanning introduces instability.

Define replan frequency:

RF = (1/T) Σ 𝒯_eff(t)

Stable system requirement:

RF ≤ RF_max

---

# 6. Single-Step Predictive Obstacle Modeling

Clearance function:

dx = max(x₁ - x, 0, x - x₂)
dy = max(y₁ - y, 0, y - y₂)

C(P, o) = sqrt(dx² + dy²)

Predicted obstacle position:

o_j(t+1) = o_j(t) + v_j

Planner evaluates:

C(P_candidate, o_j(t+1))

This introduces anticipatory behavior without multi-step expansion.

---

# 7. Stability Interpretation

System stability depends on:

- Collision frequency
- Replan frequency
- Energy feasibility maintenance
- Coverage progression

Phase-3 stability achieved when:

CR < 0.10 (medium hostility)
RF < 0.08
PSI ≈ 0

---

# 8. Limitations of Current Temporal Model

- Only 1-step prediction
- No stochastic velocity modeling
- Full plan recomputation
- No incremental graph update
- Adaptation latency simplified

These are reserved for upper-tier refinement.

---

This formalization defines the simulator as a time-driven event-responsive autonomous system with controlled replanning behavior.
