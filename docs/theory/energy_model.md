# Energy Consumption & Feasibility Model

This section defines the UAV energy dynamics and mission feasibility constraints.

---

# 1. Battery Model

Let:

B(t) ∈ ℝ⁺

be remaining battery energy at time t.

Initial condition:

B(0) = B_max

Mission must satisfy:

B(t) > B_min

---

# 2. Energy Components

Total energy consumption at time t:

E_total(t) = E_move(t) + E_hover(t)

---

# 2.1 Movement Energy

Let:

d(t) = ‖P_u(t+1) - P_u(t)‖

Base movement energy:

E_move_base(t) = c_e · d(t)

Where:

c_e = energy per meter constant

---

# 2.2 Risk-Weighted Energy

Risk multiplier:

ρ(P_u(t), t) ≥ 1

Then:

E_move(t) = c_e · d(t) · ρ(P_u(t), t)

Thus risky regions increase effective energy cost.

---

# 2.3 Hover Energy

When UAV does not translate:

E_hover(t) = c_h · Δt

Where:

c_h = hover drain constant

---

# 3. Battery Update Equation

B(t+1) = B(t) - E_move(t) - E_hover(t)

---

# 4. Return Feasibility Condition

Let:

D_return(t) = Euclidean distance to base

Estimated return energy:

E_return_est(t) = c_e · D_return(t)

Return trigger condition:

B(t) ≤ E_return_est(t) + B_reserve

Where:

B_reserve = RETURN_THRESHOLD · B_max

---

# 5. Energy Risk Constraint

Energy risk event occurs if:

B(t) - E_return_est(t) ≤ 0

This ensures safe return feasibility.

---

# 6. Energy Prediction Error (Future Metric)

Define predicted energy:

Ê_total

Actual energy:

E_total

Energy Prediction Error:

EPE = |E_total - Ê_total| / E_total

Currently deterministic → EPE ≈ 0

Extensible under stochastic risk or obstacle uncertainty.

---

# 7. Feasible Mission Constraint Set

The mission trajectory is feasible if:

1. B(t) > 0 ∀ t
2. Return feasible at all times
3. No collision violation

Thus feasible set:

Ω = { S(t) | energy and safety constraints satisfied }

---

# 8. Stability Perspective

Energy stability requires:

B(t) decreasing monotonically
No infeasible spike due to misprediction
Reserve threshold never violated

---

# 9. Limitations

- No aerodynamic modeling
- No acceleration cost
- No wind modeling
- Risk multiplier deterministic
- Return energy estimation linear

These simplifications maintain computational tractability.

---

This energy model ensures mission feasibility under dynamic environmental conditions and supports future stochastic extension.
