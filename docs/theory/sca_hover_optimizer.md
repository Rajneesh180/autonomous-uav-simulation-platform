# SCA-Inspired Hover Position Optimizer — Theory & Implementation

**Source**: Zheng & Liu, *"3D UAV Trajectory Planning With Obstacle Avoidance for Time-Constrained Data Collection"*, IEEE TVT, Jan 2025 — Section III-E (Hover Position Optimisation via SCA).

**Implemented in**: `path/hover_optimizer.py`

---

## 1. Motivation

Once the visiting sequence is fixed, the exact 3D **hover position** above each RP still needs to be chosen. The UAV could hover directly above the node's ground position, but that is suboptimal because:
- Moving obstacles may block that position
- A shifted position at the same z could give better LoS to adjacent cluster members
- Energy is wasted if the approach angle causes extra climb

SCA iteratively refines the hover position by taking gradient descent steps on a local linearisation of the non-convex objective.

---

## 2. Mathematical Formulation

### Objective per hover point R_k
```
min  f(p̃_k) = C_travel(p̃_{k-1}, p̃_k) + C_travel(p̃_k, p̃_{k+1})
             + λ_comm · R_BS(p̃_k)^{-1}    (maximise uplink rate to BS)
subject to:
    ‖p̃_k - p_node_k‖ ≤ r_hover             (stay within hover radius)
    z_k ≥ z_obs(x_k, y_k) + Δz             (altitude constraint)
```

### SCA Update (gradient descent on linearised cost)
```
p̃_k^{(i+1)} = p̃_k^{(i)} − α · ∇f(p̃_k^{(i)})
```
where `∇f` is approximated by finite differences and α is the step size.

---

## 3. Algorithm Pseudocode

```
Algorithm: SCA_Hover_Optimise(prev_pos, node, next_pos, obstacles)
────────────────────────────────────────────────────────────────────
Input : prev_pos, next_pos — adjacent waypoints
        node — target RP node
        obstacles — for altitude constraint
Output: optimised hover position p̃*

1. p̃ = (node.x, node.y, z_safe)     // initialise at node position
2. For i = 1..SCA_MAX_ITERATIONS:
     grad_x = (f(p̃ + [ε,0,0]) − f(p̃)) / ε
     grad_y = (f(p̃ + [0,ε,0]) − f(p̃)) / ε
     p̃_new = p̃ − α · (grad_x, grad_y, 0)
     Clamp: ‖p̃_new − node‖ ≤ r_hover
     Enforce: z_new = ObstacleHeightModel.enforce_altitude(p̃_new)
     If ‖p̃_new − p̃‖ < convergence_tol: Break
     p̃ = p̃_new
3. Return p̃
────────────────────────────────────────────────────────────────────
```

---

## 4. Parameters

| Parameter | Config key | Default |
|-----------|-----------|---------|
| Max iterations | `SCA_MAX_ITERATIONS` | 15 |
| Step size | `SCA_STEP_SIZE` | 3.0 m |
| Convergence tol | `SCA_CONVERGENCE_TOL` | 0.5 m |
| Hover radius | `SCA_HOVER_RADIUS` | 30.0 m |
