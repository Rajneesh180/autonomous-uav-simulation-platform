# Agent-Centric Coordinate Transform — Theory & Implementation

**Source**: Wang et al., *"Learning-Based UAV Path Planning for Data Collection With Integrated Collision Avoidance"*, IEEE IoT, 2022 — Section III-B, Table I, Eq. 8–10.

**Implemented in**: `core/agent_centric_transform.py`

---

## 1. Motivation

The D3QN learning agent observes the world in a **UAV-body-frame** coordinate system centred at the current UAV position and aligned with its current heading (yaw). This ensures the state representation is **translation and rotation invariant**, which is critical for generalisation across different mission start positions and heading angles.

Without this transform, raw global (x, y) coordinates make the policy non-transferable between missions.

---

## 2. Mathematical Formulation

For a UAV at position `(x_u, y_u)` with heading `ψ_u`, any world-frame point `p_w = (x_w, y_w)` is expressed in the agent-centric frame as:

```
Δx = x_w - x_u
Δy = y_w - y_u

[p_ac_x]   [ cos(ψ_u)   sin(ψ_u) ] [Δx]
[p_ac_y] = [-sin(ψ_u)   cos(ψ_u) ] [Δy]
```

For a 3D extension (using yaw only for lateral rotation):
```
p_ac_z = z_w - z_u
```

---

## 3. State Vector Composition (Wang et al., Table I)

The full normalised state vector s_t for the D3QN agent:

| Feature | Value | Notes |
|---------|-------|-------|
| Agent-centric target x | p_ac_x / d_norm | normalised by map diagonal |
| Agent-centric target y | p_ac_y / d_norm | |
| Target-z altitude delta | Δz / H_norm | normalised |
| Remaining buffer | node.current_buffer / node.buffer_capacity | |
| Node priority | node.priority / P_max | |
| Residual battery | uav.current_battery / uav.battery_capacity | |
| Elapsed time fraction | t / T_max | |

---

## 4. Algorithm Pseudocode

```
Algorithm: Agent_Centric_Transform(uav, target, env)
─────────────────────────────────────────────────────
Input : uav    = UAV node (position, yaw)
        target = target node (position, buffer, priority)
        env    = Environment (map size)
Output: state_vector s_t (normalised)

1. Δx = target.x - uav.x
   Δy = target.y - uav.y
2. p_ac_x = Δx * cos(ψ) + Δy * sin(ψ)
   p_ac_y = -Δx * sin(ψ) + Δy * cos(ψ)
3. d_norm = sqrt(env.width² + env.height²)
4. Normalise: p̂_ac = (p_ac_x/d_norm, p_ac_y/d_norm)
5. state = [p̂_ac_x, p̂_ac_y, Δz/H, buffer/cap, priority/P_max, battery/bat_cap, t/T_max]
6. Return state
─────────────────────────────────────────────────────
```

---

## 5. Integration Flowchart

```
Each RL step
     │
     ▼
Create agent-centric state s_t
     │
   D3QN action a_t
     │
Apply motion primitive
     │
     ▼
Observe next state s_{t+1}
     │
Compute reward r_t
```
