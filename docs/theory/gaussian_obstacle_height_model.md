# 3D Gaussian Obstacle Height Model — Theory & Implementation

**Source**: Zheng & Liu, *"3D UAV Trajectory Planning With Obstacle Avoidance for Time-Constrained Data Collection"*, IEEE TVT, Jan 2025 — Section III-F, Equation 36.

**Implemented in**: `core/obstacle_model.py` — `ObstacleHeightModel` class

---

## 1. Motivation

The existing codebase models obstacles as 2D bounding rectangles with uniform height. This is sufficient for 2D routing but fails to capture the **continuous altitude profile** every physical building or terrain feature has. The UAV can fly at any z-value without being required to climb over obstacles. Zheng & Liu formalise a 3D Gaussian obstacle function that constrains hover and waypoint altitudes analytically.

---

## 2. Mathematical Model

### 3D Gaussian Obstacle Profile

```
z_obs(x, y) = Σᵢ hᵢ · exp[ -((x − xᵢ)/aˣᵢ)² − ((y − yᵢ)/aʸᵢ)² ]   [Eq. 36]
```

where:
- `hᵢ` — peak height of obstacle i (metres)
- `(xᵢ, yᵢ)` — centre coordinates of obstacle i
- `aˣᵢ, aʸᵢ` — horizontal spread parameters (standard deviation analogue)
- The sum is over all G obstacles in the environment

### UAV Altitude Constraint

At every hovering position and every waypoint, the UAV altitude must satisfy:

```
z_UAV(x, y) > z_obs(x, y) + Δz_clearance
```

where `Δz_clearance = Config.VERTICAL_CLEARANCE` (metres).

---

## 3. Algorithm Pseudocode

```
Algorithm: Altitude_Constrained_Motion_Primitive(x_new, y_new, z_new)
──────────────────────────────────────────────────────────────────────
Input : candidate position (x_new, y_new, z_new)
Output: is_valid (bool), z_safe (float)

1. z_req = z_obs(x_new, y_new) + VERTICAL_CLEARANCE
2. If z_new < z_req:
       z_safe = z_req          // force altitude up
       is_valid = True          // adjust, don't discard
   Else:
       z_safe = z_new
       is_valid = True
3. Return is_valid, z_safe
──────────────────────────────────────────────────────────────────────
```

---

## 4. Integration Flowchart

```
Generate candidate move (x_new, y_new, z_new)
              │
              ▼
   z_req = z_obs(x_new, y_new) + Δz
              │
         z_new < z_req?
        ┌─────┤
        YES   NO
        │     │
  z_new=z_req  keep z_new
        └──────┤
              │
     Accept candidate
              │
        Score + pick best
```

---

## 5. Parameters

| Parameter | Symbol | Config key | Default |
|-----------|--------|-----------|---------|
| Obstacle height | hᵢ | derived from obs height | varies |
| Horizontal spread x | aˣᵢ | `GAUSSIAN_SPREAD_X` | 40 m |
| Horizontal spread y | aʸᵢ | `GAUSSIAN_SPREAD_Y` | 40 m |
| Vertical clearance | Δz | `VERTICAL_CLEARANCE` | 10 m |

---

## 6. Gaussian vs Box Model Comparison

| Property | Box model (old) | Gaussian model (new) |
|----------|-----------------|----------------------|
| Altitude anywhere outside box | unconstrained | smoothly tapers |
| Waypoint altitude constraint | binary (inside/outside) | continuous gradient |
| Obstacle edge behaviour | hard boundary | smooth Gaussian decay |
| Paper alignment | paper does not use box | Zheng & Liu Eq. 36 ✓ |
