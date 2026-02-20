# Rectangle-Based Obstacle Modeling (Phase-3)

## 1. Obstacle Representation

Each obstacle is modeled as an axis-aligned rectangle:

O_i = [x1, x2] × [y1, y2]

Where:
- x1 < x2
- y1 < y2

---

## 2. UAV Position

At time t:

P_u(t) = (x_u(t), y_u(t))

---

## 3. Minimum Distance to Rectangle

dx = max(x1 − x_u, 0, x_u − x2)  
dy = max(y1 − y_u, 0, y_u − y2)

Distance:

D(P, O) = sqrt(dx² + dy²)

If UAV lies inside rectangle:
dx = 0  
dy = 0  
Distance = 0  

This corresponds to collision.

---

## 4. Collision Condition

Collision occurs when:

x1 ≤ x_u ≤ x2  
AND  
y1 ≤ y_u ≤ y2  

---

## 5. Influence Radius Penalty

If D(P, O) < R_inf:

Penalty = (R_inf − D) / R_inf

Where R_inf is obstacle influence radius.

---

## 6. Predictive Motion Model

Obstacle center evolves:

C(t+1) = C(t) + v * Δt

Rectangle boundaries shift accordingly.

---

## 7. Limitations

- Axis-aligned only
- Deterministic velocity
- No probabilistic uncertainty
- No rotated geometry
