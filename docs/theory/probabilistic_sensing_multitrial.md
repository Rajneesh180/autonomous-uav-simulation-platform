# Multi-Trial Probabilistic Sensing & Minimum Hover Time

**Source**: Zheng & Liu, *"3D UAV Trajectory Planning With Obstacle Avoidance for UAV-Enabled Time-Constrained Data Collection"*, IEEE TVT, Jan 2025 — Section III-C, Equations 4–6.

**Implemented in**: `core/communication.py` — `required_sensing_trials()`, `minimum_hover_time()`

---

## 1. Motivation

A single snapshot sensing trial has probability `P_suc = e^{-τ*d}` of successfully recovering a data packet. At long distances this probability is very low, so the UAV must hover for **multiple trial slots** until the cumulative success probability exceeds a threshold ω. The minimum hover time is derived analytically — not guessed — from the channel model.

The previous codebase used a single-trial Bernoulli draw (`random.random() ≤ exp(-τ*d)`), which underestimates required hover time at far nodes.

---

## 2. Mathematical Formulation

### Single-Trial Success Probability
```
P_suc(d) = e^{-τ * d}
```
where `τ` is the decay parameter (`Config.SENSING_TAU`) and `d` is the 3D distance between UAV and node.

### Multi-Trial Cumulative Probability (n_s trials)
```
P_suc^(n_s) = 1 - (1 - e^{-τ*d})^{n_s}
```
This converges to 1 as n_s → ∞.

### Minimum Required Trials
To achieve cumulative success probability ≥ ω (target reliability):
```
n̂_s = ⌈ log(1 - ω) / log(1 - e^{-τ*d}) ⌉      [Eq. 6]
```

### Minimum Hover Time
```
T_hover_min(d) = n̂_s × T_s      [seconds]
```
where `T_s` is the duration of each sensing slot (`Config.SENSING_SLOT_DURATION`).

---

## 3. Algorithm Pseudocode

```
Algorithm: Required_Hover_Time(d, tau, omega, T_s)
──────────────────────────────────────────────────
Input : d     — UAV ↔ node distance (m)
        tau   — sensing decay parameter
        omega — target cumulative success probability (e.g. 0.95)
        T_s   — sensing slot duration (s)
Output: T_hover — minimum hover time (s), n_s — required trials

1. p_single = exp(-tau * d)
2. If p_single >= omega:
       n_s = 1   (single trial suffices)
3. Else if p_single <= 0:
       n_s = ∞   (channel unreachable)
4. Else:
       n_s = ceil( log(1 - omega) / log(1 - p_single) )
5. T_hover = n_s * T_s
6. Return (n_s, T_hover)
──────────────────────────────────────────────────
```

---

## 4. Integration Flowchart

```
UAV arrives at RP hovering position
            │
            ▼
   Compute d = dist(UAV, Node)
            │
            ▼
   n̂_s = required_sensing_trials(d, τ, ω)
            │
            ▼
   T_hover_min = n̂_s × T_s
            │
   Hover for ≥ T_hover_min steps
            │
            ▼
   Data collected → mark visited
```

---

## 5. Parameters

| Symbol | Config Key | Default | Meaning |
|--------|-----------|---------|---------|
| τ | `SENSING_TAU` | 0.05 | Channel decay rate |
| ω | `SENSING_OMEGA` | 0.95 | Target success probability |
| T_s | `SENSING_SLOT_DURATION` | 1.0 s | Duration per sensing trial |
| n̂_s | — | computed | Minimum required trials |

---

## 6. Expected Impact

At d = 50 m: single trial P_suc = e^{-0.05 × 50} = 0.082. To reach ω = 0.95, n̂_s = ⌈log(0.05)/log(1-0.082)⌉ = **27 trials**. Without this model, the UAV frequently under-hovers, leaving nodes partially unserviced.
