# Phase-3 Experiment Protocol
Dynamic Environment & Real-Time Adaptation Evaluation

This document defines the standardized evaluation framework for Phase-3.

All experiments must follow this protocol for reproducibility and academic validity.

---

# 1. Objective

Evaluate system stability, adaptability, and mission feasibility under dynamic environmental conditions.

Primary research question:

Can the UAV maintain mission performance under temporal uncertainty?

---

# 2. Experimental Variables

## 2.1 Controlled Variables

- MAP_WIDTH
- MAP_HEIGHT
- NODE_COUNT (initial)
- RANDOM_SEED (fixed per run)
- BATTERY_CAPACITY
- UAV_STEP_SIZE

These remain constant across comparative runs.

---

## 2.2 Independent Variables

1. Hostility Level:
   - Low
   - Medium
   - High

2. Moving Obstacles:
   - Disabled
   - Enabled

3. Dynamic Node Spawn Interval:
   - 10
   - 15
   - 20

4. Node Removal Probability:
   - 0.10
   - 0.15
   - 0.25

---

# 3. Dependent Metrics

The following metrics are recorded:

- Replan Frequency (RF)
- Collision Rate (CR)
- Adaptation Latency (AL)
- Path Stability Index (PSI)
- Node Churn Impact (NCI)
- Coverage Progress Rate (CPR)
- Energy Efficiency Ratio (EER)
- Final Battery

---

# 4. Experimental Duration

Each simulation runs for:

MAX_TIME_STEPS = 400

Termination occurs if:

- Battery depletion
- Return-to-base completion
- Time limit reached

---

# 5. Batch Execution Plan

For each hostility level:

Repeat simulation for:

N_runs = 10

Using different seeds.

Compute:

Mean ± Standard Deviation for all metrics.

---

# 6. Stability Criteria (Phase-3 Acceptance)

Under Medium Hostility:

CR ≤ 0.10  
RF ≤ 0.08  
Unsafe Return Count = 0  
Mission remains feasible  

Under High Hostility:

CR ≤ 0.25  
Return remains feasible  

---

# 7. Comparative Tables

Each batch produces table:

| Hostility | RF | CR | PSI | NCI | CPR | EER |
|-----------|----|----|-----|-----|-----|-----|

---

# 8. Visualization Requirements

For each run:

- Time vs Battery curve
- Time vs Replan events
- Time vs Collision events
- Coverage progression graph

Batch plots:

- RF vs Hostility
- CR vs Hostility
- PSI vs Hostility

---

# 9. Statistical Reporting

For each metric:

Report:

Mean  
Standard Deviation  
95% Confidence Interval  

Confidence interval:

CI = μ ± 1.96 * (σ / sqrt(N_runs))

---

# 10. Reproducibility Rules

Each run must log:

- Random seed
- Full config snapshot
- Hostility profile
- Dynamic parameters

Logs must be JSON structured.

---

# 11. Known Limitations

- No stochastic velocity variance
- Single-step obstacle prediction
- Full path recompute
- Deterministic energy model

These are documented for Phase-4/5 extensions.

---

This protocol ensures that Phase-3 evaluation is scientifically valid, reproducible, and publication-ready.
