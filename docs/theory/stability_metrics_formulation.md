# Stability & Adaptation Metrics Derivation

This section formalizes the evaluation metrics introduced in Phase-3.

Metrics quantify system robustness under dynamic environmental conditions.

---

# 1. Replan Frequency (RF)

Let:

𝒯_eff(t) ∈ {0,1}

be the effective replanning trigger at time t.

Total simulation duration:

T

Replan Frequency:

RF = (1 / T) ∑_{t=0}^{T} 𝒯_eff(t)

Interpretation:

Low RF → stable planning  
High RF → volatile environment or weak planner  

---

# 2. Collision Rate (CR)

Let:

C(t) ∈ {0,1}

be collision indicator at time t.

Collision Rate:

CR = (1 / T) ∑_{t=0}^{T} C(t)

Stability target (medium hostility):

CR < 0.10

---

# 3. Adaptation Latency (AL)

Define:

t_event = time environmental event occurs  
t_replan = time replanning begins  

Adaptation Latency:

AL = t_replan - t_event

Current implementation:

Immediate trigger → AL ≈ 0

Future versions may introduce delayed detection.

---

# 4. Path Stability Index (PSI)

Let:

L_i = length of path after replan i  
N_r = total number of replans  

Define average path change magnitude:

ΔL_avg = (1 / N_r) ∑ |L_i - L_{i-1}|

Normalize:

PSI = - ΔL_avg

Interpretation:

PSI ≈ 0 → stable path  
Large negative PSI → high volatility  

---

# 5. Node Churn Impact (NCI)

Let:

N_spawn = total spawned nodes  
N_remove = total removed nodes  

Node churn magnitude:

N_churn = N_spawn + N_remove

Normalize by replans:

NCI = N_churn / (1 + N_r)

Interpretation:

Measures environmental instability relative to planner adaptability.

---

# 6. Coverage Progress Rate (CPR)

Let:

V(t) = number of visited nodes at time t  

Coverage rate:

CPR = V(T) / T

Higher CPR indicates efficient adaptation.

---

# 7. Energy Efficiency Ratio (EER)

Let:

E_total = total energy consumed  
V_total = total visited nodes  

EER = V_total / E_total

Measures mission productivity per energy unit.

---

# 8. Stability Region Definition

A Phase-3 system is considered stable if:

CR ≤ 0.10  
RF ≤ 0.08  
PSI ≈ 0  
Return constraint never violated  

---

# 9. Metric Interdependency

High hostility → higher churn  
Higher churn → higher RF  
Higher RF → lower PSI  
Lower PSI → path volatility  

Thus system quality must be evaluated holistically.

---

# 10. Limitations

- Metrics are aggregate
- No variance analysis yet
- No statistical confidence intervals
- No multi-run averaging

Batch evaluation required for rigorous validation.

---

These metrics transform the simulator from a visual demo into a measurable adaptive system.
