# Stability Metrics — Phase 3

## 1. Replan Frequency (RF)

RF = Replan_Count / Total_Time_Steps

Measures system volatility.

Lower values indicate stable planning under dynamic conditions.

---

## 2. Collision Rate (CR)

CR = Collision_Count / Total_Time_Steps

Measures reactive correction load.

High values indicate insufficient anticipation.

---

## 3. Path Stability Index (PSI)

PSI = (Visited_Nodes - Replan_Count) / Visited_Nodes

Negative values indicate instability.
Positive values indicate stable progression.

---

## 4. Node Churn Impact (NCI)

NCI = Replans_Triggered_By_Node_Events / Total_Node_Events

Quantifies environmental harshness.

---

## 5. Adaptation Latency (AL)

AL = Average_Time_Between_Event_Trigger_And_Stable_Path

Currently approximate.
To be refined in Phase-3 upper tier.
