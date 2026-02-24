# TDMA Single-Node Scheduling — Theory & Implementation

**Source**: Wang et al., *"Learning-Based UAV Path Planning for Data Collection With Integrated Collision Avoidance"*, IEEE IoT, 2022 — Section III-C.

**Implemented in**: `core/buffer_aware_manager.py` — `TDMAScheduler` class

---

## 1. Motivation

In a dense IoT deployment, if multiple nodes transmit simultaneously on the same channel, their signals collide and data collection fails. The paper enforces a **Time Division Multiple Access (TDMA)** discipline: at each time step, only **one node** may transmit to the UAV — the one directly in the service queue.

The existing codebase does not enforce this: every unvisited node within range could theoretically upload simultaneously.

---

## 2. Mathematical Model

### TDMA Slot Assignment
At time step t:
```
Only node j*(t) = argmin_{j ∈ Q} dist(UAV(t), j)  transmits.
All other nodes remain silent.
```

### Achievable rate under TDMA
With N_active nodes competing and TDMA, the effective rate for node j is:
```
R_TDMA(j) = R_Shannon(j) / 1   [Mbps]   (only j transmits in that slot)
```
This is simply the full Shannon rate — TDMA gives each node its own interference-free slot.

---

## 3. Algorithm Pseudocode

```
Algorithm: TDMA_Collect(UAV, queue Q, dt)
────────────────────────────────────────────
1. If Q is empty: return 0
2. active_node = front of Q  (current mission target)
3. All other nodes: silent (no data collected)
4. data_collected = R_Shannon(active_node) * dt
5. active_node.current_buffer -= data_collected
6. If active_node.current_buffer <= 0: advance Q
7. Return data_collected
────────────────────────────────────────────
Key invariant: only ONE node's buffer drains per time step
```

---

## 4. Integration Flowchart

```
Time step t
    │
 Q = mission.target_queue
    │
 active = Q[0] (TDMA discipline)
    │
 All other nodes → silent
    │
 Collect from active_node only
    │
 Buffer drained? → remove from Q, advance
```

---

## 5. Parameters

| Parameter | Config key | Default |
|-----------|-----------|---------|
| Enforce TDMA | `ENABLE_TDMA_SCHEDULING` | True |
| Min TDMA distance | — | any (closest active node) |
