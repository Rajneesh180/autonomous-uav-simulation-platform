# Base Station Uplink Transmission Sub-Phase & AoI Data-Age Constraint

**Source**: Zheng & Liu, IEEE TVT Jan 2025 — Section III-B, Equations 24–25.

**Implemented in**: `core/base_station_uplink.py`

---

## 1. Motivation

In the reference paper, UAV mission execution follows a **Successive-Hover-Fly (SHF)** model where, after collecting data at a Data Collection Area (DCA), the UAV must fly to a dedicated **Data Transmission (DT)** hover position near the Base Station (BS) before it can offload the payload. This imposes a **data-age constraint**: the collected data must reach the BS before its AoI limit expires:

```
T_data = t_service + t_fly_to_BS ≤ T_limit        [Eq. 25]
```

The current codebase has no explicit BS uplink model — all data is assumed to vanish upon UAV return without any SNR or timing check.

---

## 2. Mathematical Model

### BS Uplink Shannon Rate

```
R_BS(p̃_k) = B · log₂( 1 + γ₀ / (‖p̃_k‖² + (H_p^k − H_b)²)^(α/2) )   [Eq. 11]
```

where:
- `p̃_k = (x_k, y_k)` — UAV horizontal offset from BS
- `H_p^k` — UAV altitude at DT position
- `H_b` — BS height (0 or antenna height)
- `α` — path loss exponent
- `γ₀` — reference SNR / channel gain

### BS Uplink Time

```
t_uplink = payload_Mbits / R_BS    [seconds]
```

### Data-Age Constraint

```
T_data = t_collect + t_fly_to_DT + t_uplink ≤ T_data_limit
```

If `T_data_limit` would be violated, the UAV must abandon the current collection route and fly to the BS immediately.

---

## 3. Algorithm Pseudocode

```
Algorithm: Check_BS_Uplink_Urgency(mission)
────────────────────────────────────────────────
Input : mission state (uav_pos, collected_data, aoi_timer, base_pos)
Output: must_uplink_now (bool)

1. For each visited node v:
       age = current_step - last_service_step[v]
       If age >= T_data_limit: must_uplink_now = True; Return True
2. payload = mission.collected_data_mbits
3. d_to_bs = dist(uav_pos, base_pos)
4. t_fly = d_to_bs / v_max
5. r_bs = R_BS(uav_pos, base_pos)
6. t_uplink = payload / r_bs  (if r_bs > 0)
7. T_data = t_fly + t_uplink
8. If T_data >= T_data_limit: Return True
9. Return False
────────────────────────────────────────────────
```

---

## 4. Integration Flowchart

```
Every N steps: check uplink urgency
              │
    must_uplink_now?
     ┌─────── YES → abort current route, fly to BS
     NO              │
     │               ▼
     │         Compute R_BS(hover pos)
     │               │
     │         Uplink payload → clear collected buffer
     │               │
     └────────── Resume mission
```

---

## 5. Parameters

| Parameter | Config key | Default |
|-----------|-----------|---------|
| T_data_limit | `BS_DATA_AGE_LIMIT` | 100 steps |
| BS SNR reference | `BS_GAMMA_0_DB` | -10 dB |
| BS path-loss exp. | `BS_PATH_LOSS_EXP` | 2.5 |
| BS antenna height | `BS_HEIGHT_M` | 5.0 m |
| Uplink check interval | `BS_UPLINK_CHECK_INTERVAL` | 10 steps |
