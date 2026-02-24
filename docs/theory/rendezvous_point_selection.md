# Rendezvous Point (RP) Selection — Theory & Algorithm

**Source**: Donipati et al. *"Optimizing UAV-Based Data Collection in IoT Networks with Dynamic Service Time and Buffer-Aware Trajectory Planning"*, IEEE TNSM, April 2025 — Section III-A, Algorithm 1.

**Implemented in**: `core/rendezvous_selector.py`

---

## 1. Motivation

In a dense IoT field of N nodes, the UAV visiting every node individually produces path complexity O(N!) for the travelling salesman sub-problem and wastes energy on closely clustered nodes. The RP selection algorithm compresses the visit set to a minimal subset R ⊆ N of **Rendezvous Points** such that every non-RP node lies within radius R_max of at least one RP and can upload its data to the RP before the UAV arrives.

---

## 2. Mathematical Formulation

### Neighbourhood Set
For each node i, define its neighbourhood:

```
N(i) = { j ∈ N | d(i, j) ≤ R_max }
```

where `d(i,j)` is the Euclidean distance between nodes i and j, and `R_max` is the maximum radio range of a ground node.

### RP Selection Objective
Find the minimum cardinality subset R such that every node in N \ R has at least one RP in its neighbourhood:

```
min |R|  subject to  ∀ j ∈ N \ R, ∃ i ∈ R : d(i,j) ≤ R_max
```

This is equivalent to a **weighted set-cover / greedy dominating-set** problem.

### Path Cost Function (time-window constrained)
```
min Σ_{i,j} C(r_i, r_j) · x_{ij}
subject to: e_j ≤ t_j ≤ l_j - s_j   (time windows)
```

---

## 3. Greedy Algorithm (Donipati et al., Algorithm 1)

```
Algorithm: Greedy_RP_Selection(N, R_max)
─────────────────────────────────────────────────────────────
Input : Set of IoT nodes N, coverage radius R_max
Output: Set of Rendezvous Points R ⊆ N

1. Compute |N(i)| = |{ j ∈ N | d(i,j) ≤ R_max }| for all i ∈ N
2. Sort nodes by |N(i)| descending
3. R ← ∅  ;  covered ← ∅
4. While N \ covered ≠ ∅:
     i* ← argmax_{i ∈ N \ covered} |N(i)|
     R ← R ∪ {i*}
     covered ← covered ∪ N(i*)
5. Return R
─────────────────────────────────────────────────────────────
Complexity: O(|N|²) from pairwise neighbourhood computation
```

### Pseudocode (Python-style)
```python
def select_rendezvous_points(nodes, r_max):
    remaining = set(node.id for node in nodes)
    rp_ids = []
    while remaining:
        # pick node with most uncovered neighbours
        best = max(remaining,
                   key=lambda i: count_neighbours(i, remaining, r_max))
        rp_ids.append(best)
        # remove best and all its neighbours from remaining
        remaining -= {j for j in remaining if dist(best, j) <= r_max}
    return rp_ids
```

---

## 4. Integration Flowchart

```
All IoT nodes N
       │
       ▼
 Compute pairwise distances d(i,j)
       │
       ▼
 Greedy RP Selection  →  R = {rp_1, rp_2, ..., rp_k}
       │
       ▼
 Non-RP nodes upload locally to nearest RP
       │
       ▼
 PCA-GLS routes UAV only through R  (k << N)
       │
       ▼
 BufferAwareManager collects at each RP
```

---

## 5. Parameters

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Coverage radius | R_max | `Config.RP_COVERAGE_RADIUS` (default 120 m) | Donipati et al. Table III |
| Min neighbours | — | 1 (any covered node qualifies) | Algorithm 1 |

---

## 6. Expected Impact

- **Path length**: ~40–60% reduction in UAV stops (50 nodes → ~12–18 RPs)
- **Energy**: Lower total propulsion energy E_p due to fewer waypoints
- **Network lifetime**: Fewer TX events per node as non-RP nodes transmit only to adjacent RP, not directly to UAV
