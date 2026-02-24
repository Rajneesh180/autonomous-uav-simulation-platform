# Genetic Algorithm Visiting Sequence Optimizer — Theory & Implementation

**Source**: Zheng & Liu, *"3D UAV Trajectory Planning With Obstacle Avoidance for Time-Constrained Data Collection"*, IEEE TVT, Jan 2025 — Section III-D, Algorithm 2.

**Implemented in**: `path/ga_sequence_optimizer.py`

---

## 1. Motivation

After Rendezvous Point selection reduces the node set to |R| RPs, the UAV must solve an **ordered visiting sequence** problem: which order to visit RPs minimises path length (energy) subject to time-window constraints `[e_j, l_j]`?

The paper adopts a **Genetic Algorithm (GA)** because the problem is NP-hard (TSP variant). The existing PCA-GLS heuristic provides a good initial seed, which the GA then refines.

---

## 2. Mathematical Formulation

### Objective function
```
min  C(π) = Σ_{k=0}^{|R|-1} dist(rp_{π(k)}, rp_{π(k+1)})
```
subject to:
```
e_j ≤ t_j ≤ l_j − s_j    ∀ j ∈ R    (time-window constraints)
```

### Individual encoding
Each **chromosome** is a permutation `π` of RP indices `{0, …, |R|−1}`.

### Fitness
```
f(π) = 1 / (1 + C(π) + λ_tw · V(π))
```
where V(π) is the total time-window **violation penalty**.

---

## 3. GA Algorithm Pseudocode

```
Algorithm: GA_Sequence(R, N_pop, G_max, p_c, p_m)
───────────────────────────────────────────────────
Input : R   = RP node set
        N_pop   = population size
        G_max   = max generations
        p_c, p_m = crossover, mutation rates
Output: best permutation π*

1.  Init: generate N_pop random permutations; seed one with PCA-GLS order
2.  Evaluate: f(π) for all π in population
3.  For g = 1..G_max:
      parents = tournament_select(population, k=3)
      offspring = ordered_crossover(parents[0], parents[1])  // OX operator
      If rand() < p_m: offspring = swap_mutation(offspring)
      Enforce time-window feasibility repair
      population = elitism_replace(population, offspring)
4.  Return argmax f(π) in population
───────────────────────────────────────────────────
Complexity: O(G_max × N_pop × |R|)
```

---

## 4. Integration Flowchart

```
RP Selection → |R| RPs
       │
PCA-GLS seed
       │
GA refine (G_max generations)
       │
 Best permutation π*
       │
MissionController.target_queue = [rp_{π*(0)}, ..., rp_{π*(|R|-1)}]
```

---

## 5. Parameters

| Parameter | Config key | Default |
|-----------|-----------|---------|
| Population size | `GA_POPULATION_SIZE` | 30 |
| Max generations | `GA_MAX_GENERATIONS` | 50 |
| Crossover rate | `GA_CROSSOVER_RATE` | 0.85 |
| Mutation rate | `GA_MUTATION_RATE` | 0.15 |
| TW penalty weight | `GA_TW_PENALTY_WEIGHT` | 5.0 |
