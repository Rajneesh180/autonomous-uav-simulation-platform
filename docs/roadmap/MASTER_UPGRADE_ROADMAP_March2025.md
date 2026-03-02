# MASTER UPGRADE ROADMAP — March 2–10, 2025

## Strategic Context

| Milestone | Date | Objective |
|-----------|------|-----------|
| **Code Freeze** | March 10 | All Phase 1–4 upgrades merged, reproducible 10-run batch results |
| **Mid-Term BTP Presentation** | March 11–12 | Present research-grade Phase 1–4 platform + roadmap for Phase 5–10 |
| **Phase 5 (RL) begins** | March 13 | D3QN/TD3 integration on top of strengthened classical foundation |
| **Final BTP** | May 12 | Complete Phase 5–10 with RL + extensions |
| **IEEE Paper** | Post-May | Publish IEEE Transaction paper combining classical + RL + novel contributions |

---

## SECTION A: CRITICAL BUGS & INTEGRATION GAPS (Days 1–2)

These are **shipped code that is broken or disconnected** — the highest priority because they represent gaps between what the reference papers demand and what actually runs.

---

### A1. Wire HoverOptimizer into Mission Pipeline ⚡ CRITICAL

**File:** `core/mission_controller.py` → `_recompute_plan()` (line ~330)
**File:** `path/hover_optimizer.py` (line 178 — `apply_sequence()` static method)

**Problem:** `HoverOptimizer` is imported at line 17 but **never called** anywhere in the pipeline. The reference paper (Zheng & Liu, IEEE TVT 2025) explicitly requires SCA-based hover position refinement *after* sequence optimization. The GA produces a node visit order, but the hover positions (where the UAV actually stops above each node) are never optimized.

**Fix (~8 lines):**
```python
# In _recompute_plan(), AFTER the GA/PCA-GLS block, BEFORE cache write-back:
if Config.ENABLE_SCA_HOVER and self.target_queue:
    self.target_queue = HoverOptimizer.apply_sequence(
        self.target_queue,
        self.env.obstacles,
        getattr(self.env, 'risk_zones', [])
    )
    print(f"[SCA Hover] Refined {len(self.target_queue)} hover positions")
```

**Impact:** Directly improves energy efficiency (J/node), path quality, and AoI by optimizing hover positions to maximize SNR and minimize propulsion cost. This is a **paper-mandated** optimization that's fully coded but disconnected.

**Effort:** 15 minutes | **Risk:** Zero (module already tested standalone)

---

### A2. Implement Full GLS Penalty Augmentation ⚡ CRITICAL

**File:** `path/pca_gls_router.py` (lines 72–76 — currently commented out)

**Problem:** The entire GLS (Guided Local Search) penalty mechanism from Donipati et al. (IEEE TNSM 2025) is **commented out** with the note "For simulation scale, the intelligent PCA sequence suffices." Only a basic 2-opt local search runs. GLS is the core contribution of the PCA-GLS router — without it, this is just PCA-greedy + 2-opt, which is a standard TSP heuristic, **not a research contribution**.

**Fix:** Implement the full GLS penalty matrix per the Donipati reference:
```python
def _gls_refine(route, penalty_matrix, lambda_gls=0.3, max_iters=20):
    """
    Guided Local Search: augment edge cost by λ * p_ij * c_ij / (1 + p_ij)
    to escape local optima. On each iteration:
    1. Run 2-opt on augmented costs
    2. Penalize the edge with max utility = c_ij / (1 + p_ij)
    3. Repeat until no improvement or max_iters
    """
    # Initialize penalty matrix p_ij = 0
    # Iterate: augmented_cost(i,j) = dist(i,j) + lambda * p[i][j]
    # After 2-opt converges: find max-utility edge, increment p[i][j]
    # Re-run 2-opt with new penalties
```

**Impact:** This transforms PCA-GLS from a standard heuristic into the actual research-grade algorithm from the reference paper. Essential for IEEE paper claim of implementing Donipati et al.'s approach.

**Effort:** 2 hours | **Risk:** Low (2-opt already works, GLS wraps it)

---

### A3. Replace Hardcoded 1.5× AoI Boost with Configurable Weight

**File:** `core/clustering/semantic_clusterer.py` (line 56)

**Problem:** `aoi_urgency * 1.5` is hardcoded — no theoretical justification for the multiplier. For a research paper, feature importance weights must be either: (a) tunable hyperparameters, (b) learned from data, or (c) derived from mathematical analysis.

**Fix:**
```python
# In config/config.py:
AOI_URGENCY_WEIGHT = 1.5          # Semantic feature importance for AoI urgency
PRIORITY_WEIGHT = 1.0             # Weight for node priority in feature vector
BUFFER_WEIGHT = 1.0               # Weight for buffer utilization

# In semantic_clusterer.py:
features.append(Config.AOI_URGENCY_WEIGHT * aoi_urgency)
```

**Impact:** Makes the clustering scientifically defensible. Can run sensitivity analysis across weight ranges for the presentation.

**Effort:** 30 minutes | **Risk:** Zero

---

## SECTION B: ALGORITHMIC DEPTH UPGRADES (Days 2–4)

These upgrades transform "implemented but shallow" features into research-grade algorithms.

---

### B1. Integrate Advanced Clustering into Mission Pipeline

**File:** `core/clustering/semantic_clusterer.py`, `core/clustering/distribution_clustering.py`

**Problem:** `distribution_clustering.py` already has 7 algorithms (KMeans, DBSCAN, OPTICS, Agglomerative, GMM, Fuzzy C-Means, Soft K-Means) with silhouette/DB/CH metrics — but **none** are wired into the actual mission. The live pipeline (`semantic_clusterer.py`) only uses KMeans or DBSCAN, chosen by a config string.

**Upgrade Plan:**
1. **Auto-Select Best Algorithm:** Add silhouette-score-based algorithm selection 
   - Try KMeans, DBSCAN, GMM on the normalized feature matrix
   - Pick the one with highest silhouette score
   - Fall back to user config choice if all fail
   
2. **Adaptive K Selection (for KMeans/GMM):**
   ```python
   from sklearn.metrics import silhouette_score
   best_k, best_score = 2, -1
   for k in range(2, min(10, len(nodes)//3)):
       labels = KMeans(k, n_init=5).fit_predict(X_reduced)
       score = silhouette_score(X_reduced, labels)
       if score > best_score:
           best_k, best_score = k, score
   ```

3. **Wire GMM as an option in `semantic_clusterer.py`:**
   ```python
   elif Config.CLUSTER_ALGO_MODE == "gmm":
       from sklearn.mixture import GaussianMixture
       gmm = GaussianMixture(n_components=best_k, covariance_type='full')
       labels = gmm.fit_predict(X_reduced)
       centroids = gmm.means_
   ```

4. **Quality-Based Reclustering in `cluster_manager.py`:**
   Replace the simple threshold (change of 5+ nodes) with:
   ```python
   def should_recluster(self, n_nodes, current_silhouette=None):
       if abs(n_nodes - self._last_node_count) >= self.recluster_threshold:
           return True
       if current_silhouette is not None and current_silhouette < 0.3:
           return True  # Cluster quality degraded
       return False
   ```

**Impact:** Makes Phase 4 "Semantic Intelligence" genuinely intelligent rather than hardcoded. Provides ablation study material for the paper (KMeans vs. GMM vs. DBSCAN across different distributions).

**Effort:** 4 hours | **Risk:** Medium (needs testing across all dataset modes)

---

### B2. Feature Importance Weighting in Clustering

**File:** `core/clustering/feature_scaler.py`, `core/clustering/semantic_clusterer.py`

**Problem:** All 10 semantic features are treated equally (after minmax/zscore). In reality, AoI urgency and buffer utilization should dominate spatial features for IoT data freshness optimization.

**Upgrade:**
```python
# In Config:
SEMANTIC_FEATURE_WEIGHTS = {
    'x': 0.5, 'y': 0.5, 'z': 0.3,
    'priority': 1.2, 'risk': 0.8, 'signal': 0.6,
    'deadline': 1.0, 'buffer': 1.3, 'reliability': 0.7, 'aoi': 1.5
}

# In semantic_clusterer.py, after PCA reduction:
# Apply diagonal weight matrix before clustering
W = np.diag([Config.SEMANTIC_FEATURE_WEIGHTS[k] for k in feature_names])
X_weighted = X_normalized @ W
```

**Impact:** Justifies the "semantic" in "Semantic Clustering" — features are weighted by domain importance. Can cite this as a contribution in the paper.

**Effort:** 1.5 hours | **Risk:** Low

---

### B3. Weighted Multi-Objective Fitness for GA

**File:** `path/ga_sequence_optimizer.py`

**Current State:** GA fitness is primarily distance-based with a time-window penalty. The reference paper (Zheng & Liu, TVT 2025) optimizes a multi-objective: minimize total completion time T + energy consumption E while satisfying time windows.

**Upgrade:** Extend fitness function:
```python
def _fitness(self, chromosome):
    total_dist = self._route_distance(chromosome)
    energy_cost = sum(EnergyModel.propulsion_power(v_segment) for ...)
    tw_violations = self._count_tw_violations(chromosome)
    aoi_penalty = sum(node.aoi_timer for node in chromosome) / len(chromosome)
    
    return -(
        Config.GA_DIST_WEIGHT * total_dist +
        Config.GA_ENERGY_WEIGHT * energy_cost +
        Config.GA_TW_PENALTY_WEIGHT * tw_violations +
        Config.GA_AOI_WEIGHT * aoi_penalty
    )
```

**Impact:** Makes the GA a true multi-objective optimizer aligned with the reference. Essential for fair comparison with RL in Phase 5 ("classical baseline" must be strong).

**Effort:** 2 hours | **Risk:** Low

---

## SECTION C: TESTING & VALIDATION INFRASTRUCTURE (Days 3–5)

The current test suite has **2 trivial tests** in `tests/test_core.py`. This is unacceptable for a research platform. An IEEE reviewer would question reproducibility.

---

### C1. Core Unit Test Suite

**File:** `tests/test_core.py` (expand from 2 → 25+ tests)

Priority test cases:
```
✅ test_energy_model_propulsion_curve    # Verify P(v) matches Eq. 3 of TVT paper
✅ test_communication_rician_fading      # Verify Shannon rate computation
✅ test_rendezvous_selection             # RP count < node count, all nodes covered
✅ test_pca_gls_route_shorter_than_naive # GLS beats naive nearest-neighbor
✅ test_ga_better_than_seed             # GA fitness ≥ PCA-GLS seed fitness
✅ test_hover_optimizer_improves_cost   # SCA reduces aggregate hover cost
✅ test_semantic_clustering_purity      # Silhouette > 0.3 for clean distributions
✅ test_buffer_aware_collection         # Buffer fills, drains, doesn't overflow
✅ test_aoi_decreases_on_visit          # AoI timer resets after collection
✅ test_obstacle_avoidance_no_collision # No collision in 100-step run
✅ test_batch_runner_ci_bounds          # 95% CI computed correctly
✅ test_node_battery_drain              # First-order radio model depletes correctly
✅ test_bs_uplink_trigger               # BS uplink fires before data-age limit
✅ test_gaussian_altitude_constraint    # UAV altitude > obstacle peak + clearance
✅ test_digital_twin_updates            # ISAC sensing populates obstacle map
```

**Impact:** Demonstrates research rigor. Every time you change a module, tests catch regressions. BTP evaluators will be impressed by CI-grade infrastructure.

**Effort:** 6 hours | **Risk:** None (pure additive)

---

### C2. Ablation Study Runner

**File:** `experiments/ablation_runner.py` (new)

An ablation study is **mandatory** for any IEEE paper. It answers: "How much does each component contribute?"

```python
class AblationRunner:
    """
    Systematically toggles each feature flag and runs 10-run batches
    to measure marginal contribution.
    """
    ABLATION_CONFIGS = [
        ("Full Pipeline", {}),                                         # baseline
        ("-RP Selection", {"ENABLE_RENDEZVOUS_SELECTION": False}),
        ("-GA Optimizer", {"ENABLE_GA_SEQUENCE": False}),
        ("-SCA Hover", {"ENABLE_SCA_HOVER": False}),
        ("-Semantic Clustering", {"ENABLE_SEMANTIC_CLUSTERING": False}),
        ("-Probabilistic Sensing", {"ENABLE_PROBABILISTIC_SENSING": False}),
        ("-BS Uplink", {"ENABLE_BS_UPLINK_MODEL": False}),
        ("-Moving Obstacles", {"ENABLE_MOVING_OBSTACLES": False}),
        ("-TDMA", {"ENABLE_TDMA_SCHEDULING": False}),
    ]
    
    def run_all(self):
        results = {}
        for name, overrides in self.ABLATION_CONFIGS:
            # Apply overrides to Config
            # Run BatchRunner(runs=10)
            # Store aggregated metrics
            results[name] = aggregated
        return results  # → Renders as bar chart comparison
```

**Impact:** Quantifies the value of every module. This is the single most important figure in any systems paper. Professor will ask "which module matters most?" — this answers it definitively.

**Effort:** 3 hours | **Risk:** None

---

### C3. Scalability Experiment

**File:** `experiments/scalability_runner.py` (new)

Test performance across node counts: N ∈ {20, 50, 100, 200, 500}

```python
NODE_COUNTS = [20, 50, 100, 200, 500]
for n in NODE_COUNTS:
    Config.NODE_COUNT = n
    results = BatchRunner(runs=5).execute()
    # Record: coverage%, energy/node, avg_aoi, computation_time, replans
```

**Impact:** Shows the platform scales. IEEE reviewers always ask "does it work beyond toy scenarios?"

**Effort:** 1 hour | **Risk:** None

---

## SECTION D: VISUALIZATION & PRESENTATION (Days 4–6)

The visualization infrastructure is already strong (11 plot types, IEEE styling, dual PNG+PDF export). Here's what's missing for the mid-term.

---

### D1. Ablation Study Visualization

**File:** `visualization/plot_renderer.py` (add method)

```python
@staticmethod
def render_ablation_chart(ablation_results: dict, save_dir: str):
    """
    Grouped bar chart: metrics (coverage, DR%, AoI, energy/node)
    across all ablation configurations.
    Each bar group = one metric, grouped by config variant.
    """
```

This produces the single most important figure for the BTP presentation.

---

### D2. Convergence Plots for GA and GLS

**File:** `visualization/plot_renderer.py` (add method)

```python
@staticmethod
def render_convergence_curve(fitness_history: list, save_dir: str, 
                              algorithm_name: str = "GA"):
    """
    Fitness vs. generation for GA, or cost vs. iteration for GLS.
    Shows that the optimizer is actually improving over iterations.
    """
```

**Why:** If your GA runs 50 generations but converges at generation 5, that's valuable insight (either reduce generations or improve diversity).

---

### D3. Clustering Quality Evolution Plot

**File:** `visualization/plot_renderer.py` (add method)

Track silhouette score at each reclustering event across the mission:

```python
@staticmethod
def render_clustering_quality(silhouette_history: list, save_dir: str):
    """Silhouette score vs. mission step — shows cluster quality over time."""
```

---

### D4. LaTeX Table Export for Experiment Results

**File:** `metrics/latex_exporter.py` (new)

```python
class LaTeXExporter:
    @staticmethod
    def results_to_latex_table(results: dict, caption: str) -> str:
        """Export batch results as a LaTeX tabular environment."""
        # Generates:
        # \begin{table}[h]\centering
        # \caption{...}
        # \begin{tabular}{l|ccccc}
        # Metric & Mean & Std & Min & Max & 95% CI \\
        # ...
```

**Impact:** When writing the IEEE paper, you can directly paste generated LaTeX tables.

**Effort:** 2 hours | **Risk:** None

---

## SECTION E: CODE QUALITY & ARCHITECTURE (Days 5–7)

---

### E1. Type Hints Throughout

**Current:** Partial type hints. Many functions lack annotations.

**Upgrade:** Add comprehensive type hints to all public methods. Focus on:
- `mission_controller.py` (703 lines — many untyped methods)
- `metric_engine.py` (354 lines)
- `simulation_runner.py`
- All clustering modules

**Impact:** IDE support, self-documenting code, professor can read signatures to understand architecture.

**Effort:** 3 hours | **Risk:** None

---

### E2. Docstrings to Research Standard

**Current:** Some modules have excellent docstrings (metric_engine, communication), others have minimal or none.

**Upgrade:** Every public method gets:
```python
def method(self, param: Type) -> ReturnType:
    """
    One-line description.
    
    Mathematical formulation:
        Eq. X: result = f(param) as defined in [Reference Paper].
    
    Args:
        param: Description with units.
    
    Returns:
        Description with units.
    """
```

**Impact:** When you write the IEEE paper, your code IS the supplementary material. Reviewers may check it.

**Effort:** 4 hours | **Risk:** None

---

### E3. Configuration Validation Layer

**File:** `config/config.py` (add validation method)

```python
@classmethod
def validate(cls):
    """Run all config consistency checks before simulation."""
    assert cls.BATTERY_CAPACITY > 0, "Battery must be positive"
    assert cls.NODE_COUNT >= 5, "Need at least 5 nodes"
    assert cls.CLUSTER_ALGO_MODE in ("kmeans", "dbscan", "gmm", "auto"), \
        f"Unknown clustering mode: {cls.CLUSTER_ALGO_MODE}"
    assert cls.RP_COVERAGE_RADIUS > 0, "RP radius must be positive"
    # ... 20+ checks
    print("[Config] All validation checks passed ✓")
```

**Impact:** Prevents silent misconfigurations. Run at startup.

**Effort:** 1 hour | **Risk:** None

---

## SECTION F: RESEARCH POSITIONING & NOVEL CONTRIBUTIONS (Days 7–10)

This section defines **what makes YOUR paper different** from the 50+ existing papers.

---

### F1. Your Unique Position in the Literature

**Existing Landscape (2024–2026):**

| Approach | Representative Papers | Limitation |
|----------|----------------------|------------|
| Pure DRL | D3QN (Wang et al.), TD3 (Chen et al.), Rainbow DQN | No classical routing baseline for fair comparison |
| Pure Heuristic | PCA-GLS (Donipati), GA (Zheng & Liu) | No learning component, static optimization |
| Attention/Transformer | AUTO (IEEE TIE 2025), CaDA (Nov 2024) | Requires massive training, no interpretability |
| LLM-Heuristic | ReEvo (NeurIPS 2024), DRAGON (AAMAS 2026) | Requires LLM API, not lightweight |
| Multi-Agent Federated | AirFed (Oct 2025) | Requires multiple UAVs, complex infrastructure |

**Your Paper's Novelty (proposed):**
> "A Hybrid Classical-RL Framework for UAV-IoT Data Collection: Bridging Heuristic Optimization with Deep Reinforcement Learning"

**Key contributions:**
1. **Unified platform** implementing 4 IEEE reference paper algorithms under one framework (no existing work does this)
2. **Rigorous ablation study** quantifying the marginal value of each classical module (RP, GLS, GA, SCA, Semantic Clustering)
3. **Phase 5+: RL augmentation** where DRL (D3QN/TD3) REPLACES specific modules (e.g., replaces GA sequencing) while keeping others — unlike papers that use RL for everything
4. **Multi-objective optimization** combining AoI, energy, coverage, and network lifetime (most papers optimize 1–2 objectives)
5. **Temporal dynamic environment** with moving obstacles, risk zones, buffer dynamics, ISAC sensing — more realistic than any single reference paper

---

### F2. Specific Novel Algorithmic Contributions to Implement

#### F2a. Semantic-Aware GLS (New — Not in Any Reference Paper)

Extend GLS penalty augmentation to consider semantic features:
```python
# Instead of penalizing by pure distance:
augmented_cost(i,j) = α * dist(i,j) + β * aoi_urgency(j) + γ * buffer_util(j)
                     + λ * penalty[i][j]
```
This creates **Semantic GLS (S-GLS)** — a novel routing heuristic that uses semantic features in the guided local search. **No existing paper does this.** This is a legitimate contribution.

#### F2b. Adaptive Reclustering with Silhouette Feedback

Instead of reclustering on a fixed schedule or node-count threshold:
```python
# Track silhouette score. If it drops below threshold → recluster.
# If AoI of unvisited nodes exceeds limit → force recluster with higher AoI weight.
```
This creates **AoI-Driven Adaptive Clustering** — the clustering reacts to data freshness.

#### F2c. Hybrid Motion Primitive Scoring with Energy Prediction

Current motion primitive scoring (line ~450 in mission_controller.py) uses 4 weighted components. Add a 5th:
```python
ENERGY_PREDICTION_SCORE = predicted_energy_to_complete_remaining / battery_remaining
```
This makes the UAV energy-aware in its real-time decisions, not just in planning.

---

### F3. What Phase 5 RL Will Look Like (For Your Presentation Slide)

Based on the research, here's the strongest RL architecture for your specific problem:

| Component | Classical (Now) | RL Replacement (Phase 5) | Hybrid Approach |
|-----------|----------------|--------------------------|-----------------|
| Visit Sequencing | GA + PCA-GLS | D3QN (action = next node) | RL selects from GA's top-5 candidates |
| Hover Position | SCA Gradient | TD3 (continuous xyz) | TD3 fine-tunes SCA output |
| Reclustering | Threshold-based | Meta-RL (learn when to recluster) | RL triggers, classical executes |
| Motion Primitive | Weighted scoring | DQN (select primitive) | DQN trained on scoring function as reward |
| Energy Management | Rule-based return | Actor-Critic (continuous power control) | Hybrid: rules + RL for edge cases |

**State Space** (already designed in `agent_centric_transform.py`):
- 7-dim agent-centric: [dist_to_target, heading_error, battery_%, nearest_obstacle_dist, risk_level, cluster_density, data_urgency]

**Reward Function:**
```
R(t) = w₁·ΔData_collected + w₂·(-ΔEnergy) + w₃·(-ΔAoI_avg) 
     + w₄·(-collision_penalty) + w₅·coverage_bonus
```

**Training Curriculum:**
1. Train on static environment (no obstacles) → learn basic routing
2. Add static obstacles → learn avoidance
3. Add moving obstacles → learn predictive avoidance
4. Add risk zones + buffer dynamics → full problem

---

## SECTION G: DAY-BY-DAY EXECUTION PLAN

### Day 1 (March 2) — Critical Fixes
- [ ] **A1**: Wire HoverOptimizer (15 min)
- [ ] **A3**: Configurable AoI weight (30 min)
- [ ] **E3**: Config validation (1 hour)
- [ ] Run full 10-run batch to establish new baseline
- [ ] Git commit: "fix: wire SCA hover optimizer into mission pipeline"

### Day 2 (March 3) — GLS + Clustering
- [ ] **A2**: Implement full GLS penalty augmentation (2 hours)
- [ ] **B1** (partial): Add GMM + auto-k to semantic_clusterer (2 hours)
- [ ] Run comparative batch: GLS vs no-GLS
- [ ] Git commit: "feat: full GLS penalty mechanism per Donipati et al."

### Day 3 (March 4) — Clustering Completion + Testing Start
- [ ] **B1** (complete): Quality-based reclustering, algorithm auto-selection (2 hours)
- [ ] **B2**: Feature importance weighting (1.5 hours)
- [ ] **C1** (start): First 10 unit tests (3 hours)
- [ ] Git commit: "feat: adaptive semantic clustering with silhouette feedback"

### Day 4 (March 5) — GA Upgrade + More Testing
- [ ] **B3**: Multi-objective GA fitness (2 hours)
- [ ] **C1** (complete): Remaining 15 unit tests (3 hours)
- [ ] Git commit: "feat: multi-objective GA with energy+AoI fitness"

### Day 5 (March 6) — Ablation Infrastructure
- [ ] **C2**: Ablation study runner (3 hours)
- [ ] **D1**: Ablation visualization (1.5 hours)
- [ ] Run full ablation study (10 runs × 9 configs = 90 runs)
- [ ] Git commit: "experiment: complete ablation study of all modules"

### Day 6 (March 7) — Visualization + LaTeX
- [ ] **D2**: Convergence plots (1 hour)
- [ ] **D3**: Clustering quality plot (1 hour)
- [ ] **D4**: LaTeX table exporter (2 hours)
- [ ] Generate all publication figures for presentation
- [ ] Git commit: "viz: IEEE-grade convergence/clustering/ablation plots"

### Day 7 (March 8) — Scalability + Code Quality
- [ ] **C3**: Scalability experiment (1 hour + running time)
- [ ] **E1**: Type hints (3 hours)
- [ ] **E2**: Research-standard docstrings (4 hours — split across day)
- [ ] Git commit: "refactor: comprehensive type hints + docstrings"

### Day 8 (March 9) — Novel Contributions
- [ ] **F2a**: Semantic GLS implementation (3 hours)
- [ ] **F2b**: Adaptive reclustering with AoI feedback (2 hours)
- [ ] Run final comparative experiments: Semantic-GLS vs standard GLS
- [ ] Git commit: "feat: Semantic GLS — novel routing contribution"

### Day 9 (March 10) — Polish + Freeze
- [ ] Run final 10-run batch on all configurations
- [ ] Generate all figures and LaTeX tables
- [ ] Update README with experiment reproduction instructions
- [ ] Verify all tests pass
- [ ] Tag release: `v4.0-midterm`
- [ ] Git commit: "release: v4.0 — Phase 1-4 complete with research-grade validation"

---

## SECTION H: METRICS TO PRESENT AT MID-TERM

These are the exact metrics your BTP panel will expect:

| Metric | Source Paper | Your Current Implementation | Status |
|--------|-------------|---------------------------|--------|
| Data Collection Rate (DR%) | Wang et al. 2022 | ✅ `metric_engine.compute_data_collection_rate()` | Working |
| Coverage Ratio (%) | All papers | ✅ `metric_engine.coverage_ratio()` | Working |
| Mission Completion Time | Zheng & Liu 2025 | ✅ `metric_engine.compute_mission_completion_time()` | Working |
| Average Peak AoI (s) | Zheng & Liu 2025 | ✅ `metric_engine.compute_average_aoi()` | Working |
| Network Lifetime | Donipati et al. 2025 | ✅ `metric_engine.compute_network_lifetime()` | Working |
| Collision Rate | Wang et al. 2022 | ✅ `metric_engine.compute_collision_rate()` | Working |
| Energy per Node (J) | Derived | ✅ `metric_engine.energy_efficiency()` | Working |
| Path Stability Index | Novel | ✅ `metric_engine.compute_stability_metrics()` | Working |
| Priority Satisfaction (%) | Novel (Phase 4) | ✅ `metric_engine.compute_semantic_metrics()` | Working |
| Achievable Rate (Mbps) | Chen et al. 2025 | ✅ `metric_engine.compute_average_achievable_rate()` | Working |

**What's MISSING for the presentation:**
- Ablation study results (Section C2)
- Scalability study results (Section C3)  
- Convergence proof that GA/GLS actually improve over iterations (Section D2)
- Comparison table: your method vs. each reference paper's reported numbers

---

## SECTION I: COMPETITIVE PAPER LANDSCAPE (2024–2026)

### Landmark Papers to Cite in Your IEEE Submission

**Core RL for Routing:**
1. RL4CO (KDD 2025 Oral) — Unified RL benchmark for combinatorial optimization, 27 environments
2. MVMoE (ICML 2024) — Multi-task VRP with Mixture-of-Experts, zero-shot cross-problem generalization

**Neural Architecture for CO:**
3. CaDA (Nov 2024) — Cross-problem VRP with constraint-aware dual-attention, SOTA on 16 VRP variants
4. DRHG (AAAI 2025) — Destroy-and-repair with hypergraphs, TSP up to 10K nodes
5. GREAT (Aug 2024) — Edge-based GNN for non-Euclidean TSP/CVRP/OP

**UAV-Specific DRL:**
6. AUTO (IEEE TIE 2025) — Graph transformer + actor-critic for UAV trajectory (with hardware experiments!)
7. AirFed (Oct 2025) — Federated graph-enhanced MARL for multi-UAV MEC
8. Rainbow DQN (Sep 2024) — UAV-enabled data collection via Rainbow Learning
9. Meta-DRL (Jan 2025) — Age and power minimization via meta-DRL

**LLM-for-Heuristics (Cutting Edge):**
10. ReEvo (NeurIPS 2024) — LLM as hyper-heuristic with reflective evolution for CO
11. DRAGON (AAMAS 2026) — LLM-driven decomposition for large-scale CO

**Your positioning:** You bridge papers 1–5 (which are pure optimization/RL) with papers 6–9 (which are UAV-specific but single-method). Nobody has built a unified platform that:
- Implements 4 classical algorithms from IEEE references
- Validates each via ablation
- Then layers RL on top selectively

---

## SECTION J: FILE-LEVEL CHANGE SUMMARY

| File | Changes | Priority |
|------|---------|----------|
| `core/mission_controller.py` | Wire HoverOptimizer (+8 lines), energy prediction motion primitive (+10 lines) | Day 1 |
| `path/pca_gls_router.py` | Full GLS implementation (+60 lines) | Day 2 |
| `core/clustering/semantic_clusterer.py` | GMM/auto-k/weighted features (+80 lines) | Days 2–3 |
| `core/clustering/cluster_manager.py` | Silhouette-based reclustering (+20 lines) | Day 3 |
| `core/clustering/feature_scaler.py` | Feature importance weights (+15 lines) | Day 3 |
| `path/ga_sequence_optimizer.py` | Multi-objective fitness (+30 lines) | Day 4 |
| `config/config.py` | New params + validation method (+40 lines) | Days 1–4 |
| `tests/test_core.py` | 25+ test cases (+300 lines) | Days 3–5 |
| `experiments/ablation_runner.py` | New file (+100 lines) | Day 5 |
| `experiments/scalability_runner.py` | New file (+60 lines) | Day 7 |
| `metrics/latex_exporter.py` | New file (+80 lines) | Day 6 |
| `visualization/plot_renderer.py` | 3 new plot methods (+120 lines) | Day 6 |

**Total estimated new/changed lines: ~930**
**Total estimated effort: ~45 hours across 9 days**

---

## SECTION K: WHAT TO SAY AT THE PRESENTATION

**Slide Structure (suggested):**

1. **Problem Statement**: UAV-IoT data collection in dynamic hostile environments
2. **Literature Survey**: 4 IEEE reference papers + 11 cutting-edge 2024–2026 papers
3. **System Architecture**: Modular pipeline diagram (RP → Clustering → PCA-GLS → GA → SCA Hover → Mission Execution)
4. **Phase 1–4 Results**: Main metrics table + radar chart
5. **Ablation Study**: "Which modules matter most?" bar chart
6. **Scalability**: Performance vs. node count curves
7. **Novel Contributions**: Semantic GLS, AoI-Driven Adaptive Clustering
8. **Phase 5–10 Roadmap**: D3QN/TD3 RL integration plan with curriculum training
9. **IEEE Paper Plan**: Target journal, timeline, novelty positioning

**Key sentence for the panel:**
> "We have built a research-grade simulation platform that integrates algorithms from 4 IEEE papers, validated each module through rigorous ablation, and identified specific novel contributions — Semantic GLS routing and AoI-driven adaptive clustering — that form the basis of our IEEE Transaction paper."

---

*Document generated: March 2025*
*Platform: Autonomous UAV Simulation Platform v4.0*
*Author: BTP Research Team*
