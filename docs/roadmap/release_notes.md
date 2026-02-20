# Release Notes

## v0.3.1 — Phase-3 Closure Patch

### Summary

This patch finalizes Phase-3 (Dynamic Environment & Real-Time Adaptation).

It stabilizes architecture, formalizes mathematical modeling,
and completes documentation required for research-grade release.

---

### Added

- Formal rectangle obstacle mathematical formulation
- Structured stability metric documentation
- Dynamic environment model formalization
- Experiment protocol definition
- Simulation runner abstraction layer
- Centralized stability metrics inside MetricEngine

---

### Refactored

- Removed metric computation from main()
- Modularized simulation execution
- Cleaned import paths
- Repository structure normalization

---

### Known Limitations

- Replanning is full recomputation (no incremental graph update)
- Adaptation latency pairing simplistic
- No probabilistic obstacle uncertainty
- Axis-aligned rectangle assumption

---

Phase-3 is now considered structurally and academically closed.
