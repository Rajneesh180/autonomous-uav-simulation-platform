# Experiments

Reusable experiment harnesses for systematic evaluation.

## Available Runners

| Runner | Purpose | Usage |
|--------|---------|-------|
| `ablation_runner.py` | Feature ablation studies — toggles one feature at a time and measures impact | `python -m experiments.ablation_runner` |
| `scalability_runner.py` | Node-count scaling tests — sweeps node count and tracks metrics | `python -m experiments.scalability_runner` |

## Adding a New Experiment

1. Create `experiments/<name>_runner.py`
2. Accept config overrides via argparse or function params
3. Output results to `results/` using the standard artifact pipeline
4. Document the experiment in this README
