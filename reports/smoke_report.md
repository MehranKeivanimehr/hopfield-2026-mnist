# CMKC Smoke Experiment Report

- Config: `configs/smoke.json`
- Purpose: verify the continual-learning pipeline on a tiny CPU-only task sequence
- Average accuracy: 0.5000
- Average forgetting: 0.0000

## Task Accuracy Matrix

| After task | Evaluated task accuracies |
| --- | --- |
| 1 | 0.5000 |
| 2 | 0.5000, 0.5000 |

## Interpretation

This is a smoke result, not a benchmark claim. It confirms that the code can train sequential tasks, update prototype memory, run replay, use a teacher snapshot for alignment distillation, evaluate previous tasks, and export metrics.

Paper-grade results should use VQACL or NExT-QA splits, multiple random seeds, and ablations against replay-only and no-consolidation baselines.
