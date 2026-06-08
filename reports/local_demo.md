# CMKC Local Demo Results

Generated: 2026-06-08 20:59:10 UTC

This report is produced by `scripts/run_local_demo.py` on the bundled toy continual-VQA sequence.
It is a reproducibility check, not a claim of benchmark performance.

| Run | Average accuracy | Average forgetting | Elapsed seconds |
| --- | ---: | ---: | ---: |
| No CMKC ablation | 0.2500 | 0.5000 | 0.432 |
| CMKC smoke | 0.2500 | 0.5000 | 0.528 |

## Included Runs

- `configs/ablation_no_cmkc.json`: task loss only, no anchors, alignment distillation, or replay.
- `configs/smoke.json`: full CMKC losses on the same tiny task sequence.

## Next Benchmark Step

Replace `data/sample_sequence.json` with a real continual VQA sequence, then repeat the same report workflow over at least three seeds.
