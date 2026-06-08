# GitHub Release Checklist

Use this checklist before making the repository public.

## Ready Now

- Python package under `src/cmkc/`
- CPU smoke config
- task-loss-only ablation config
- one-command local demo
- generated local demo report
- pytest smoke tests
- GitHub Actions test workflow
- dataset format documentation
- cluster launch notes

## Do Not Claim Yet

- Do not claim state-of-the-art results.
- Do not claim CMKC improves continual VQA on real benchmarks until real splits are run.
- Do not treat the bundled sample sequence as evidence of method quality.

## Public README Claims That Are Safe

- CMKC is a reproducible PyTorch scaffold for continual VQA experiments.
- The repo includes cross-modal prototype memory, alignment distillation, exemplar replay, and ablation configs.
- The bundled demo validates the end-to-end training/evaluation/reporting loop.

## Before a Results Release

1. Add real continual VQA task manifests.
2. Run at least three seeds.
3. Include a task-loss-only baseline.
4. Include replay/prototype/alignment ablations.
5. Report mean and standard deviation.
6. Save configs, summaries, and reports for every run.
7. Tag the release with the exact commit used for experiments.

## Suggested First Public Tag

`v0.1.0-reproducible-scaffold`

This tag should mean the repo is cloneable, installable, testable, and runnable locally. It should not imply benchmark results.
