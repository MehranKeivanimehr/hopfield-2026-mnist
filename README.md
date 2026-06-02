# CMKC: Cross-Modal Knowledge Consolidation

CMKC is a PyTorch research codebase for continual vision-language learning. It focuses on visual question answering style task sequences where a model must learn new visual-language tasks without forgetting earlier ones.

The central idea is that each modality should help stabilize the other:

- language-anchored visual stabilization
- visual-grounded language stabilization
- cross-modal alignment distillation
- exemplar replay with semantic prototype memory

This repository includes a fast CPU smoke run, a heavier Transformer/ResNet baseline path, cluster scripts, and a research proposal with references.

## Why This Repo Exists

Continual learning in vision-language systems is not just image forgetting or text forgetting. The model can also forget the alignment between visual evidence and language semantics. CMKC is a compact experimental framework for testing that failure mode and prototyping cross-modal consolidation losses.

## Repository Layout

- `src/cmkc/`: CMKC package
- `train.py`: training entry point
- `evaluate.py`: prints a saved `summary.json`
- `configs/smoke.json`: fast CPU-only smoke experiment
- `configs/baseline.json`: ResNet-18 plus Transformer baseline config
- `data/sample_sequence.json`: tiny two-task sample sequence
- `docs/dataset_format.md`: JSON/JSONL dataset format
- `proposal.md`: research proposal and method motivation
- `references.bib`: BibTeX references
- `scripts/report_results.py`: creates a Markdown report from run artifacts
- `scripts/cluster/`: UF Focus cluster setup and SLURM helpers
- `reports/cluster_report_2026-03-19.md`: benchmark selection and execution plan
- `reports/smoke_report.md`: checked-in example smoke result

## Quickstart

Use Python 3.11. On this machine, plain `python` may point to an older Python, so prefer `py -3.11` on Windows.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3.11 -m pip install --upgrade pip
py -3.11 -m pip install -e ".[dev]"
```

Run the fast smoke experiment:

```powershell
py -3.11 train.py --config configs/smoke.json
py -3.11 scripts/report_results.py --run-dir artifacts/smoke --config configs/smoke.json
py -3.11 evaluate.py --summary artifacts/smoke/summary.json
```

The smoke config uses:

- `vision_backbone: tiny_cnn`
- `text_backbone: hash`
- CPU execution
- the tiny sample task sequence in `data/sample/`

It is meant to validate the full continual-learning loop quickly. It is not a publishable benchmark.

## Baseline Training

For the heavier baseline:

```powershell
py -3.11 train.py --config configs/baseline.json
```

`configs/baseline.json` uses ResNet-18 and `distilbert-base-uncased`, so the first run may download pretrained weights. For offline cluster runs, cache model weights before launching jobs.

Multi-GPU launch:

```bash
torchrun --nproc_per_node=8 train.py --config configs/baseline.json
```

## Dataset Format

Each task has a train JSONL and validation JSONL file. A row looks like:

```json
{
  "image": "data/sample/images/demo_1.jpg",
  "question": "What color is the object?",
  "answer": "red",
  "question_type": "color",
  "concept_id": "red"
}
```

See `docs/dataset_format.md` for the full format.

## Metrics

The trainer writes:

- `<task>_metrics.json` after each task
- `summary.json` after the full sequence
- optional model checkpoints when `save_every_task` is true

Reported metrics:

- task accuracy matrix
- final average accuracy
- final average forgetting

Generate a readable report:

```powershell
py -3.11 scripts/report_results.py --run-dir artifacts/smoke --config configs/smoke.json
```

## Testing

```powershell
py -3.11 -m pytest
```

The tests exercise config loading, sample data loading, tiny model forward passes, and a one-epoch continual training run.

## Toward Valuable Results

The smoke run proves the mechanics. Valuable results require real continual VQA benchmarks and multiple seeds.

Recommended path:

1. Reproduce QUAD/VQACL results on VQAv2.
2. Reproduce or adapt NExT-QA continual VQA splits.
3. Add CMKC losses to the official benchmark path.
4. Report ablations:
   - no replay
   - no prototype memory
   - no alignment distillation
   - full CMKC
5. Run at least three seeds and report mean plus standard deviation.

The prepared cluster plan is in `reports/cluster_report_2026-03-19.md`.

## Citation

If this repository is useful, cite it as:

```bibtex
@software{cmkc2026,
  title = {CMKC: Cross-Modal Knowledge Consolidation for Continual Vision-Language Learning},
  author = {Keivanimehr, Mohammad},
  year = {2026},
  url = {https://github.com/REPLACE_WITH_USERNAME/REPLACE_WITH_REPO}
}
```

## Status

This is an active research scaffold. It is ready for local smoke runs and method development, while paper-grade benchmark results still require the real task splits, dataset paths, and cluster execution.
