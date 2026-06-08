# Hopfield 2026 Lab + CMKC Research Scaffold

This repository now contains two research tracks:

1. **Hopfield 2026 Lab**: a reproducible Python upgrade of the original MNIST Hopfield project, with classical Hebbian recall, pseudoinverse memory, modern Hopfield/attention retrieval, capacity sweeps, corruption sweeps, and GitHub-ready figures.
2. **CMKC**: a separate scaffold for continual vision-language learning experiments.

If you are here for visible results, start with the Hopfield 2026 Lab.

## Hopfield 2026 Lab Result

The upgraded Hopfield experiment runs on the real MNIST IDX files from the original project and compares old/new associative-memory variants.

| Method | Pixel mode | Label retrieval | Exact retrieval | Pixel accuracy |
| --- | --- | ---: | ---: | ---: |
| nearest | full | 0.9869 | 0.9820 | 0.9979 |
| pseudoinverse | full | 0.9758 | 0.9655 | 0.9953 |
| modern attention | variance | 0.9677 | 0.9583 | 0.9874 |
| Hebbian Hopfield | full | 0.1000 | 0.0408 | 0.8797 |

![Hopfield 2026 method comparison](docs/assets/hopfield_2026/method_comparison.png)

Full result page: `reports/hopfield_2026_results.md`

Run it:

```powershell
py -3.11 scripts\run_hopfield_2026_lab.py --output-dir artifacts\hopfield_2026_trials10 --trials-per-setting 10
```

The honest takeaway: classical Hebbian Hopfield recall fails badly on sparse MNIST; pseudoinverse and modern Hopfield/attention-style memory make retrieval work, while nearest-neighbor remains a very strong baseline.

## CMKC: Cross-Modal Knowledge Consolidation

CMKC is a PyTorch research codebase for continual vision-language learning. It focuses on visual question answering style task sequences where a model must learn new visual-language tasks without forgetting earlier ones.

The central idea is that each modality should help stabilize the other:

- language-anchored visual stabilization
- visual-grounded language stabilization
- cross-modal alignment distillation
- exemplar replay with semantic prototype memory

This repository includes a fast CPU smoke run, a task-loss-only ablation, a heavier Transformer/ResNet baseline path, cluster scripts, generated local reports, and a research proposal with references.

It also includes a Python reproduction path for the original Hopfield/MNIST project using the real MNIST IDX files from `Desktop/CV Project/Level 1`.

## Why This Repo Exists

Continual learning in vision-language systems is not just image forgetting or text forgetting. The model can also forget the alignment between visual evidence and language semantics. CMKC is a compact experimental framework for testing that failure mode and prototyping cross-modal consolidation losses.

## Repository Layout

- `src/cmkc/`: CMKC package
- `train.py`: training entry point
- `evaluate.py`: prints a saved `summary.json`
- `configs/smoke.json`: fast CPU-only smoke experiment
- `configs/ablation_no_cmkc.json`: task-loss-only ablation for the bundled sample sequence
- `configs/baseline.json`: ResNet-18 plus Transformer baseline config
- `data/sample_sequence.json`: tiny two-task sample sequence
- `docs/dataset_format.md`: JSON/JSONL dataset format
- `proposal.md`: research proposal and method motivation
- `references.bib`: BibTeX references
- `scripts/report_results.py`: creates a Markdown report from run artifacts
- `scripts/cluster/`: UF Focus cluster setup and SLURM helpers
- `reports/cluster_report_2026-03-19.md`: benchmark selection and execution plan
- `reports/smoke_report.md`: checked-in example smoke result
- `reports/local_demo.md`: generated local demo comparison
- `reports/hopfield_mnist_results.md`: real MNIST Hopfield recovery results
- `reports/hopfield_2026_results.md`: upgraded Hopfield 2026 benchmark results

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

## One-Command Local Demo

Run the bundled ablation and full CMKC smoke config, then regenerate Markdown reports:

```powershell
py -3.11 scripts/run_local_demo.py
```

This creates:

- `artifacts/ablation_no_cmkc/summary.json`
- `artifacts/ablation_no_cmkc/report.md`
- `artifacts/smoke/summary.json`
- `artifacts/smoke/report.md`
- `reports/local_demo.md`

Current local demo result on the bundled toy sequence:

| Run | Average accuracy | Average forgetting |
| --- | ---: | ---: |
| No CMKC ablation | 0.2500 | 0.5000 |
| CMKC smoke | 0.2500 | 0.5000 |

These numbers only prove reproducibility and wiring on a tiny sample sequence. They are not benchmark claims.

## Real Hopfield MNIST Run

The original CV project data is available locally as MNIST IDX files. Run the Python Hopfield reproduction:

```powershell
py -3.11 scripts\run_hopfield_mnist.py --pixel-mode full --trials 12 --patterns-per-digit 1 --output-dir artifacts\hopfield_mnist_full
py -3.11 scripts\run_hopfield_mnist.py --pixel-mode variance --variance-threshold 0.8 --trials 12 --patterns-per-digit 1 --output-dir artifacts\hopfield_mnist_variance
```

Summary:

| Mode | Stored pixels | Mean retrieval accuracy | Mean recalled pixel accuracy |
| --- | ---: | ---: | ---: |
| Full 28x28 pixels | 784 | 0.1000 | 0.8857 |
| Variance-filtered pixels | 232 | 0.6828 | 0.8844 |

See `reports/hopfield_mnist_results.md` for the full table. The full-pixel result reveals a background-attractor failure; retrieval accuracy is the honest metric here.

## Config Overrides

Any config value can be overridden from the command line:

```powershell
py -3.11 train.py --config configs/smoke.json --set output_dir=artifacts/debug --set epochs_per_task=2
```

Every run writes:

- `config.resolved.json`
- `run_info.json`
- per-task `*_metrics.json`
- final `summary.json`
- `report.md` when `scripts/report_results.py` is called

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
- training loss trace
- elapsed runtime and environment metadata

Generate a readable report:

```powershell
py -3.11 scripts/report_results.py --run-dir artifacts/smoke --config configs/smoke.json
```

## Testing

```powershell
py -3.11 -m pytest
```

The tests exercise config loading, sample data loading, tiny model forward passes, and a one-epoch continual training run.

Verified locally on June 8, 2026:

```text
5 passed
```

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
