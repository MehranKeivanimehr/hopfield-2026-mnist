# Hopfield MNIST Results

Generated locally on June 8, 2026 from the real MNIST IDX files in:

`C:\Users\m.keivanimehr\OneDrive - University of Florida\Desktop\CV Project\Level 1`

## Protocol

- Stored patterns: 10 total, one binarized MNIST training exemplar per digit.
- Recall: asynchronous Hopfield updates, up to 30 iterations.
- Trials: 12 random corruptions per stored pattern, corruption type, and severity.
- Metrics:
  - `Input acc`: pixel accuracy of the corrupted input against the clean stored pattern.
  - `Recalled acc`: pixel accuracy after Hopfield recall.
  - `Retrieval acc`: whether the recalled state is nearest to the correct stored pattern.

## Main Finding

The classic full-pixel Hopfield network collapses to a background-dominated attractor on sparse MNIST images. Variance-filtering the pixels before storage removes constant background pixels and makes associative recall meaningfully work.

| Mode | Stored pixels | Mean retrieval accuracy | Mean recalled pixel accuracy |
| --- | ---: | ---: | ---: |
| Full 28x28 pixels | 784 | 0.1000 | 0.8857 |
| Variance-filtered pixels | 232 | 0.6828 | 0.8844 |

The full-pixel result has deceptively high pixel accuracy because MNIST images are mostly background. Retrieval accuracy exposes the failure.

## Variance-Filtered Summary

| Noise | Severity | Input acc | Recalled acc | Improvement | Retrieval acc | Exact match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bitflip | 0.05 | 0.9488 | 0.9300 | -0.0187 | 0.8500 | 0.5250 |
| bitflip | 0.10 | 0.8978 | 0.9333 | +0.0356 | 0.8667 | 0.5167 |
| bitflip | 0.20 | 0.8004 | 0.9028 | +0.1023 | 0.7833 | 0.4167 |
| bitflip | 0.30 | 0.6957 | 0.8661 | +0.1704 | 0.7167 | 0.3167 |
| bitflip | 0.40 | 0.6047 | 0.7744 | +0.1697 | 0.4667 | 0.1833 |
| occlusion | 0.05 | 0.9739 | 0.9409 | -0.0330 | 0.8917 | 0.5500 |
| occlusion | 0.10 | 0.9239 | 0.8950 | -0.0289 | 0.7333 | 0.4000 |
| occlusion | 0.20 | 0.8205 | 0.8429 | +0.0223 | 0.5583 | 0.2583 |
| occlusion | 0.30 | 0.7678 | 0.7808 | +0.0130 | 0.4167 | 0.1583 |
| occlusion | 0.40 | 0.6894 | 0.6778 | -0.0116 | 0.1667 | 0.0417 |
| saltpepper | 0.05 | 0.9745 | 0.9419 | -0.0326 | 0.9167 | 0.5500 |
| saltpepper | 0.10 | 0.9503 | 0.9355 | -0.0147 | 0.8500 | 0.5500 |
| saltpepper | 0.20 | 0.9018 | 0.9190 | +0.0172 | 0.8333 | 0.4833 |
| saltpepper | 0.30 | 0.8462 | 0.9204 | +0.0742 | 0.8500 | 0.4833 |
| saltpepper | 0.40 | 0.7982 | 0.8990 | +0.1008 | 0.8000 | 0.4000 |

## Reproduce

```powershell
py -3.11 scripts\run_hopfield_mnist.py --pixel-mode full --trials 12 --patterns-per-digit 1 --output-dir artifacts\hopfield_mnist_full
py -3.11 scripts\run_hopfield_mnist.py --pixel-mode variance --variance-threshold 0.8 --trials 12 --patterns-per-digit 1 --output-dir artifacts\hopfield_mnist_variance
```

Generated artifacts:

- `artifacts/hopfield_mnist_full/summary.csv`
- `artifacts/hopfield_mnist_variance/summary.csv`
- `artifacts/hopfield_mnist_variance/report.md`
- `artifacts/hopfield_mnist_variance/*.png`

## Honest Limitation

This is associative recall of stored MNIST examples, not general digit classification. The result is useful for making the original project reproducible and technically honest, but it is not a modern ML benchmark.
