# Hopfield 2026 MNIST Lab Results

This is the upgraded, GitHub-facing version of the original MNIST Hopfield project. It compares old and newer associative-memory views under the same corrupted-memory retrieval protocol.

## What Was Tested

- Real MNIST IDX files from the original project folder.
- Stored patterns: 10, 30, 50, and 100 MNIST training examples.
- Corruptions: bit-flip, salt-pepper, and square occlusion.
- Trials: 10 random corruptions per stored pattern and setting.
- Pixel modes:
  - `full`: all 784 MNIST pixels.
  - `variance`: only pixels that vary across stored memories, reducing background-attractor bias.
- Methods:
  - `hebbian`: classical Hopfield memory.
  - `pseudoinverse`: projection/pseudoinverse associative memory.
  - `modern_attention`: modern Hopfield / softmax-attention style retrieval.
  - `nearest`: nearest-neighbor memory baseline.

## Main Result

| Method | Pixel mode | Label retrieval | Exact retrieval | Pixel accuracy | Improvement |
| --- | --- | ---: | ---: | ---: | ---: |
| nearest | full | 0.9869 | 0.9820 | 0.9979 | +0.1321 |
| nearest | variance | 0.9777 | 0.9694 | 0.9900 | +0.1558 |
| pseudoinverse | full | 0.9758 | 0.9655 | 0.9953 | +0.1295 |
| modern_attention | variance | 0.9677 | 0.9583 | 0.9874 | +0.1533 |
| pseudoinverse | variance | 0.9590 | 0.9441 | 0.9643 | +0.1302 |
| modern_attention | full | 0.9006 | 0.8656 | 0.9785 | +0.1127 |
| hebbian | variance | 0.2836 | 0.2042 | 0.7273 | -0.1068 |
| hebbian | full | 0.1000 | 0.0408 | 0.8797 | +0.0139 |

Bootstrap 95% confidence intervals over the 12 capacity/noise settings:

| Method | Pixel mode | Label retrieval 95% CI |
| --- | --- | ---: |
| nearest | full | [0.9753, 0.9969] |
| nearest | variance | [0.9564, 0.9950] |
| pseudoinverse | full | [0.9537, 0.9947] |
| modern_attention | variance | [0.9366, 0.9933] |
| pseudoinverse | variance | [0.9197, 0.9937] |
| modern_attention | full | [0.8057, 0.9793] |
| hebbian | variance | [0.1497, 0.4492] |
| hebbian | full | [0.1000, 0.1000] |

## Figures

![Method comparison](../docs/assets/hopfield_2026/method_comparison.png)

![Capacity sweep](../docs/assets/hopfield_2026/capacity_sweep.png)

![Accuracy heatmap](../docs/assets/hopfield_2026/accuracy_heatmap.png)

![Qualitative gallery](../docs/assets/hopfield_2026/qualitative_gallery.png)

## Honest Interpretation

The project now has a clear result:

1. Classical Hebbian Hopfield recall is a bad fit for sparse MNIST pixels. It collapses into background-dominated attractors.
2. Pseudoinverse memory largely fixes this while staying in the classical associative-memory family.
3. Modern Hopfield / attention-style retrieval is strong, especially after variance filtering.
4. Nearest-neighbor is a brutally strong baseline for this exact stored-memory task and must be reported. The project should not pretend modern Hopfield beats nearest-neighbor here.

That honest baseline is a strength for GitHub: the repo becomes a useful lab for seeing where classical Hopfield fails and how projection/attention-style memories behave.

## Reproduce

```powershell
py -3.11 scripts\run_hopfield_2026_lab.py --output-dir artifacts\hopfield_2026_trials10 --trials-per-setting 10
```

## References

- Krotov and Hopfield, *Dense Associative Memory for Pattern Recognition*, NeurIPS 2016.
- Ramsauer et al., *Hopfield Networks is All You Need*, 2020.
