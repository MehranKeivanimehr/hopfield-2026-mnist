# Cluster Run Notes

The current codebase is designed for multi-GPU data parallel training with a simple launch command:

```bash
torchrun --nproc_per_node=8 train.py --config configs/baseline.json
```

## Recommended Environment

- Python 3.11
- CUDA-compatible PyTorch build for the cluster driver
- 8 x A6000 GPUs
- local SSD scratch for dataset caching and checkpoints

## What Still Needs To Be Filled In

- real task sequence manifest
- dataset paths accessible from the cluster
- answer vocabulary for the target benchmark
- target model scaling decision

For a first full run, use a classification-style continual VQA benchmark. This scaffold is intentionally conservative so it can be debugged on one GPU and scaled to eight GPUs once the data manifest is stable.
