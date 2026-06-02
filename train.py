from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from cmkc.config import load_config
from cmkc.trainer import CMKCTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CMKC on a continual VQA task sequence.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed = _maybe_init_distributed()
    config = load_config(args.config)
    try:
        trainer = CMKCTrainer(config)
        trainer.run()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def _maybe_init_distributed() -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1 or dist.is_initialized():
        return False
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    return True


if __name__ == "__main__":
    main()
