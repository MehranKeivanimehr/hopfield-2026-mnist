from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    seed: int
    device: str
    output_dir: str
    num_workers: int
    sequence_path: str
    batch_size: int
    epochs_per_task: int
    learning_rate: float
    weight_decay: float
    max_question_length: int
    image_size: int
    vision_backbone: str
    text_backbone: str
    hidden_dim: int
    dropout: float
    lambda_vis_anchor: float
    lambda_lang_anchor: float
    lambda_align: float
    lambda_replay: float
    memory_budget_per_task: int
    prototype_decay: float
    save_every_task: bool
    pretrained_vision: bool = True
    freeze_backbones: bool = False


def load_config(path: str | Path) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return TrainConfig(**payload)
