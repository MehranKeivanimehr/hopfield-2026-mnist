from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    return dict(config.__dict__)


def apply_overrides(config: TrainConfig, overrides: list[str]) -> TrainConfig:
    payload = config_to_dict(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Config override must use key=value format, got: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if key not in payload:
            raise KeyError(f"Unknown config key: {key}")
        payload[key] = _parse_override_value(raw_value, payload[key])
    return TrainConfig(**payload)


def _parse_override_value(raw_value: str, current_value: Any) -> Any:
    value = raw_value.strip()
    if isinstance(current_value, bool):
        lowered = value.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Expected a boolean override value, got: {raw_value}")
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(value)
    if isinstance(current_value, float):
        return float(value)
    return value
