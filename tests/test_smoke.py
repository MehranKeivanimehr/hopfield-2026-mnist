from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from cmkc.config import load_config
from cmkc.data import VQATaskDataset, load_sequence
from cmkc.model import ContinualVQAModel
from cmkc.trainer import CMKCTrainer


def test_smoke_config_loads() -> None:
    config = load_config("configs/smoke.json")

    assert config.device == "cpu"
    assert config.vision_backbone == "tiny_cnn"
    assert config.text_backbone == "hash"
    assert config.pretrained_vision is False


def test_sample_dataset_loads_images() -> None:
    config = load_config("configs/smoke.json")
    sequence = load_sequence(config.sequence_path)
    answer_to_id = {answer: idx for idx, answer in enumerate(sequence.answer_vocab)}

    dataset = VQATaskDataset(sequence.tasks[0].train_path, answer_to_id, config.image_size)
    sample = dataset[0]

    assert sample["image"].shape == (3, config.image_size, config.image_size)
    assert sample["answer_id"] in answer_to_id.values()
    assert sample["anchor_key"]


def test_tiny_model_forward_pass() -> None:
    config = load_config("configs/smoke.json")
    sequence = load_sequence(config.sequence_path)
    answer_to_id = {answer: idx for idx, answer in enumerate(sequence.answer_vocab)}
    dataset = VQATaskDataset(sequence.tasks[0].train_path, answer_to_id, config.image_size)
    sample = dataset[0]
    model = ContinualVQAModel(
        text_backbone=config.text_backbone,
        vision_backbone=config.vision_backbone,
        hidden_dim=config.hidden_dim,
        num_answers=len(sequence.answer_vocab),
        dropout=config.dropout,
        max_question_length=config.max_question_length,
        pretrained_vision=config.pretrained_vision,
        freeze_backbones=config.freeze_backbones,
    )

    outputs = model(sample["image"].unsqueeze(0), [sample["question"]])

    assert outputs.logits.shape == (1, len(sequence.answer_vocab))
    assert outputs.visual_features.shape[-1] == config.hidden_dim
    assert outputs.language_features.shape[-1] == config.hidden_dim


def test_trainer_writes_summary(tmp_path: Path) -> None:
    config = replace(
        load_config("configs/smoke.json"),
        output_dir=str(tmp_path / "run"),
        epochs_per_task=1,
        save_every_task=False,
    )

    CMKCTrainer(config).run()

    summary_path = tmp_path / "run" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "task_accuracy_matrix" in summary
    assert len(summary["task_accuracy_matrix"]) == 2
