from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class TaskSpec:
    name: str
    train_path: str
    val_path: str


@dataclass
class SequenceSpec:
    tasks: list[TaskSpec]
    answer_vocab: list[str]


def load_sequence(path: str | Path) -> SequenceSpec:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    tasks = [TaskSpec(**task) for task in payload["tasks"]]
    return SequenceSpec(tasks=tasks, answer_vocab=payload["answer_vocab"])


class VQATaskDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        answer_to_id: dict[str, int],
        image_size: int,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.answer_to_id = answer_to_id
        self.samples = self._load_rows(self.jsonl_path)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _load_rows(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples[index]
        image_path = Path(row["image"])
        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)

        answer = row["answer"]
        if answer not in self.answer_to_id:
            raise KeyError(f"Answer '{answer}' missing from answer vocabulary.")

        anchor_key = row.get("concept_id") or row.get("question_type") or answer
        return {
            "image": image_tensor,
            "question": row["question"],
            "answer_id": self.answer_to_id[answer],
            "answer_text": answer,
            "question_type": row.get("question_type", "unknown"),
            "anchor_key": anchor_key,
        }


def collate_vqa_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "questions": [item["question"] for item in batch],
        "answer_ids": torch.tensor([item["answer_id"] for item in batch], dtype=torch.long),
        "answer_texts": [item["answer_text"] for item in batch],
        "question_types": [item["question_type"] for item in batch],
        "anchor_keys": [item["anchor_key"] for item in batch],
    }
