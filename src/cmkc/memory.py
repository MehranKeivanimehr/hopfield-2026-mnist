from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import torch


@dataclass
class MemoryItem:
    question: str
    answer_id: int
    answer_text: str
    question_type: str
    anchor_key: str
    image_tensor: torch.Tensor


class ExemplarMemory:
    def __init__(self, budget_per_task: int) -> None:
        self.budget_per_task = budget_per_task
        self.items: list[MemoryItem] = []

    def add_task_examples(self, batch_items: list[MemoryItem]) -> None:
        if len(batch_items) <= self.budget_per_task:
            self.items.extend(batch_items)
            return
        self.items.extend(random.sample(batch_items, self.budget_per_task))

    def sample(self, batch_size: int) -> list[MemoryItem]:
        if not self.items:
            return []
        if len(self.items) <= batch_size:
            return list(self.items)
        return random.sample(self.items, batch_size)


class PrototypeMemory:
    def __init__(self, decay: float) -> None:
        self.decay = decay
        self.visual: dict[str, torch.Tensor] = {}
        self.language: dict[str, torch.Tensor] = {}
        self.counts = defaultdict(int)

    def update(self, anchor_keys: list[str], visual_features: torch.Tensor, language_features: torch.Tensor) -> None:
        for idx, key in enumerate(anchor_keys):
            v = visual_features[idx].detach().cpu()
            l = language_features[idx].detach().cpu()
            if key not in self.visual:
                self.visual[key] = v
                self.language[key] = l
            else:
                self.visual[key] = self.decay * self.visual[key] + (1.0 - self.decay) * v
                self.language[key] = self.decay * self.language[key] + (1.0 - self.decay) * l
            self.counts[key] += 1

    def get_visual(self, key: str, device: torch.device) -> torch.Tensor | None:
        tensor = self.visual.get(key)
        return None if tensor is None else tensor.to(device)

    def get_language(self, key: str, device: torch.device) -> torch.Tensor | None:
        tensor = self.language.get(key)
        return None if tensor is None else tensor.to(device)
