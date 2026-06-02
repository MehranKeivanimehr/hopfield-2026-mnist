from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from .config import TrainConfig
from .data import VQATaskDataset, collate_vqa_batch, load_sequence
from .memory import ExemplarMemory, MemoryItem, PrototypeMemory
from .metrics import ContinualMetrics
from .model import ContinualVQAModel, ModelOutputs


class CMKCTrainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self._set_seed(config.seed)
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available() and config.device.startswith("cuda"):
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)
        else:
            self.device = torch.device("cpu")
        self.output_dir = Path(config.output_dir)
        if self.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sequence = load_sequence(config.sequence_path)
        self.answer_to_id = {answer: idx for idx, answer in enumerate(self.sequence.answer_vocab)}
        self.prototype_memory = PrototypeMemory(decay=config.prototype_decay)
        self.exemplar_memory = ExemplarMemory(budget_per_task=config.memory_budget_per_task)
        self.metrics = ContinualMetrics()

        base_model = ContinualVQAModel(
            text_backbone=config.text_backbone,
            vision_backbone=config.vision_backbone,
            hidden_dim=config.hidden_dim,
            num_answers=len(self.sequence.answer_vocab),
            dropout=config.dropout,
            max_question_length=config.max_question_length,
            pretrained_vision=config.pretrained_vision,
            freeze_backbones=config.freeze_backbones,
        ).to(self.device)
        if self.is_distributed:
            self.model = DDP(base_model, device_ids=[local_rank] if self.device.type == "cuda" else None)
        else:
            self.model = base_model
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.teacher_model: ContinualVQAModel | None = None

    def run(self) -> None:
        for task_index, task in enumerate(self.sequence.tasks):
            train_dataset = VQATaskDataset(task.train_path, self.answer_to_id, self.config.image_size)
            train_sampler = (
                DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
                if self.is_distributed
                else None
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=train_sampler is None,
                sampler=train_sampler,
                num_workers=self.config.num_workers,
                collate_fn=collate_vqa_batch,
            )
            self._train_task(task.name, train_loader)
            self._update_exemplar_memory(train_dataset)
            self.teacher_model = copy.deepcopy(self._unwrap_model()).to(self.device).eval()

            row: list[float] = []
            for eval_task in self.sequence.tasks[: task_index + 1]:
                eval_dataset = VQATaskDataset(eval_task.val_path, self.answer_to_id, self.config.image_size)
                eval_sampler = (
                    DistributedSampler(eval_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
                    if self.is_distributed
                    else None
                )
                eval_loader = DataLoader(
                    eval_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    sampler=eval_sampler,
                    num_workers=self.config.num_workers,
                    collate_fn=collate_vqa_batch,
                )
                row.append(self.evaluate(eval_loader))
            self.metrics.add_row(row)

            if self.rank == 0:
                summary = {
                    "task": task.name,
                    "seen_tasks": task_index + 1,
                    "row_accuracy": row,
                    "average_accuracy": self.metrics.average_accuracy(),
                    "average_forgetting": self.metrics.average_forgetting(),
                }
                self._write_json(self.output_dir / f"{task.name}_metrics.json", summary)
                if self.config.save_every_task:
                    torch.save(self._unwrap_model().state_dict(), self.output_dir / f"{task.name}.pt")

        if self.rank == 0:
            self._write_json(
                self.output_dir / "summary.json",
                {
                    "task_accuracy_matrix": self.metrics.task_accuracy_matrix,
                    "average_accuracy": self.metrics.average_accuracy(),
                    "average_forgetting": self.metrics.average_forgetting(),
                },
            )

    def _train_task(self, task_name: str, train_loader: DataLoader) -> None:
        self.model.train()
        for epoch in range(self.config.epochs_per_task):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            progress = tqdm(
                train_loader,
                desc=f"{task_name} epoch {epoch + 1}",
                leave=False,
                disable=self.rank != 0,
            )
            for batch in progress:
                loss, stats = self._compute_batch_loss(batch)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                progress.set_postfix({key: f"{value:.4f}" for key, value in stats.items()})

    def _compute_batch_loss(self, batch: dict[str, object]) -> tuple[torch.Tensor, dict[str, float]]:
        images = batch["images"].to(self.device)
        questions = batch["questions"]
        answer_ids = batch["answer_ids"].to(self.device)
        anchor_keys = batch["anchor_keys"]

        outputs = self.model(images, questions)
        task_loss = F.cross_entropy(outputs.logits, answer_ids)
        vis_anchor_loss = self._visual_anchor_loss(anchor_keys, outputs)
        lang_anchor_loss = self._language_anchor_loss(anchor_keys, outputs)
        align_loss = self._alignment_distillation_loss(images, questions, outputs)
        replay_loss = self._replay_loss()

        total = (
            task_loss
            + self.config.lambda_vis_anchor * vis_anchor_loss
            + self.config.lambda_lang_anchor * lang_anchor_loss
            + self.config.lambda_align * align_loss
            + self.config.lambda_replay * replay_loss
        )

        self.prototype_memory.update(anchor_keys, outputs.visual_features, outputs.language_features)
        stats = {
            "total": total.item(),
            "task": task_loss.item(),
            "vis": vis_anchor_loss.item(),
            "lang": lang_anchor_loss.item(),
            "align": align_loss.item(),
            "replay": replay_loss.item(),
        }
        return total, stats

    def _visual_anchor_loss(self, anchor_keys: list[str], outputs: ModelOutputs) -> torch.Tensor:
        model = self._unwrap_model()
        losses: list[torch.Tensor] = []
        for idx, key in enumerate(anchor_keys):
            proto = self.prototype_memory.get_language(key, self.device)
            if proto is None:
                continue
            target = model.align_language_to_visual(proto.unsqueeze(0)).squeeze(0)
            losses.append(F.mse_loss(outputs.visual_features[idx], target))
        return torch.stack(losses).mean() if losses else outputs.visual_features.new_tensor(0.0)

    def _language_anchor_loss(self, anchor_keys: list[str], outputs: ModelOutputs) -> torch.Tensor:
        model = self._unwrap_model()
        losses: list[torch.Tensor] = []
        for idx, key in enumerate(anchor_keys):
            proto = self.prototype_memory.get_visual(key, self.device)
            if proto is None:
                continue
            target = model.align_visual_to_language(proto.unsqueeze(0)).squeeze(0)
            losses.append(F.mse_loss(outputs.language_features[idx], target))
        return torch.stack(losses).mean() if losses else outputs.language_features.new_tensor(0.0)

    def _alignment_distillation_loss(
        self,
        images: torch.Tensor,
        questions: list[str],
        outputs: ModelOutputs,
    ) -> torch.Tensor:
        if self.teacher_model is None:
            return outputs.fused_features.new_tensor(0.0)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(images, questions)
        student_similarity = F.cosine_similarity(outputs.visual_features, outputs.language_features, dim=-1)
        teacher_similarity = F.cosine_similarity(
            teacher_outputs.visual_features,
            teacher_outputs.language_features,
            dim=-1,
        )
        fused_loss = F.mse_loss(outputs.fused_features, teacher_outputs.fused_features)
        sim_loss = F.mse_loss(student_similarity, teacher_similarity)
        return fused_loss + sim_loss

    def _replay_loss(self) -> torch.Tensor:
        replay_items = self.exemplar_memory.sample(self.config.batch_size)
        if not replay_items:
            return torch.tensor(0.0, device=self.device)

        replay_images = torch.stack([item.image_tensor for item in replay_items], dim=0).to(self.device)
        replay_questions = [item.question for item in replay_items]
        replay_answers = torch.tensor([item.answer_id for item in replay_items], dtype=torch.long, device=self.device)
        outputs = self.model(replay_images, replay_questions)
        return F.cross_entropy(outputs.logits, replay_answers)

    def _update_exemplar_memory(self, dataset: VQATaskDataset) -> None:
        candidates: list[MemoryItem] = []
        for idx in range(len(dataset)):
            item = dataset[idx]
            candidates.append(
                MemoryItem(
                    question=item["question"],
                    answer_id=item["answer_id"],
                    answer_text=item["answer_text"],
                    question_type=item["question_type"],
                    anchor_key=item["anchor_key"],
                    image_tensor=item["image"],
                )
            )
        self.exemplar_memory.add_task_examples(candidates)

    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = torch.tensor(0.0, device=self.device)
        total = torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            for batch in loader:
                images = batch["images"].to(self.device)
                questions = batch["questions"]
                answer_ids = batch["answer_ids"].to(self.device)
                outputs = self.model(images, questions)
                predictions = outputs.logits.argmax(dim=-1)
                correct += (predictions == answer_ids).sum()
                total += torch.tensor(answer_ids.size(0), device=self.device)
        if self.is_distributed:
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        self.model.train()
        return (correct / torch.clamp(total, min=1.0)).item()

    def _unwrap_model(self) -> ContinualVQAModel:
        return self.model.module if isinstance(self.model, DDP) else self.model

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
