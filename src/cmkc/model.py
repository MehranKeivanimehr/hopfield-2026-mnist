from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.models as tv_models

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover - exercised only in minimal local installs
    AutoModel = None
    AutoTokenizer = None


@dataclass
class ModelOutputs:
    logits: torch.Tensor
    visual_features: torch.Tensor
    language_features: torch.Tensor
    fused_features: torch.Tensor


class HashTextEncoder(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int = 4096) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

    def forward(self, questions: list[str], max_length: int, device: torch.device) -> torch.Tensor:
        token_rows = [self._encode(question, max_length) for question in questions]
        token_ids = torch.tensor(token_rows, dtype=torch.long, device=device)
        mask = (token_ids != 0).unsqueeze(-1)
        embedded = self.embedding(token_ids)
        summed = (embedded * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts

    def _encode(self, question: str, max_length: int) -> list[int]:
        tokens = question.lower().strip().split()[:max_length]
        ids = [self._hash_token(token) for token in tokens]
        return ids + [0] * (max_length - len(ids))

    def _hash_token(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "big") % (self.vocab_size - 1) + 1


def build_tiny_vision_encoder() -> tuple[nn.Module, int]:
    width = 64
    encoder = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, width, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )
    return encoder, width


class ContinualVQAModel(nn.Module):
    def __init__(
        self,
        text_backbone: str,
        vision_backbone: str,
        hidden_dim: int,
        num_answers: int,
        dropout: float,
        max_question_length: int,
        pretrained_vision: bool = True,
        freeze_backbones: bool = False,
    ) -> None:
        super().__init__()
        self.max_question_length = max_question_length
        self.uses_hash_text = text_backbone in {"hash", "simple", "tiny"}

        if self.uses_hash_text:
            self.tokenizer = None
            self.text_encoder = HashTextEncoder(hidden_size=hidden_dim)
            text_width = hidden_dim
        else:
            if AutoTokenizer is None or AutoModel is None:
                raise ImportError("Install transformers or use text_backbone='hash' for a local smoke run.")
            self.tokenizer = AutoTokenizer.from_pretrained(text_backbone)
            self.text_encoder = AutoModel.from_pretrained(text_backbone)
            text_width = self.text_encoder.config.hidden_size

        if vision_backbone == "tiny_cnn":
            self.vision_encoder, vision_width = build_tiny_vision_encoder()
        elif vision_backbone == "resnet18":
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained_vision else None
            vision_model = tv_models.resnet18(weights=weights)
            vision_width = vision_model.fc.in_features
            vision_model.fc = nn.Identity()
            self.vision_encoder = vision_model
        else:
            raise ValueError("Supported vision backbones are 'resnet18' and 'tiny_cnn'.")

        if freeze_backbones:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = False
            for parameter in self.vision_encoder.parameters():
                parameter.requires_grad = False

        self.visual_projector = nn.Linear(vision_width, hidden_dim)
        self.language_projector = nn.Linear(text_width, hidden_dim)
        self.align_visual_to_language = nn.Linear(hidden_dim, hidden_dim)
        self.align_language_to_visual = nn.Linear(hidden_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_answers)

    def encode_questions(self, questions: list[str], device: torch.device) -> torch.Tensor:
        if self.uses_hash_text:
            return self.text_encoder(questions, self.max_question_length, device)

        encoded = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_question_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = self.text_encoder(**encoded)
        return outputs.last_hidden_state[:, 0]

    def forward(self, images: torch.Tensor, questions: list[str]) -> ModelOutputs:
        device = images.device
        visual_base = self.vision_encoder(images)
        language_base = self.encode_questions(questions, device)
        visual_features = self.visual_projector(visual_base)
        language_features = self.language_projector(language_base)
        fused = self.fusion(torch.cat([visual_features, language_features], dim=-1))
        logits = self.classifier(fused)
        return ModelOutputs(
            logits=logits,
            visual_features=visual_features,
            language_features=language_features,
            fused_features=fused,
        )
