from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TextClassifierOutput:
    logits: torch.Tensor
    features: torch.Tensor


class TextClassifier(nn.Module):
    """Thin wrapper: HF encoder → mean-pooled features → linear head.

    Exposes penultimate `features` so margin functions that need δ(x) can
    consume them without re-running the encoder.
    """

    def __init__(self, backbone_name: str, n_classes: int, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    @classmethod
    def tokenizer_for(cls, backbone_name: str):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(backbone_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> TextClassifierOutput:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # mean-pool over valid tokens
        last = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).type_as(last)
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        features = self.dropout(pooled)
        logits = self.head(features)
        return TextClassifierOutput(logits=logits, features=features)
