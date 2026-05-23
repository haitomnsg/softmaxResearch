from __future__ import annotations

import torch


@torch.no_grad()
def classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float()
    return {
        "accuracy": correct.mean().item(),
        "n": int(targets.numel()),
        "n_correct": int(correct.sum().item()),
    }
