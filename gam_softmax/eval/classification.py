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


@torch.no_grad()
def expected_calibration_error(
    confidences: torch.Tensor,
    correct: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Standard equal-width ECE (Guo et al. 2017).

    Bins predictions by their top-class probability and returns the
    sample-weighted mean gap between bin confidence and bin accuracy. Lower is
    better; 0 means the model's stated confidence matches how often it is right.

    Reported because *calibration under label noise* is where a margin loss is
    supposed to pay off: cross-entropy keeps pushing p_t → 1 even on corrupted
    labels, which makes it overconfident, while a margin loss stops pushing once
    the gap is met.

    Args:
        confidences: (N,) top-class probabilities.
        correct: (N,) 1.0 where the prediction was right, else 0.0.
        n_bins: number of equal-width bins over [0, 1].
    """
    confidences = confidences.detach().flatten().float()
    correct = correct.detach().flatten().float()
    if confidences.numel() == 0:
        return float("nan")

    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=confidences.device)
    ece = torch.zeros((), device=confidences.device)
    n = confidences.numel()
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # left-open bins, except the first which also takes conf == 0
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences <= hi)
        count = in_bin.sum()
        if count == 0:
            continue
        acc = correct[in_bin].mean()
        conf = confidences[in_bin].mean()
        ece = ece + (count.float() / n) * (acc - conf).abs()
    return float(ece.item())
