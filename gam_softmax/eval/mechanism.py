"""Mechanism diagnostics: where does each loss send its gradient? (Phase E, docs/10 §3)

Every loss in this repo is a function of the logits, so the per-sample gradient
∂L/∂o_i can be read off any loss with one autograd call on detached logits, with no
backward pass through the model and no RNG use. That makes the diagnostic generic
(CE, AS-Softmax, the M4 family, small-loss, GCE, SCE alike) and free to run inside
the memorization probe.

The organising quantity is the sample's *gap*

    g_i = max_{j≠t} p_j − p_t        ∈ [−1, 1]

(g < 0: the given label wins; g > 0: some other class beats it). For a margin loss
with per-sample margin δ_i, slot j is dropped iff p_t − p_j ≥ δ_i, so the sample's
loss and gradient are exactly zero iff g_i ≤ −δ_i. With δ_i > 0 that is AS-Softmax's
"already fitted by the margin"; with δ_i < 0 it is *rejection*: the label loses by
up to |δ_i| and still costs nothing. ``summarize`` splits zero-gradient samples by
the sign of g accordingly.
"""

from __future__ import annotations

import torch

N_BINS = 20  # gap histogram on [−1, 1], width 0.1


def label_gap(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """g_i = max_{j≠t} p_j − p_t per row, in probability space."""
    probs = torch.softmax(logits.detach().float(), dim=-1)
    p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    others = probs.scatter(1, targets.unsqueeze(1), -1.0)
    return others.max(dim=-1).values - p_t


def per_sample_grad_l1(loss_fn, logits: torch.Tensor, targets: torch.Tensor,
                       step_frac: float, features: torch.Tensor | None = None) -> torch.Tensor:
    """‖∂L/∂o_i‖₁ per sample, for the batch loss as the loss itself reduces it.

    The batch reduction (mean over B, or over the kept subset for small-loss) only
    rescales every row by the same constant, and every number ``summarize`` reports
    is a ratio, so the rows are used as returned.
    """
    with torch.enable_grad():
        z = logits.detach().float().requires_grad_(True)
        feats = features.detach() if features is not None else None
        res = loss_fn(z, targets, features=feats, step_frac=step_frac)
        loss = res["loss"]
        if not loss.requires_grad:
            return torch.zeros(z.shape[0], device=z.device)
        (gz,) = torch.autograd.grad(loss, z, allow_unused=True)
    if gz is None:
        return torch.zeros(z.shape[0], device=z.device)
    return gz.abs().sum(dim=-1).detach()


def summarize(gap: torch.Tensor, grad: torch.Tensor, flipped: torch.Tensor) -> dict:
    """Epoch-level mechanism numbers over the probe (all scale-free).

    - ``diag_grad_share_flipped``: share of the probe's gradient mass carried by
      samples whose training label is corrupted. Under CE this tracks the noise rate
      early and rises as clean samples get fitted; a loss that rejects mislabeled
      samples drives it down.
    - ``diag_zero_{clean,flipped}``: fraction with exactly zero gradient.
    - ``diag_rejected_{clean,flipped}``: zero gradient while another class beats the
      label (g > 0), i.e. ignored rather than fitted.
    - ``diag_gap_hist_*`` / ``diag_grad_hist_*``: counts and gradient mass per gap bin
      (``N_BINS`` bins on [−1, 1]); the grad histograms of both groups sum to 1.
    """
    gap = gap.detach().float().cpu()
    grad = grad.detach().float().cpu()
    flipped = flipped.bool().cpu()
    clean = ~flipped
    zero = grad == 0
    total = float(grad.sum())
    bins = ((gap + 1.0) / 2.0 * N_BINS).long().clamp_(0, N_BINS - 1)

    def frac(mask: torch.Tensor, sub: torch.Tensor) -> float | None:
        n = int(sub.sum())
        return float((mask & sub).sum()) / n if n else None

    def hist(sub: torch.Tensor, weights: torch.Tensor | None) -> list[float]:
        w = sub.float() if weights is None else weights * sub.float()
        h = torch.zeros(N_BINS).index_add_(0, bins, w)
        return [round(float(x), 6) for x in h]

    norm = grad / total if total > 0 else torch.zeros_like(grad)
    return {
        "diag_grad_share_flipped": float(norm[flipped].sum()) if total > 0 and bool(flipped.any()) else None,
        "diag_zero_clean": frac(zero, clean),
        "diag_zero_flipped": frac(zero, flipped),
        "diag_rejected_clean": frac(zero & (gap > 0), clean),
        "diag_rejected_flipped": frac(zero & (gap > 0), flipped),
        "diag_gap_hist_clean": hist(clean, None),
        "diag_gap_hist_flipped": hist(flipped, None),
        "diag_grad_hist_clean": hist(clean, norm),
        "diag_grad_hist_flipped": hist(flipped, norm),
    }
