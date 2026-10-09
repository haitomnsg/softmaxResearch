"""Phase E mechanism diagnostics (`gam_softmax.eval.mechanism`) and the rejection
geometry the paper's propositions state, checked against the implementation:

- Prop. 1: a GAM sample has exactly zero loss and gradient iff g ≤ −δ_i.
- Prop. 3: `abs_gap` rejects exactly the leads in `rejection_band` = [g_lo, g_hi], with
  g_hi < β − δ_base: leads above g_hi (the most confidently contradicted labels) are
  trained on, against the leading class only.
- Prop. 2: M4's batch-relative suspicion is invariant to affine maps of the batch's
  p_t, so the share it flags cannot follow the noise rate.
"""

from __future__ import annotations

import pytest
import torch

from gam_softmax.eval.mechanism import N_BINS, label_gap, per_sample_grad_l1, summarize
from gam_softmax.losses import ASSoftmaxLoss, GAMSoftmaxLoss, SoftmaxLoss
from gam_softmax.losses.noisy_label import GCELoss, SmallLossLoss
from gam_softmax.margins import AbsoluteGapMargin, SampleConfidenceMargin
from gam_softmax.schedules import ConstantSchedule

N = 5
DELTA = 0.15


def _logits_for_gap(gaps: list[float]) -> torch.Tensor:
    """Rows with target class 0 and class 1 leading it by g (g < 0: class 0 leads)."""
    rows = []
    for g in gaps:
        r = 0.05 * (1.0 - abs(g))                  # mass on each of the 3 other classes
        p_t = (1.0 - abs(g) - 3 * r) / 2.0
        p_a, p_b = (p_t, p_t + g) if g >= 0 else (p_t - g, p_t)
        rows.append([p_a, p_b, r, r, r])
    return torch.log(torch.tensor(rows, dtype=torch.float64)).float()


def _gap_loss(temp: float = 0.1, beta: float = 1.0) -> GAMSoftmaxLoss:
    m = AbsoluteGapMargin(N, ConstantSchedule(DELTA), beta=beta, temp=temp,
                          reject_warmup_frac=0.0, reject_ramp_end=0.3,
                          delta_floor=-1.0, delta_ceiling=DELTA)
    return GAMSoftmaxLoss(N, m)


# ----------------------------------------------------------------------------- gap + grads

def test_label_gap_sign_and_value():
    gaps = [-0.5, -0.1, 0.0, 0.2, 0.7]
    z = _logits_for_gap(gaps)
    g = label_gap(z, torch.zeros(len(gaps), dtype=torch.long))
    assert torch.allclose(g, torch.tensor(gaps), atol=1e-5)


def test_ce_never_zero_and_share_is_ratio():
    z = torch.randn(8, N)
    y = torch.randint(0, N, (8,))
    grad = per_sample_grad_l1(SoftmaxLoss(N), z, y, step_frac=0.5)
    assert (grad > 0).all()
    # CE: ‖p − e_t‖₁ = 2(1 − p_t), divided by B by the mean reduction
    p_t = torch.softmax(z, -1).gather(1, y[:, None]).squeeze(1)
    assert torch.allclose(grad, 2 * (1 - p_t) / 8, atol=1e-6)
    flipped = torch.tensor([True, False] * 4)
    d = summarize(label_gap(z, y), grad, flipped)
    assert d["diag_grad_share_flipped"] == pytest.approx(float(grad[flipped].sum() / grad.sum()), rel=1e-5)
    assert d["diag_zero_clean"] == 0.0 and d["diag_zero_flipped"] == 0.0


def test_as_softmax_fitted_samples_get_zero_grad_but_are_not_rejected():
    # label 0 wins by 0.5 (≥ δ) → every slot masked → loss 0; label wins by 0.05 → trained
    z = _logits_for_gap([-0.5, -0.05])
    y = torch.zeros(2, dtype=torch.long)
    grad = per_sample_grad_l1(ASSoftmaxLoss(N, delta=DELTA), z, y, step_frac=1.0)
    assert grad[0] == 0 and grad[1] > 0
    d = summarize(label_gap(z, y), grad, torch.tensor([False, False]))
    assert d["diag_zero_clean"] == 0.5
    assert d["diag_rejected_clean"] == 0.0      # g < 0: fitted, not rejected
    assert d["diag_grad_share_flipped"] is None


def test_small_loss_rejected_samples_have_zero_grad():
    z = _logits_for_gap([-0.5, -0.4, 0.3, 0.6])
    y = torch.zeros(4, dtype=torch.long)
    grad = per_sample_grad_l1(SmallLossLoss(N, forget_rate=0.5, ramp_frac=0.0), z, y, step_frac=1.0)
    assert (grad[:2] > 0).all() and (grad[2:] == 0).all()
    d = summarize(label_gap(z, y), grad, torch.tensor([False, False, True, True]))
    assert d["diag_rejected_flipped"] == 1.0
    assert d["diag_grad_share_flipped"] == 0.0


def test_histograms_count_and_normalise():
    z = torch.randn(40, N)
    y = torch.randint(0, N, (40,))
    flipped = torch.rand(40) < 0.4
    grad = per_sample_grad_l1(GCELoss(N, q=0.7), z, y, step_frac=0.5)
    d = summarize(label_gap(z, y), grad, flipped)
    assert len(d["diag_gap_hist_clean"]) == N_BINS
    assert sum(d["diag_gap_hist_clean"]) + sum(d["diag_gap_hist_flipped"]) == pytest.approx(40)
    mass = sum(d["diag_grad_hist_clean"]) + sum(d["diag_grad_hist_flipped"])
    assert mass == pytest.approx(1.0, abs=1e-4)
    assert sum(d["diag_grad_hist_flipped"]) == pytest.approx(d["diag_grad_share_flipped"], abs=1e-4)


# ----------------------------------------------------------------------------- rejection geometry

def test_abs_gap_band_values():
    """The numbers the paper quotes for the frozen recipe (β = 1, δ_base = 0.15)."""
    lo, hi = _gap_loss(temp=0.1).margin_fn.rejection_band(DELTA)
    assert lo == pytest.approx(0.038, abs=1e-3) and hi == pytest.approx(0.850, abs=1e-3)
    lo3, hi3 = _gap_loss(temp=0.3).margin_fn.rejection_band(DELTA)
    assert lo3 == pytest.approx(0.265, abs=1e-3) and hi3 == pytest.approx(0.635, abs=1e-3)
    # the margin bottoms out at δ_base − β, so the band ends below β − δ_base
    assert hi < 1.0 - DELTA + 1e-6
    assert _gap_loss(beta=0.1).margin_fn.rejection_band(DELTA) is None


@pytest.mark.parametrize("temp", [0.1, 0.3])
def test_abs_gap_zero_grad_matches_band(temp):
    loss = _gap_loss(temp=temp)
    lo, hi = loss.margin_fn.rejection_band(DELTA)
    gaps = [x / 100 for x in range(-90, 100, 1)]
    # stay clear of the band edges and of the fitted-region edge, where float32 decides
    gaps = [g for g in gaps if min(abs(g - lo), abs(g - hi), abs(g + DELTA)) > 0.006]
    z = _logits_for_gap(gaps)
    y = torch.zeros(len(gaps), dtype=torch.long)
    grad = per_sample_grad_l1(loss, z, y, step_frac=1.0)      # full β
    for g, gr in zip(gaps, grad.tolist()):
        fitted = g <= -DELTA
        rejected = lo <= g <= hi
        assert (gr == 0) == (fitted or rejected), f"g={g}: grad={gr}"


def test_escape_region_trains_against_the_leader_only():
    """g > g_hi: δ bottoms out at δ_base − β > −g, so the leading class stays in the loss."""
    loss = _gap_loss(temp=0.1)
    z = _logits_for_gap([0.95])
    res = loss(z, torch.zeros(1, dtype=torch.long), step_frac=1.0)
    keep = res["mask"][0]
    assert keep.tolist() == [True, True, False, False, False]
    assert res["loss"] > 0


def test_m4_suspicion_is_affine_invariant():
    """Prop. 2: M4's s_i depends on p_t only through its batch z-score."""
    m = SampleConfidenceMargin(N, ConstantSchedule(DELTA), beta=1.0, temp=1.0)
    p_t = torch.tensor([0.9, 0.7, 0.4, 0.2, 0.05, 0.6])
    for a, b in [(1.0, 0.0), (0.5, 0.3), (0.2, 0.01)]:
        q = a * p_t + b
        probs = torch.stack([q, *[(1 - q) / (N - 1)] * (N - 1)], dim=1)
        s = m._suspicion(torch.log(probs), torch.zeros(len(q), dtype=torch.long), 1.0)
        if a == 1.0:
            ref = s
        assert torch.allclose(s, ref, atol=1e-5)
