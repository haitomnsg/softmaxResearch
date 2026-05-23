from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from gam_softmax.losses import ASSoftmaxLoss, SoftmaxLoss


def test_softmax_loss_matches_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(8, 5, requires_grad=True)
    targets = torch.randint(0, 5, (8,))

    loss_fn = SoftmaxLoss(n_classes=5)
    out = loss_fn(logits, targets)
    expected = F.cross_entropy(logits, targets)

    assert torch.allclose(out["loss"], expected, atol=1e-6)


def test_softmax_loss_dict_contract():
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 0])

    out = SoftmaxLoss(n_classes=3)(logits, targets)

    assert {"loss", "mask", "delta_eff", "masked_ratio"} <= set(out.keys())
    assert out["mask"] is None
    assert out["delta_eff"] is None
    assert float(out["masked_ratio"]) == 0.0


def test_softmax_loss_backprop():
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 0])

    out = SoftmaxLoss(n_classes=3)(logits, targets)
    out["loss"].backward()

    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
    assert torch.isfinite(logits.grad).all()


def test_softmax_loss_label_smoothing_differs():
    torch.manual_seed(0)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))

    plain = SoftmaxLoss(n_classes=5)(logits, targets)["loss"]
    smoothed = SoftmaxLoss(n_classes=5, label_smoothing=0.1)(logits, targets)["loss"]

    assert not torch.allclose(plain, smoothed)


# ---------- AS-Softmax ----------

def test_as_softmax_reduces_to_ce_at_delta_one():
    """At δ=1, the mask condition p_t - p_i ≥ 1 is never satisfied for finite
    logits (probs are strictly < 1), so AS-Softmax === standard CE."""
    torch.manual_seed(0)
    logits = torch.randn(16, 7)
    targets = torch.randint(0, 7, (16,))

    as_out = ASSoftmaxLoss(n_classes=7, delta=1.0)(logits, targets)
    expected = F.cross_entropy(logits, targets)

    assert torch.allclose(as_out["loss"], expected, atol=1e-5)
    assert float(as_out["masked_ratio"]) == 0.0


def test_as_softmax_target_never_masked():
    logits = torch.tensor([[10.0, 0.0, 0.0]])  # p_t ≈ 1, others ≈ 0
    targets = torch.tensor([0])

    out = ASSoftmaxLoss(n_classes=3, delta=0.1)(logits, targets)
    keep = out["mask"]  # (1, 3) bool

    assert keep[0, 0].item() is True  # target kept
    assert keep[0, 1].item() is False
    assert keep[0, 2].item() is False


def test_as_softmax_mask_matches_threshold():
    """Hand-built: logits [2,1,0] -> p ≈ [0.665, 0.244, 0.090].
    p_t - p_1 = 0.421, p_t - p_2 = 0.574. At δ=0.5, class 2 masked, class 1 kept."""
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    targets = torch.tensor([0])

    out = ASSoftmaxLoss(n_classes=3, delta=0.5)(logits, targets)
    keep = out["mask"][0].tolist()

    assert keep == [True, True, False]


def test_as_softmax_masked_ratio():
    """logits picked so 1 of 2 non-target classes is masked per sample → ratio 0.5."""
    logits = torch.tensor([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    targets = torch.tensor([0, 0])

    out = ASSoftmaxLoss(n_classes=3, delta=0.5)(logits, targets)
    assert float(out["masked_ratio"]) == pytest.approx(0.5)


def test_as_softmax_backprop_finite():
    torch.manual_seed(0)
    logits = torch.randn(8, 5, requires_grad=True)
    targets = torch.randint(0, 5, (8,))

    out = ASSoftmaxLoss(n_classes=5, delta=0.1)(logits, targets)
    out["loss"].backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_as_softmax_warmup_ramps_delta_eff():
    loss_fn = ASSoftmaxLoss(n_classes=5, delta=0.4, warmup_frac=0.2)

    assert loss_fn._delta_eff(0.0) == pytest.approx(0.0)
    assert loss_fn._delta_eff(0.1) == pytest.approx(0.2)
    assert loss_fn._delta_eff(0.2) == pytest.approx(0.4)
    assert loss_fn._delta_eff(0.5) == pytest.approx(0.4)


def test_as_softmax_dict_contract():
    logits = torch.randn(4, 5, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3])

    out = ASSoftmaxLoss(n_classes=5, delta=0.1)(logits, targets)

    assert {"loss", "mask", "delta_eff", "masked_ratio"} <= set(out.keys())
    assert out["mask"].dtype == torch.bool
    assert out["mask"].shape == logits.shape
    assert float(out["delta_eff"]) == pytest.approx(0.1)


def test_as_softmax_rejects_bad_delta():
    with pytest.raises(ValueError):
        ASSoftmaxLoss(n_classes=5, delta=0.0)
    with pytest.raises(ValueError):
        ASSoftmaxLoss(n_classes=5, delta=1.5)
    with pytest.raises(ValueError):
        ASSoftmaxLoss(n_classes=5, delta=0.1, warmup_frac=1.0)
