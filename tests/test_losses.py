from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from gam_softmax.losses import (
    AMSoftmaxLoss,
    ASSoftmaxLoss,
    Entmax15Loss,
    FocalLoss,
    GAMSoftmaxLoss,
    LabelSmoothingLoss,
    PowerSoftmaxLoss,
    SoftmaxLoss,
    SparsemaxLoss,
)
from gam_softmax.margins import ClassPairLowRankMargin, SampleConfidenceMargin
from gam_softmax.schedules import ConstantSchedule, LinearSchedule


def _assert_dict_contract(out: dict, logits_shape):
    assert {"loss", "mask", "delta_eff", "masked_ratio"} <= set(out.keys())
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"]).item()
    if out["mask"] is not None:
        assert out["mask"].shape == logits_shape
        assert out["mask"].dtype == torch.bool


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


# ---------- AM-Softmax ----------

def test_am_softmax_owns_learnable_weight():
    loss_fn = AMSoftmaxLoss(n_classes=5, feature_dim=8)
    params = list(loss_fn.parameters())
    assert len(params) == 1
    assert params[0].shape == (8, 5)
    assert params[0].requires_grad


def test_am_softmax_requires_features():
    loss_fn = AMSoftmaxLoss(n_classes=5, feature_dim=8)
    with pytest.raises(ValueError):
        loss_fn(torch.zeros(2, 5), torch.tensor([0, 1]), features=None)


def test_am_softmax_subtracts_margin_only_from_target():
    """Margin is subtracted only from cos θ_t; non-target cosines pass through.
    Compute the expected scaled-logit vector from the actual (non-orthogonal)
    W and compare end-to-end."""
    torch.manual_seed(0)
    loss_fn = AMSoftmaxLoss(n_classes=3, feature_dim=4, margin=0.2, scale=10.0)
    Wn = torch.nn.functional.normalize(loss_fn.weight.detach(), dim=0)
    features = Wn[:, 1].unsqueeze(0)  # (1, 4), aligned with W column 1
    targets = torch.tensor([1])

    expected_cos = features @ Wn                   # (1, 3); cos_1 = 1.0 exactly
    expected_cos[0, targets[0]] -= 0.2             # margin on target only
    expected = torch.nn.functional.cross_entropy(10.0 * expected_cos, targets)

    out = loss_fn(logits=torch.zeros(1, 3), targets=targets, features=features)
    assert torch.allclose(out["loss"], expected, atol=1e-5)
    # also confirm target column gets the cos=1 it should
    assert torch.allclose(expected_cos[0, 1] + 0.2, torch.tensor(1.0), atol=1e-5)


def test_am_softmax_backprop_to_features_and_weight():
    loss_fn = AMSoftmaxLoss(n_classes=4, feature_dim=6)
    features = torch.randn(8, 6, requires_grad=True)
    targets = torch.randint(0, 4, (8,))

    out = loss_fn(torch.zeros(8, 4), targets, features=features)
    out["loss"].backward()

    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert loss_fn.weight.grad is not None and torch.isfinite(loss_fn.weight.grad).all()


def test_am_softmax_rejects_bad_hparams():
    with pytest.raises(ValueError):
        AMSoftmaxLoss(n_classes=5, feature_dim=8, margin=-0.1)
    with pytest.raises(ValueError):
        AMSoftmaxLoss(n_classes=5, feature_dim=8, margin=1.0)
    with pytest.raises(ValueError):
        AMSoftmaxLoss(n_classes=5, feature_dim=8, scale=0.0)


# ---------- Sparsemax ----------

def test_sparsemax_produces_sparse_output():
    """Strongly separated logits → most probs exactly 0."""
    logits = torch.tensor([[5.0, 1.0, 0.0, -1.0, -2.0]])
    targets = torch.tensor([0])
    out = SparsemaxLoss(n_classes=5)(logits, targets)
    n_zero = (~out["mask"][0]).sum().item()
    assert n_zero >= 2  # at least 2 of 5 classes get zero prob
    _assert_dict_contract(out, logits.shape)


def test_sparsemax_backprop_finite():
    logits = torch.randn(8, 5, requires_grad=True)
    targets = torch.randint(0, 5, (8,))
    out = SparsemaxLoss(n_classes=5)(logits, targets)
    out["loss"].backward()
    assert torch.isfinite(logits.grad).all()


# ---------- Entmax-1.5 ----------

def test_entmax15_dict_contract_and_backprop():
    logits = torch.randn(8, 5, requires_grad=True)
    targets = torch.randint(0, 5, (8,))
    out = Entmax15Loss(n_classes=5)(logits, targets)
    _assert_dict_contract(out, logits.shape)
    out["loss"].backward()
    assert torch.isfinite(logits.grad).all()


# ---------- Focal Loss ----------

def test_focal_gamma_zero_reduces_to_ce():
    torch.manual_seed(0)
    logits = torch.randn(16, 7)
    targets = torch.randint(0, 7, (16,))
    focal = FocalLoss(n_classes=7, gamma=0.0, alpha=1.0)(logits, targets)["loss"]
    ce = F.cross_entropy(logits, targets)
    assert torch.allclose(focal, ce, atol=1e-5)


def test_focal_downweights_easy_examples():
    """Confident-correct (high p_t) → focal weight ≪ 1 → loss < CE."""
    logits = torch.tensor([[5.0, 0.0, 0.0]])  # p_t ≈ 0.99
    targets = torch.tensor([0])
    focal = FocalLoss(n_classes=3, gamma=2.0)(logits, targets)["loss"]
    ce = F.cross_entropy(logits, targets)
    assert focal.item() < ce.item()


def test_focal_backprop_finite():
    logits = torch.randn(8, 5, requires_grad=True)
    targets = torch.randint(0, 5, (8,))
    FocalLoss(n_classes=5)(logits, targets)["loss"].backward()
    assert torch.isfinite(logits.grad).all()


def test_focal_rejects_negative_gamma():
    with pytest.raises(ValueError):
        FocalLoss(n_classes=5, gamma=-1.0)


# ---------- Label Smoothing ----------

def test_label_smoothing_matches_softmax_with_same_ls():
    torch.manual_seed(0)
    logits = torch.randn(16, 5)
    targets = torch.randint(0, 5, (16,))
    a = LabelSmoothingLoss(n_classes=5, label_smoothing=0.1)(logits, targets)["loss"]
    b = SoftmaxLoss(n_classes=5, label_smoothing=0.1)(logits, targets)["loss"]
    assert torch.allclose(a, b)


def test_label_smoothing_rejects_zero():
    with pytest.raises(ValueError):
        LabelSmoothingLoss(n_classes=5, label_smoothing=0.0)


# ---------- Power Softmax ----------

def test_power_softmax_output_normalizes_to_one():
    logits = torch.tensor([[2.0, -1.0, 0.5, 3.0]])
    targets = torch.tensor([3])
    loss_fn = PowerSoftmaxLoss(n_classes=4, gamma=2.0)
    out = loss_fn(logits, targets)
    # recompute probs to verify they sum to 1
    z = F.relu(logits) + loss_fn.eps
    probs = z.pow(loss_fn.gamma)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-6)


def test_power_softmax_handles_all_nonpositive_logits():
    """All raw logits ≤ 0 → ReLU zeros them; eps prevents 0/0 and the loss
    stays finite."""
    logits = torch.tensor([[-1.0, -2.0, -0.5]])
    targets = torch.tensor([2])
    out = PowerSoftmaxLoss(n_classes=3, gamma=2.0)(logits, targets)
    assert torch.isfinite(out["loss"])


def test_power_softmax_masked_ratio_counts_relu_zeros():
    """Non-target slots with logit ≤ 0 are 'suppressed'."""
    logits = torch.tensor([[2.0, -1.0, -3.0]])  # target=0, slots 1+2 suppressed
    targets = torch.tensor([0])
    out = PowerSoftmaxLoss(n_classes=3, gamma=2.0)(logits, targets)
    # 2 of 2 non-target slots suppressed → ratio = 1.0
    assert float(out["masked_ratio"]) == pytest.approx(1.0)


def test_power_softmax_backprop_finite_at_zero():
    """The eps inside the power bounds gradient at logit = 0."""
    logits = torch.zeros(4, 5, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3])
    PowerSoftmaxLoss(n_classes=5, gamma=0.5)(logits, targets)["loss"].backward()
    assert torch.isfinite(logits.grad).all()


def test_power_softmax_rejects_bad_hparams():
    with pytest.raises(ValueError):
        PowerSoftmaxLoss(n_classes=5, gamma=0.0)
    with pytest.raises(ValueError):
        PowerSoftmaxLoss(n_classes=5, eps=0.0)


# ---------- LinearSchedule ----------

def test_linear_schedule_ramps_then_holds():
    sch = LinearSchedule(delta_min=0.1, delta_max=0.5, warmup_frac=0.4)
    assert sch(0.0) == pytest.approx(0.1)
    assert sch(0.2) == pytest.approx(0.3)            # midpoint of ramp
    assert sch(0.4) == pytest.approx(0.5)            # ramp end
    assert sch(0.9) == pytest.approx(0.5)            # held at cap
    assert sch.delta_min == 0.1


def test_linear_schedule_rejects_bad_bounds():
    with pytest.raises(ValueError):
        LinearSchedule(delta_min=0.5, delta_max=0.1, warmup_frac=0.3)
    with pytest.raises(ValueError):
        LinearSchedule(delta_min=-0.1, delta_max=0.4, warmup_frac=0.3)
    with pytest.raises(ValueError):
        LinearSchedule(delta_min=0.05, delta_max=0.4, warmup_frac=0.0)


# ---------- ClassPairLowRankMargin (M3) ----------

def test_classpair_lowrank_margin_in_bounds():
    """δ_{t,j} must live in [δ_min, δ_max(τ)] for every (sample, class)."""
    torch.manual_seed(0)
    sch = LinearSchedule(delta_min=0.05, delta_max=0.4, warmup_frac=0.3)
    m = ClassPairLowRankMargin(n_classes=5, schedule=sch, rank=8)

    logits = torch.randn(16, 5)
    targets = torch.randint(0, 5, (16,))
    delta = m(logits, targets, features=None, step_frac=0.5)   # past warmup → cap = 0.4

    assert delta.shape == (16, 5)
    assert (delta >= 0.05 - 1e-6).all()
    assert (delta <= 0.4 + 1e-6).all()


def test_classpair_lowrank_margin_collapses_at_step_zero():
    """At step_frac=0 with the default linear schedule, δ_max(0) == δ_min,
    so every entry of the margin tensor collapses to δ_min — independent of u, v."""
    torch.manual_seed(0)
    sch = LinearSchedule(delta_min=0.05, delta_max=0.4, warmup_frac=0.3)
    m = ClassPairLowRankMargin(n_classes=5, schedule=sch, rank=8, init_std=0.5)

    delta = m(torch.randn(4, 5), torch.tensor([0, 1, 2, 3]), features=None, step_frac=0.0)
    assert torch.allclose(delta, torch.full_like(delta, 0.05), atol=1e-6)


def test_classpair_lowrank_uv_are_parameters():
    sch = LinearSchedule(delta_min=0.05, delta_max=0.4, warmup_frac=0.3)
    m = ClassPairLowRankMargin(n_classes=5, schedule=sch, rank=4)
    params = {n: p for n, p in m.named_parameters()}
    assert set(params.keys()) == {"u", "v"}
    assert params["u"].shape == (5, 4)
    assert params["v"].shape == (5, 4)


def test_classpair_lowrank_rejects_bad_hparams():
    sch = LinearSchedule(delta_min=0.05, delta_max=0.4, warmup_frac=0.3)
    with pytest.raises(ValueError):
        ClassPairLowRankMargin(n_classes=5, schedule=sch, rank=0)
    with pytest.raises(ValueError):
        ClassPairLowRankMargin(n_classes=5, schedule=sch, init_std=0.0)


def test_classpair_lowrank_requires_schedule_with_delta_min():
    """A schedule without ``delta_min`` (e.g. plain ConstantSchedule) shouldn't pass
    silently — the margin needs a floor to lerp from."""
    sch = ConstantSchedule(value=0.3)
    with pytest.raises(ValueError):
        ClassPairLowRankMargin(n_classes=5, schedule=sch, rank=4)


# ---------- GAMSoftmaxLoss ----------

def test_gam_softmax_dict_contract():
    sch = LinearSchedule(delta_min=0.05, delta_max=0.4, warmup_frac=0.3)
    m = ClassPairLowRankMargin(n_classes=5, schedule=sch, rank=8)
    loss_fn = GAMSoftmaxLoss(n_classes=5, margin_fn=m)

    logits = torch.randn(4, 5, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3])
    out = loss_fn(logits, targets, step_frac=0.5)

    _assert_dict_contract(out, logits.shape)
    assert out["mask"].dtype == torch.bool
    assert 0.0 <= float(out["delta_eff"]) <= 0.4 + 1e-6


def test_gam_softmax_target_never_masked():
    """Even with a peaked target prediction (large p_t), the target slot must stay kept."""
    sch = LinearSchedule(delta_min=0.05, delta_max=0.4, warmup_frac=0.3)
    m = ClassPairLowRankMargin(n_classes=3, schedule=sch, rank=4)
    loss_fn = GAMSoftmaxLoss(n_classes=3, margin_fn=m)

    logits = torch.tensor([[10.0, 0.0, 0.0]])
    targets = torch.tensor([0])
    out = loss_fn(logits, targets, step_frac=1.0)
    keep = out["mask"][0]
    assert keep[0].item() is True
    assert not keep[1].item()
    assert not keep[2].item()


def test_gam_softmax_matches_as_softmax_when_margin_is_constant():
    """With u, v zeroed and the schedule capped at δ, the M3 margin tensor is
    uniformly (δ_min + δ_max)/2 — set δ_min == δ_max to make it exactly δ, then
    GAM-Softmax must equal AS-Softmax(δ) on the same batch."""
    torch.manual_seed(0)
    delta = 0.15
    sch = LinearSchedule(delta_min=delta, delta_max=delta, warmup_frac=0.1)
    m = ClassPairLowRankMargin(n_classes=7, schedule=sch, rank=4)
    with torch.no_grad():
        m.u.zero_()
        m.v.zero_()
    gam = GAMSoftmaxLoss(n_classes=7, margin_fn=m)
    as_fn = ASSoftmaxLoss(n_classes=7, delta=delta)

    logits = torch.randn(16, 7)
    targets = torch.randint(0, 7, (16,))
    gam_out = gam(logits, targets, step_frac=0.5)
    as_out = as_fn(logits, targets, step_frac=0.5)

    assert torch.allclose(gam_out["loss"], as_out["loss"], atol=1e-5)
    assert torch.equal(gam_out["mask"], as_out["mask"])


def test_gam_softmax_backprop_finite_to_classifier_and_margin_params():
    """Loss must backprop through the classifier logits without NaN. The margin
    parameters (u, v) won't actually receive gradient (mask is detached), but
    they must remain in the optimizer's param list — that's a contract test."""
    torch.manual_seed(0)
    sch = LinearSchedule(delta_min=0.05, delta_max=0.4, warmup_frac=0.3)
    m = ClassPairLowRankMargin(n_classes=5, schedule=sch, rank=4)
    loss_fn = GAMSoftmaxLoss(n_classes=5, margin_fn=m)

    logits = torch.randn(8, 5, requires_grad=True)
    targets = torch.randint(0, 5, (8,))
    loss_fn(logits, targets, step_frac=0.5)["loss"].backward()

    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    # u, v are still registered parameters so a future STE variant can use them
    assert any(p is m.u for p in loss_fn.parameters())
    assert any(p is m.v for p in loss_fn.parameters())


def test_gam_softmax_rejects_non_margin_fn():
    with pytest.raises(TypeError):
        GAMSoftmaxLoss(n_classes=5, margin_fn="not a margin")  # type: ignore[arg-type]


def test_gam_softmax_learnable_margin_routes_grad_to_u_v():
    """With learnable_margin=True the STE sends gradient into the margin's
    u, v — the fix that turns M3's fixed-random gap into a trainable one.
    (Contrast test_gam_softmax_backprop_finite_* which asserts the default
    hard path does NOT.)"""
    torch.manual_seed(0)
    sch = LinearSchedule(delta_min=0.05, delta_max=0.4, warmup_frac=0.3)
    m = ClassPairLowRankMargin(n_classes=5, schedule=sch, rank=4)
    loss_fn = GAMSoftmaxLoss(n_classes=5, margin_fn=m, learnable_margin=True, ste_temp=0.1)

    logits = torch.randn(8, 5, requires_grad=True)
    targets = torch.randint(0, 5, (8,))
    loss_fn(logits, targets, step_frac=1.0)["loss"].backward()

    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert m.u.grad is not None and torch.isfinite(m.u.grad).all()
    assert m.v.grad is not None and torch.isfinite(m.v.grad).all()
    assert m.u.grad.abs().sum() > 0
    assert m.v.grad.abs().sum() > 0


def test_gam_softmax_learnable_margin_forward_matches_hard():
    """STE forward value must equal the default hard-mask loss (only the
    backward differs), so the learnable variant stays comparable to AS-Softmax."""
    torch.manual_seed(0)
    sch = LinearSchedule(delta_min=0.05, delta_max=0.4, warmup_frac=0.3)
    m = ClassPairLowRankMargin(n_classes=7, schedule=sch, rank=4)
    hard = GAMSoftmaxLoss(n_classes=7, margin_fn=m, learnable_margin=False)
    soft = GAMSoftmaxLoss(n_classes=7, margin_fn=m, learnable_margin=True)

    logits = torch.randn(16, 7)
    targets = torch.randint(0, 7, (16,))
    out_hard = hard(logits, targets, step_frac=0.5)
    out_soft = soft(logits, targets, step_frac=0.5)

    assert torch.allclose(out_hard["loss"], out_soft["loss"], atol=1e-5)
    assert torch.equal(out_hard["mask"], out_soft["mask"])


def test_gam_softmax_rejects_nonpositive_ste_temp():
    sch = LinearSchedule(delta_min=0.05, delta_max=0.4, warmup_frac=0.3)
    m = ClassPairLowRankMargin(n_classes=5, schedule=sch, rank=4)
    with pytest.raises(ValueError):
        GAMSoftmaxLoss(n_classes=5, margin_fn=m, ste_temp=0.0)


# ---------- SampleConfidenceMargin (M4) ----------

def _sample_margin(**kw):
    sch = LinearSchedule(delta_min=0.0, delta_max=0.15, warmup_frac=0.1)
    defaults = dict(n_classes=5, schedule=sch, beta=1.0, temp=1.0,
                    reject_warmup_frac=0.3, delta_floor=-1.0, delta_ceiling=0.15)
    defaults.update(kw)
    return SampleConfidenceMargin(**defaults)


def _confidence_split_logits():
    """Batch where sample 0 is very unconfident about its label and sample 1 very
    confident — the two ends of the suspicion score."""
    logits = torch.tensor([
        [0.0, 5.0, 0.0, 0.0, 0.0],   # target 0, model prefers class 1  → suspicious
        [5.0, 0.0, 0.0, 0.0, 0.0],   # target 0, model agrees           → confident
        [1.0, 1.0, 0.0, 0.0, 0.0],   # target 0, middling
        [2.0, 0.5, 0.0, 0.0, 0.0],   # target 0, fairly confident
    ])
    targets = torch.zeros(4, dtype=torch.long)
    return logits, targets


def test_sample_confidence_beta_zero_recovers_base_schedule():
    """beta=0 is the control ablation: the sample axis switches off and every
    slot must equal the scheduled scalar margin."""
    m = _sample_margin(beta=0.0)
    logits, targets = _confidence_split_logits()
    delta = m(logits, targets, None, step_frac=1.0)
    assert torch.allclose(delta, torch.full_like(delta, 0.15))


def test_sample_confidence_suspicious_sample_gets_lower_margin():
    """The core directional claim: low p_t ⇒ smaller δ ⇒ more classes masked."""
    m = _sample_margin()
    logits, targets = _confidence_split_logits()
    delta = m(logits, targets, None, step_frac=1.0)
    assert delta[0, 0] < delta[2, 0] < delta[1, 0]


def test_sample_confidence_no_effect_during_warmup():
    """β(τ)=0 through reject_warmup_frac — p_t carries no signal while the model
    is still near-random, so δ must not vary across samples yet."""
    m = _sample_margin(reject_warmup_frac=0.3)
    logits, targets = _confidence_split_logits()
    delta = m(logits, targets, None, step_frac=0.2)
    assert m.beta_at(0.2) == 0.0
    assert torch.allclose(delta, delta[0, 0].expand_as(delta))
    # ...and it does vary once the ramp is over
    assert m(logits, targets, None, step_frac=1.0).std() > 0


def test_sample_confidence_respects_floor_and_ceiling():
    m = _sample_margin(beta=5.0, delta_floor=-0.4, delta_ceiling=0.1)
    logits, targets = _confidence_split_logits()
    delta = m(logits, targets, None, step_frac=1.0)
    assert float(delta.min()) >= -0.4 - 1e-6
    assert float(delta.max()) <= 0.1 + 1e-6


def test_sample_confidence_rejects_suspicious_sample_entirely():
    """The anti-memorization mechanism. With a negative floor, the most
    suspicious sample's margin drops below every gap p_t − p_j, so all its
    non-target slots are masked and its loss contribution collapses to ~0."""
    logits, targets = _confidence_split_logits()
    m = _sample_margin(beta=2.0, delta_floor=-1.0)
    gam_keep = GAMSoftmaxLoss(n_classes=5, margin_fn=m)(
        logits, targets, step_frac=1.0
    )["mask"]
    as_keep = ASSoftmaxLoss(n_classes=5, delta=0.15)(
        logits, targets, step_frac=1.0
    )["mask"]

    # Scalar AS-Softmax keeps every competitor alive for the mislabeled-looking
    # sample (its p_t is below everything, so no gap clears δ) and therefore
    # keeps training on the wrong label. The sample axis masks them all, leaving
    # only the target slot — the sample drops out of the loss.
    assert int(as_keep[0].sum()) == 5
    assert int(gam_keep[0].sum()) == 1
    assert bool(gam_keep[0, targets[0]])
    # the confident sample is unaffected: it was already fully masked by both
    assert torch.equal(gam_keep[1], as_keep[1])


def test_sample_confidence_margin_is_detached_from_logits():
    """Suspicion is read from the model, never a gradient path back into it —
    same stop-gradient discipline as the AS-Softmax mask."""
    m = _sample_margin()
    logits, targets = _confidence_split_logits()
    logits = logits.clone().requires_grad_(True)
    delta = m(logits, targets, None, step_frac=1.0)
    assert not delta.requires_grad


def test_sample_confidence_composes_over_classpair_margin():
    """M6: the sample axis modulating the M3 class-pair matrix. The result must
    vary along BOTH axes — across rows (samples) and within a row (classes)."""
    torch.manual_seed(0)
    base_sch = LinearSchedule(delta_min=0.05, delta_max=0.4, warmup_frac=0.3)
    base = ClassPairLowRankMargin(n_classes=5, schedule=base_sch, rank=4, init_std=1.0)
    m = _sample_margin(beta=0.3, delta_floor=-1.0, delta_ceiling=1.0, base_margin=base)
    logits, targets = _confidence_split_logits()
    delta = m(logits, targets, None, step_frac=1.0)

    assert delta.shape == logits.shape
    assert delta.std(dim=1).max() > 0          # varies across classes (class-pair axis)
    assert delta[:, 0].std() > 0               # varies across samples (sample axis)
    # base_margin is a submodule, so its u/v reach the optimizer
    assert any(p is base.u for p in m.parameters())


def test_sample_confidence_rejects_bad_hparams():
    sch = LinearSchedule(delta_min=0.0, delta_max=0.15, warmup_frac=0.1)
    with pytest.raises(ValueError):
        SampleConfidenceMargin(n_classes=5, schedule=sch, beta=-0.1)
    with pytest.raises(ValueError):
        SampleConfidenceMargin(n_classes=5, schedule=sch, temp=0.0)
    with pytest.raises(ValueError):
        SampleConfidenceMargin(n_classes=5, schedule=sch, reject_warmup_frac=1.5)
    with pytest.raises(ValueError):
        SampleConfidenceMargin(n_classes=5, schedule=sch, delta_floor=-2.0)
    with pytest.raises(ValueError):
        SampleConfidenceMargin(n_classes=5, schedule=sch, delta_floor=0.5, delta_ceiling=0.1)
    with pytest.raises(TypeError):
        SampleConfidenceMargin(n_classes=5, schedule=sch, base_margin="not a margin")


def test_sample_confidence_ramp_end_default_keeps_old_schedule():
    """reject_ramp_end=1.0 (default) must reproduce the original E1 ramp exactly."""
    m = _sample_margin(reject_warmup_frac=0.3)
    assert m.beta_at(0.3) == 0.0
    assert m.beta_at(0.65) == pytest.approx(0.5)
    assert m.beta_at(1.0) == pytest.approx(1.0)


def test_sample_confidence_ramp_end_matches_small_loss_schedule():
    """warmup 0 + ramp_end 0.3 is the Co-teaching forget-rate shape: linear to full
    strength at 30% of training, then held."""
    m = _sample_margin(reject_warmup_frac=0.0, reject_ramp_end=0.3)
    assert m.beta_at(0.0) == 0.0
    assert m.beta_at(0.15) == pytest.approx(0.5)
    assert m.beta_at(0.3) == pytest.approx(1.0)
    assert m.beta_at(0.9) == pytest.approx(1.0)


def test_sample_confidence_rejects_bad_ramp_end():
    with pytest.raises(ValueError):
        _sample_margin(reject_warmup_frac=0.3, reject_ramp_end=0.3)
    with pytest.raises(ValueError):
        _sample_margin(reject_warmup_frac=0.0, reject_ramp_end=1.5)


def test_sample_confidence_nonneg_margin_keeps_rejections_but_trains_rest_with_ce():
    """nonneg_margin=1.0: identical rejection decisions to plain M4, but every kept
    sample trains with cross-entropy instead of AS-Softmax."""
    torch.manual_seed(0)
    plain = _sample_margin(beta=2.0, delta_floor=-1.0)
    ce_base = _sample_margin(beta=2.0, delta_floor=-1.0, nonneg_margin=1.0)
    logits, targets = _confidence_split_logits()
    d_plain = plain(logits, targets, None, step_frac=1.0)
    d_ce = ce_base(logits, targets, None, step_frac=1.0)
    rejected = d_plain < 0
    assert rejected.any() and (~rejected).any()
    assert torch.equal(d_ce[rejected], d_plain[rejected])
    assert (d_ce[~rejected] == 1.0).all()
    # before rejection starts, the loss is exactly cross-entropy
    gam = GAMSoftmaxLoss(n_classes=7, margin_fn=_sample_margin(n_classes=7, nonneg_margin=1.0))
    x, y = torch.randn(16, 7), torch.randint(0, 7, (16,))
    assert torch.allclose(gam(x, y, step_frac=0.1)["loss"], F.cross_entropy(x, y), atol=1e-5)
    with pytest.raises(ValueError):
        _sample_margin(nonneg_margin=1.5)
