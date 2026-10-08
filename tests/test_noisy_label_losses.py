from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from gam_softmax.losses import GCELoss, SCELoss, SmallLossLoss


def _batch(b=32, n=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, generator=g, requires_grad=True), torch.randint(0, n, (b,), generator=g)


@pytest.mark.parametrize("loss_fn", [GCELoss(10), SCELoss(10), SmallLossLoss(10)])
def test_dict_contract_and_backprop(loss_fn):
    logits, y = _batch()
    out = loss_fn(logits, y, step_frac=0.5)
    assert {"loss", "mask", "delta_eff", "masked_ratio"} <= set(out)
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    out["loss"].backward()
    assert torch.isfinite(logits.grad).all()


def test_gce_small_q_approaches_ce():
    logits, y = _batch()
    gce = GCELoss(10, q=1e-4)(logits, y)["loss"]
    assert torch.allclose(gce, F.cross_entropy(logits, y), atol=1e-2)


def test_gce_q_one_is_mae():
    logits, y = _batch()
    p_t = F.softmax(logits, -1).gather(1, y[:, None]).squeeze(1)
    assert torch.allclose(GCELoss(10, q=1.0)(logits, y)["loss"], (1 - p_t).mean())


def test_gce_bounded_on_confidently_wrong_sample():
    # the robustness property: loss stays ≤ 1/q however wrong the label is, CE blows up
    logits = torch.tensor([[50.0, -50.0]])
    y = torch.tensor([1])
    assert GCELoss(2, q=0.7)(logits, y)["loss"] <= 1 / 0.7 + 1e-6
    assert F.cross_entropy(logits, y) > 50


def test_sce_alpha_only_is_ce_and_beta_only_is_scaled_mae():
    logits, y = _batch()
    assert torch.allclose(SCELoss(10, alpha=1.0, beta=0.0)(logits, y)["loss"], F.cross_entropy(logits, y))
    p_t = F.softmax(logits, -1).gather(1, y[:, None]).squeeze(1)
    assert torch.allclose(SCELoss(10, alpha=0.0, beta=1.0)(logits, y)["loss"], (4.0 * (1 - p_t)).mean())


def test_small_loss_zero_rate_is_ce():
    logits, y = _batch()
    out = SmallLossLoss(10, forget_rate=0.0)(logits, y, step_frac=1.0)
    assert torch.allclose(out["loss"], F.cross_entropy(logits, y))
    assert out["masked_ratio"].item() == 0.0


def test_small_loss_keeps_lowest_loss_samples():
    logits, y = _batch(b=10)
    per = F.cross_entropy(logits, y, reduction="none")
    out = SmallLossLoss(10, forget_rate=0.4, ramp_frac=0.0)(logits, y)
    assert torch.allclose(out["loss"], per.sort().values[:6].mean())
    assert out["masked_ratio"].item() == pytest.approx(0.4)


def test_small_loss_rate_ramps_then_holds():
    fn = SmallLossLoss(10, forget_rate=0.4, ramp_frac=0.2)
    assert fn.current_rate(0.0) == 0.0
    assert fn.current_rate(0.1) == pytest.approx(0.2)
    assert fn.current_rate(0.2) == pytest.approx(0.4)
    assert fn.current_rate(0.9) == pytest.approx(0.4)


def test_small_loss_no_grad_to_rejected_samples():
    logits, y = _batch(b=10)
    per = F.cross_entropy(logits.detach(), y, reduction="none")
    SmallLossLoss(10, forget_rate=0.5, ramp_frac=0.0)(logits, y)["loss"].backward()
    rejected = per.argsort()[5:]
    assert (logits.grad[rejected] == 0).all()


def test_rejects_bad_hparams():
    with pytest.raises(ValueError):
        GCELoss(10, q=0.0)
    with pytest.raises(ValueError):
        SCELoss(10, alpha=0.0, beta=0.0)
    with pytest.raises(ValueError):
        SCELoss(10, log_zero=1.0)
    with pytest.raises(ValueError):
        SmallLossLoss(10, forget_rate=1.0)
    with pytest.raises(ValueError):
        SmallLossLoss(10, ramp_frac=1.5)
