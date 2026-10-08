"""Phase C′ margins: `LossMixtureMargin` (M4-v2) and `AbsoluteGapMargin`, plus the
sample-index plumbing they rely on. Both subclass `SampleConfidenceMargin` and change
only the suspicion statistic, so these tests focus on that statistic, the state
discipline, and the GAM dict contract.
"""

from __future__ import annotations

import torch
import pytest

from gam_softmax.losses import ASSoftmaxLoss, GAMSoftmaxLoss
from gam_softmax.margins import AbsoluteGapMargin, LossMixtureMargin, SampleConfidenceMargin
from gam_softmax.schedules import ConstantSchedule

N_CLASSES = 5
DELTA = 0.15


def _mix(n_train=64, **kw) -> LossMixtureMargin:
    kw.setdefault("beta", 1.0)
    kw.setdefault("reject_warmup_frac", 0.0)
    kw.setdefault("reject_ramp_end", 0.3)
    kw.setdefault("delta_floor", -1.0)
    kw.setdefault("delta_ceiling", DELTA)
    kw.setdefault("refit_every", 1)
    kw.setdefault("min_seen_frac", 0.5)
    return LossMixtureMargin(N_CLASSES, ConstantSchedule(DELTA), n_train=n_train, **kw)


def _gap(**kw) -> AbsoluteGapMargin:
    kw.setdefault("beta", 1.0)
    kw.setdefault("reject_warmup_frac", 0.0)
    kw.setdefault("reject_ramp_end", 0.3)
    kw.setdefault("delta_floor", -1.0)
    kw.setdefault("delta_ceiling", DELTA)
    return AbsoluteGapMargin(N_CLASSES, ConstantSchedule(DELTA), **kw)


def _logits_with_target_prob(p_t: torch.Tensor, targets: torch.Tensor, lead_class: int | None = None,
                             p_lead: torch.Tensor | None = None) -> torch.Tensor:
    """Build logits whose softmax gives the target probability ``p_t`` per row, with the
    remaining mass spread evenly over the other classes — or, if ``lead_class`` is set,
    ``p_lead`` on that class and the rest spread over the others."""
    B = p_t.numel()
    probs = torch.zeros(B, N_CLASSES)
    for i in range(B):
        others = [c for c in range(N_CLASSES) if c != int(targets[i])]
        if lead_class is not None and lead_class != int(targets[i]):
            rest = [c for c in others if c != lead_class]
            probs[i, lead_class] = p_lead[i]
            probs[i, rest] = (1.0 - p_t[i] - p_lead[i]) / len(rest)
        else:
            probs[i, others] = (1.0 - p_t[i]) / len(others)
        probs[i, int(targets[i])] = p_t[i]
    return torch.log(probs.clamp_min(1e-9))


# ----------------------------------------------------------------------------- mixture fit

def test_mixture_recovers_planted_bimodal_weights():
    torch.manual_seed(0)
    m = _mix()
    lo = torch.randn(600) * 0.1 + 0.3           # fitted samples: small loss
    hi = torch.randn(400) * 0.3 + 4.0           # mislabeled samples: large loss
    fit = m.fit_mixture(torch.cat([lo, hi]))
    assert fit["k"] == 2
    assert abs(fit["pi_hi"] - 0.4) < 0.05
    assert fit["mu_lo"] < 1.0 < fit["mu_hi"]
    assert fit["bic2"] < fit["bic1"]


def test_mixture_bic_gate_picks_one_component_on_unimodal_losses():
    torch.manual_seed(0)
    m = _mix()
    fit = m.fit_mixture(torch.randn(1000) * 0.2 + 0.5)
    assert fit["k"] == 1
    # the gate can be switched off (ablation)
    m2 = _mix(bic_gate=False)
    assert m2.fit_mixture(torch.randn(1000) * 0.2 + 0.5)["k"] == 2


def test_mixture_degenerate_constant_losses_do_not_crash():
    m = _mix()
    fit = m.fit_mixture(torch.full((100,), 0.7))
    assert fit["k"] == 1


def test_anchor_gate_rejects_a_high_mode_below_chance_loss():
    """Clean data mid-training is bimodal too (fitted vs not-yet-fitted), but its high
    mode sits below the chance-level loss log C. The gate must leave that alone."""
    torch.manual_seed(0)
    n = 400
    m = _mix(n_train=n)                                   # C = 5 → chance loss = log 5 ≈ 1.61
    m.ema_loss.copy_(torch.cat([torch.rand(200) * 0.05 + 0.02,       # fitted: ≈ 0.02–0.07
                                torch.rand(200) * 0.4 + 0.6]))       # unfitted-clean: ≈ 0.6–1.0 < 1.61
    m.seen.fill_(True)
    m._refit()
    assert m.last_fit["k"] == 2                            # bimodal ...
    assert m.last_fit["hi_mean_loss"] < m.chance_loss      # ... but below chance
    assert not bool(m.mix_active) and m.last_est_noise_rate == 0.0
    # same shape of data shifted above chance → active, and the rate is the planted 0.5
    m.ema_loss.copy_(torch.cat([torch.rand(200) * 0.05 + 0.02, torch.rand(200) * 0.4 + 2.5]))
    m._refit()
    assert bool(m.mix_active) and abs(m.last_est_noise_rate - 0.5) < 0.05
    # the gate can be switched off (ablation)
    m2 = _mix(n_train=n, anchor_gate=False)
    m2.ema_loss.copy_(torch.cat([torch.rand(200) * 0.05 + 0.02, torch.rand(200) * 0.4 + 0.6]))
    m2.seen.fill_(True)
    m2._refit()
    assert bool(m2.mix_active)


# ----------------------------------------------------------------------------- state discipline

def test_neutral_before_first_fit_recovers_as_softmax():
    """Until the mixture is fitted, s = 0.5 → δ = δ_base: GAM equals AS-Softmax exactly."""
    torch.manual_seed(0)
    m = _mix(n_train=1000, min_seen_frac=0.9)       # a 16-batch cannot reach 90% seen
    gam = GAMSoftmaxLoss(N_CLASSES, m)
    as_ = ASSoftmaxLoss(N_CLASSES, delta=DELTA, warmup_frac=0.0)
    logits = torch.randn(16, N_CLASSES, requires_grad=True)
    targets = torch.randint(0, N_CLASSES, (16,))
    idx = torch.arange(16)
    out = gam(logits, targets, step_frac=0.5, sample_idx=idx)
    ref = as_(logits, targets, step_frac=1.0)
    assert torch.allclose(out["loss"], ref["loss"], atol=1e-6)
    assert torch.equal(out["mask"], ref["mask"])
    assert m.last_est_noise_rate is None
    assert "est_noise_rate" not in out


def test_eval_call_without_sample_idx_touches_no_state():
    torch.manual_seed(0)
    m = _mix(n_train=32)
    logits = torch.randn(8, N_CLASSES)
    targets = torch.randint(0, N_CLASSES, (8,))
    before = (m.ema_loss.clone(), m.seen.clone(), int(m.n_updates))
    s = m._suspicion(logits, targets, 0.5, sample_idx=None)
    assert s.shape == (8,) and torch.allclose(s, torch.full((8,), 0.5))
    assert torch.equal(m.ema_loss, before[0]) and torch.equal(m.seen, before[1])
    assert int(m.n_updates) == before[2]


def test_ema_update_first_visit_then_momentum():
    m = _mix(n_train=4, ema_momentum=0.5, min_seen_frac=1.0)   # no refit in this test
    targets = torch.tensor([0, 1])
    idx = torch.tensor([2, 3])
    l1 = _logits_with_target_prob(torch.tensor([0.5, 0.25]), targets)
    m._suspicion(l1, targets, 0.1, sample_idx=idx)
    loss1 = -torch.log(torch.tensor([0.5, 0.25]))
    assert torch.allclose(m.ema_loss[idx], loss1, atol=1e-5)
    assert m.seen[idx].all() and not m.seen[[0, 1]].any()
    l2 = _logits_with_target_prob(torch.tensor([0.9, 0.9]), targets)
    m._suspicion(l2, targets, 0.2, sample_idx=idx)
    loss2 = -torch.log(torch.tensor([0.9, 0.9]))
    assert torch.allclose(m.ema_loss[idx], 0.5 * loss1 + 0.5 * loss2, atol=1e-5)


def test_mixture_rejects_high_loss_samples_and_estimates_rate():
    """Plant a 40%-noisy loss history, then check the margin fully rejects the noisy
    half and leaves the clean half at the AS-Softmax margin, and logs ≈ 0.4."""
    torch.manual_seed(0)
    n = 200
    m = _mix(n_train=n, refit_every=1, min_seen_frac=0.5)
    # seed the EMA buffer directly: clean ≈ 0.3, noisy ≈ 4.0
    noisy = torch.zeros(n, dtype=torch.bool); noisy[:80] = True
    m.ema_loss.copy_(torch.where(noisy, torch.randn(n) * 0.3 + 4.0, torch.randn(n) * 0.1 + 0.3))
    m.seen.fill_(True)
    # one training step on 4 noisy + 4 clean samples: the model assigns the label 2% vs 80%
    targets = torch.zeros(8, dtype=torch.long)
    p_t = torch.tensor([0.02] * 4 + [0.8] * 4)
    logits = _logits_with_target_prob(p_t, targets)
    idx = torch.cat([torch.arange(4), torch.arange(100, 104)])
    gam = GAMSoftmaxLoss(N_CLASSES, m)
    out = gam(logits.requires_grad_(), targets, step_frac=0.5, sample_idx=idx)
    assert m.last_est_noise_rate is not None and abs(m.last_est_noise_rate - 0.4) < 0.05
    assert abs(float(out["est_noise_rate"]) - 0.4) < 0.05
    keep = out["mask"]
    # noisy rows: every non-target slot masked (hard rejection) → zero gradient on them
    assert not keep[:4, 1:].any()
    # clean rows: s ≈ 0 → δ clamps to the ceiling (= δ_base): same mask as AS-Softmax
    as_ = ASSoftmaxLoss(N_CLASSES, delta=DELTA, warmup_frac=0.0)
    ref = as_(logits, targets, step_frac=1.0)
    assert torch.equal(keep[4:], ref["mask"][4:])
    out["loss"].backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.allclose(logits.grad[:4], torch.zeros(4, N_CLASSES))


def test_mixture_beta_zero_is_as_softmax_even_with_active_mixture():
    torch.manual_seed(0)
    m = _mix(n_train=100, beta=0.0)
    m.ema_loss.copy_(torch.cat([torch.randn(40) * 0.3 + 4.0, torch.randn(60) * 0.1 + 0.3]))
    m.seen.fill_(True)
    m._refit()
    assert bool(m.mix_active)
    logits = torch.randn(10, N_CLASSES)
    targets = torch.randint(0, N_CLASSES, (10,))
    delta = m(logits, targets, None, 0.9, sample_idx=torch.arange(10))
    assert torch.allclose(delta, torch.full_like(delta, DELTA))


# ----------------------------------------------------------------------------- absolute gap

def test_abs_gap_correctly_classified_sample_keeps_as_margin():
    m = _gap(temp=0.1)
    targets = torch.tensor([0, 1, 2])
    logits = _logits_with_target_prob(torch.tensor([0.9, 0.6, 0.4]), targets)   # target leads everywhere
    delta = m(logits, targets, None, 1.0)
    assert torch.allclose(delta, torch.full_like(delta, DELTA))   # s < 0.5 → δ > base → clamped to ceiling


def test_abs_gap_contradicted_sample_is_fully_rejected():
    m = _gap(temp=0.1)
    targets = torch.tensor([0, 0])
    # row 0: class 3 leads the label by 0.7 → rejected. row 1: the label leads, but only
    # narrowly (p_t − p_j ≤ 0.13 < δ = 0.15), so AS-Softmax keeps its competitors → kept.
    logits = _logits_with_target_prob(torch.tensor([0.1, 0.3]), targets,
                                      lead_class=3, p_lead=torch.tensor([0.8, 0.2]))
    gam = GAMSoftmaxLoss(N_CLASSES, m)
    logits.requires_grad_()
    out = gam(logits, targets, step_frac=1.0)
    keep = out["mask"]
    assert not keep[0, 1:].any()           # every competitor masked → sample contributes nothing
    assert keep[1, 1:].any()
    out["loss"].backward()
    assert torch.allclose(logits.grad[0], torch.zeros(N_CLASSES))
    assert torch.isfinite(logits.grad).all()


def test_abs_gap_temp_sets_the_rejection_threshold():
    """A lead of 0.15 rejects at temp 0.1 but not at temp 0.3 (docstring geometry)."""
    targets = torch.tensor([0])
    logits = _logits_with_target_prob(torch.tensor([0.3]), targets, lead_class=1, p_lead=torch.tensor([0.45]))
    for temp, expect_rejected in ((0.1, True), (0.3, False)):
        out = GAMSoftmaxLoss(N_CLASSES, _gap(temp=temp))(logits, targets, step_frac=1.0)
        assert (not out["mask"][0, 1:].any()) == expect_rejected, temp


def test_abs_gap_beta_zero_recovers_as_softmax_and_rejects_bad_hparams():
    torch.manual_seed(0)
    logits = torch.randn(12, N_CLASSES)
    targets = torch.randint(0, N_CLASSES, (12,))
    out = GAMSoftmaxLoss(N_CLASSES, _gap(beta=0.0))(logits, targets, step_frac=1.0)
    ref = ASSoftmaxLoss(N_CLASSES, delta=DELTA, warmup_frac=0.0)(logits, targets, step_frac=1.0)
    assert torch.allclose(out["loss"], ref["loss"], atol=1e-6)
    with pytest.raises(ValueError):
        _gap(offset=1.5)
    with pytest.raises(ValueError):
        _mix(ema_momentum=1.0)
    with pytest.raises(ValueError):
        _mix(n_train=1)
    with pytest.raises(ValueError):
        _mix(refit_every=0)


# ----------------------------------------------------------------------------- plumbing

def test_wants_sample_idx_flag_propagates_only_for_stateful_margins():
    assert LossMixtureMargin.wants_sample_idx is True
    assert AbsoluteGapMargin.wants_sample_idx is False
    assert SampleConfidenceMargin.wants_sample_idx is False
    assert GAMSoftmaxLoss(N_CLASSES, _mix()).wants_sample_idx is True
    assert GAMSoftmaxLoss(N_CLASSES, _gap()).wants_sample_idx is False
    # a stateless margin never receives the kwarg (its 4-arg forward would reject it)
    torch.manual_seed(0)
    out = GAMSoftmaxLoss(N_CLASSES, _gap())(torch.randn(4, N_CLASSES), torch.zeros(4, dtype=torch.long),
                                            step_frac=0.5, sample_idx=torch.arange(4))
    assert out["loss"].ndim == 0


def test_text_batch_carries_dataset_positions():
    from gam_softmax.data.text.newsgroups20 import _ListTextDataset, _collate

    class _Tok:
        def __call__(self, text, truncation, max_length, padding, return_tensors):
            ids = torch.tensor([[len(text)] + [0] * (max_length - 1)])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    ds = _ListTextDataset(["a", "bb", "ccc", "dddd"], [3, 1, 2, 0], _Tok(), max_seq_len=4)
    batch = _collate([ds[3], ds[1]])
    assert torch.equal(batch.idx, torch.tensor([3, 1]))
    assert torch.equal(batch.labels, torch.tensor([0, 1]))
    assert torch.equal(batch.input_ids[:, 0], torch.tensor([4, 2]))
