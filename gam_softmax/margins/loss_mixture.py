from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from gam_softmax.margins.sample_confidence import SampleConfidenceMargin


class LossMixtureMargin(SampleConfidenceMargin):
    """M4-v2 (Phase C′): suspicion from a per-sample loss history and a mixture model.

    Keeps an exponential moving average of each training example's loss
    ℓ_i = −log p_{t,i} across the steps it is seen. Every ``refit_every``
    training steps it fits a 1-component and a 2-component Gaussian mixture to
    **log** EMA losses of the examples seen so far (a few EM iterations in
    torch, ~10⁴ scalars, negligible cost). The mixture is *active* only if

    1. BIC prefers two components (there is bimodality), and
    2. the high component's mean loss is at least ``log(n_classes)``, the loss
       of a uniform prediction: the model rates those labels **below chance**.

    Otherwise s_i = 0.5 for everyone, δ = δ_base, and the loss is exactly
    AS-Softmax (no noise detected, nothing rejected). When active, s_i is the
    posterior probability that sample i's EMA loss came from the high
    component, and the high component's mixing weight is exposed as
    ``last_est_noise_rate``: the method's own estimate of the noise rate, a
    free and checkable prediction (Phase C′ criterion P2).

    Why the log and the anchor (design-time diagnostic, 2026-10-08, one CE
    epoch on 20NG; see docs/10 §3 Phase C′)
    ---------------------------------------------------------------------------
    Raw CE losses are half-bounded and skewed; a Gaussian fit over-weights the
    tail (estimated rate 0.59 at a true 0.40, precision 0.67). On log losses
    the estimate was 0.39–0.40 with precision 0.82–0.87 and AUC 0.94–0.96.
    But BIC alone is not a noise detector: on **clean** data, samples the model
    has not fitted yet form a second mode too (BIC picked two components with
    weight 0.44–0.52 at zero noise), and a rejected sample is never fitted, so
    that mistake would lock in. The two modes differ in *where* they sit:
    mislabeled samples' loss is ≈ 3.5 ≥ log 20 = 3.0 (the model believes the
    label less than a uniform guess), while the unfitted-clean mode was at 1.5
    then 0.77 and falling. The anchor is the chance-level loss, so it is not a
    tuned threshold.

    This is DivideMix's sample-selection statistic (per-sample loss → 2-GMM
    posterior), expressed as a per-sample margin instead of a separate
    selection stage. Everything downstream (ramp, clamps, the δ formula) is
    inherited from ``SampleConfidenceMargin``, so the only thing that differs
    from M4 is the suspicion statistic.

    State discipline
    ----------------
    ``sample_idx`` (training-set positions of the batch) is passed only on
    training steps. With ``sample_idx=None`` (evaluation) the margin is
    read-only: it scores the batch's *instantaneous* losses under the current
    mixture and updates nothing.

    Parameters (beyond the parent's)
    --------------------------------
    n_train : int
        Size of the training set; sizes the per-sample buffers.
    ema_momentum : float
        Weight on the old EMA value, in [0, 1). 0 keeps only the latest loss.
    refit_every : int
        Training steps between mixture refits.
    em_iters : int
        EM iterations per refit.
    min_seen_frac : float
        Fraction of the training set that must have been seen at least once
        before the first fit. Until then s_i = 0.5 (neutral, AS-Softmax).
    bic_gate : bool
        If False, skip the BIC test and always take the 2-component fit.
    anchor_gate : bool
        If False, skip the chance-level test on the high component's mean.
        (Both gates off = plain DivideMix-style posterior; ablations only.)
    """

    wants_sample_idx = True
    LOG_EPS = 1e-3   # log(ℓ + LOG_EPS): keeps a perfectly fitted sample finite

    def __init__(
        self,
        *args,
        n_train: int,
        ema_momentum: float = 0.5,
        refit_every: int = 100,
        em_iters: int = 20,
        min_seen_frac: float = 0.5,
        bic_gate: bool = True,
        anchor_gate: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if n_train < 2:
            raise ValueError(f"n_train must be >= 2, got {n_train}")
        if not 0.0 <= ema_momentum < 1.0:
            raise ValueError(f"ema_momentum must be in [0, 1), got {ema_momentum}")
        if refit_every < 1:
            raise ValueError(f"refit_every must be >= 1, got {refit_every}")
        if em_iters < 1:
            raise ValueError(f"em_iters must be >= 1, got {em_iters}")
        if not 0.0 < min_seen_frac <= 1.0:
            raise ValueError(f"min_seen_frac must be in (0, 1], got {min_seen_frac}")
        self.n_train = n_train
        self.ema_momentum = ema_momentum
        self.refit_every = refit_every
        self.em_iters = em_iters
        self.min_seen_frac = min_seen_frac
        self.bic_gate = bic_gate
        self.anchor_gate = anchor_gate

        self.register_buffer("ema_loss", torch.zeros(n_train))
        self.register_buffer("seen", torch.zeros(n_train, dtype=torch.bool))
        # [pi_hi, mu_lo, var_lo, mu_hi, var_hi]; valid only when mix_active
        self.register_buffer("mix", torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0]))
        self.register_buffer("mix_active", torch.tensor(False))
        self.register_buffer("n_updates", torch.tensor(0, dtype=torch.long))
        # None until the first fit, then the high component's weight (0.0 if a
        # gate deactivated the mixture). GAMSoftmaxLoss logs it as est_noise_rate.
        self.last_est_noise_rate: float | None = None
        self.last_fit: dict | None = None      # full record of the last refit, for diagnostics/tests

    # ------------------------------------------------------------------ mixture

    @staticmethod
    def _log_normal(x: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        return -0.5 * (math.log(2 * math.pi) + torch.log(var) + (x - mu) ** 2 / var)

    @classmethod
    def _transform(cls, loss: torch.Tensor) -> torch.Tensor:
        """Loss → the space the mixture is fitted in (log)."""
        return torch.log(loss.clamp_min(0.0) + cls.LOG_EPS)

    @property
    def chance_loss(self) -> float:
        """−log(1/C): the loss of a uniform prediction, the anchor for the gate."""
        return math.log(self.n_classes)

    @torch.no_grad()
    def fit_mixture(self, x: torch.Tensor) -> dict:
        """1- vs 2-component Gaussian mixture on a 1-D sample (already in the
        fitted space, i.e. log losses), with ``k`` chosen by BIC.

        Returns a dict with ``k``, the 2-component params and both BICs. The
        chance-level anchor is applied by ``_refit``, not here. Exposed for tests.
        """
        x = x.double()
        N = x.numel()
        # --- one component
        mu1 = x.mean()
        var1 = x.var(unbiased=False).clamp_min(1e-6)
        ll1 = self._log_normal(x, mu1, var1).sum()
        bic1 = -2.0 * ll1 + 2.0 * math.log(N)
        # --- two components, initialised by a median split so the ordering lo/hi is fixed
        med = x.median()
        lo, hi = x[x <= med], x[x > med]
        if hi.numel() == 0:                     # degenerate (all equal): no bimodality
            return {"k": 1, "pi_hi": 0.0, "mu_lo": float(mu1), "var_lo": float(var1),
                    "mu_hi": float(mu1), "var_hi": float(var1), "bic1": float(bic1), "bic2": float("inf")}
        mu = torch.stack([lo.mean(), hi.mean()])
        var = torch.stack([lo.var(unbiased=False), hi.var(unbiased=False)]).clamp_min(1e-6)
        pi = torch.tensor([lo.numel() / N, hi.numel() / N], dtype=x.dtype, device=x.device)
        for _ in range(self.em_iters):
            logp = torch.log(pi.clamp_min(1e-12)) + self._log_normal(x.unsqueeze(1), mu, var)  # (N, 2)
            r = torch.softmax(logp, dim=1)                                                       # responsibilities
            nk = r.sum(0).clamp_min(1e-9)
            pi = nk / N
            mu = (r * x.unsqueeze(1)).sum(0) / nk
            var = ((r * (x.unsqueeze(1) - mu) ** 2).sum(0) / nk).clamp_min(1e-6)
        logp = torch.log(pi.clamp_min(1e-12)) + self._log_normal(x.unsqueeze(1), mu, var)
        ll2 = torch.logsumexp(logp, dim=1).sum()
        bic2 = -2.0 * ll2 + 5.0 * math.log(N)
        # EM can swap the components; "hi" is the one with the larger mean
        hi_k = int(mu.argmax())
        lo_k = 1 - hi_k
        k = 2 if (not self.bic_gate or bic2 < bic1) else 1
        return {"k": k, "pi_hi": float(pi[hi_k]), "mu_lo": float(mu[lo_k]), "var_lo": float(var[lo_k]),
                "mu_hi": float(mu[hi_k]), "var_hi": float(var[hi_k]), "bic1": float(bic1), "bic2": float(bic2)}

    @torch.no_grad()
    def _refit(self) -> None:
        fit = self.fit_mixture(self._transform(self.ema_loss[self.seen]))
        # high component's mean, back in loss units
        hi_mean_loss = math.exp(fit["mu_hi"]) - self.LOG_EPS
        active = fit["k"] == 2 and (not self.anchor_gate or hi_mean_loss >= self.chance_loss)
        self.last_fit = {**fit, "hi_mean_loss": hi_mean_loss, "active": active}
        if active:
            self.mix.copy_(torch.tensor([fit["pi_hi"], fit["mu_lo"], fit["var_lo"], fit["mu_hi"], fit["var_hi"]],
                                        dtype=self.mix.dtype, device=self.mix.device))
            self.mix_active.fill_(True)
            self.last_est_noise_rate = fit["pi_hi"]
        else:
            self.mix_active.fill_(False)
            self.last_est_noise_rate = 0.0

    @torch.no_grad()
    def posterior_high(self, loss: torch.Tensor) -> torch.Tensor:
        """P(high-loss component | ℓ) under the current mixture, for raw losses ℓ;
        0.5 everywhere if the mixture is inactive."""
        if not bool(self.mix_active):
            return torch.full_like(loss, 0.5)
        x = self._transform(loss)
        pi_hi, mu_lo, var_lo, mu_hi, var_hi = [self.mix[i] for i in range(5)]
        lp_hi = torch.log(pi_hi.clamp_min(1e-12)) + self._log_normal(x, mu_hi, var_hi)
        lp_lo = torch.log((1.0 - pi_hi).clamp_min(1e-12)) + self._log_normal(x, mu_lo, var_lo)
        return torch.sigmoid(lp_hi - lp_lo)

    # ------------------------------------------------------------------ margin API

    @torch.no_grad()
    def _suspicion(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        step_frac: float,
        sample_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = F.cross_entropy(logits.float(), targets, reduction="none")      # (B,) = −log p_t
        if sample_idx is None:
            # evaluation / read-only call: score instantaneous losses, touch no state
            return self.posterior_high(loss)

        idx = sample_idx.to(self.ema_loss.device)
        old = self.ema_loss[idx]
        first = ~self.seen[idx]
        new = torch.where(first, loss, self.ema_momentum * old + (1.0 - self.ema_momentum) * loss)
        self.ema_loss[idx] = new
        self.seen[idx] = True
        self.n_updates += 1

        if int(self.n_updates) % self.refit_every == 0 and \
                float(self.seen.float().mean()) >= self.min_seen_frac:
            self._refit()
        return self.posterior_high(new)
