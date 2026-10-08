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
    the EMA losses of the examples seen so far (a few EM iterations in torch,
    ~10⁴ scalars, negligible cost) and picks one by BIC:

    - one component wins → no bimodality, i.e. no detectable label noise.
      s_i = 0.5 for everyone, so δ = δ_base and the loss is exactly AS-Softmax.
    - two components win → s_i = posterior probability that sample i's EMA loss
      came from the **high**-loss component. The high component's mixing weight
      is exposed as ``last_est_noise_rate``: the method's own estimate of the
      noise rate, a free and checkable prediction (Phase C′ criterion P2).

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
        If False, always use the 2-component fit (ablation of the gate).
    """

    wants_sample_idx = True

    def __init__(
        self,
        *args,
        n_train: int,
        ema_momentum: float = 0.5,
        refit_every: int = 100,
        em_iters: int = 20,
        min_seen_frac: float = 0.5,
        bic_gate: bool = True,
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

        self.register_buffer("ema_loss", torch.zeros(n_train))
        self.register_buffer("seen", torch.zeros(n_train, dtype=torch.bool))
        # [pi_hi, mu_lo, var_lo, mu_hi, var_hi]; valid only when mix_active
        self.register_buffer("mix", torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0]))
        self.register_buffer("mix_active", torch.tensor(False))
        self.register_buffer("n_updates", torch.tensor(0, dtype=torch.long))
        # None until the first fit, then the high component's weight (0.0 if the
        # BIC gate chose one component). GAMSoftmaxLoss logs it as est_noise_rate.
        self.last_est_noise_rate: float | None = None

    # ------------------------------------------------------------------ mixture

    @staticmethod
    def _log_normal(x: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        return -0.5 * (math.log(2 * math.pi) + torch.log(var) + (x - mu) ** 2 / var)

    @torch.no_grad()
    def fit_mixture(self, x: torch.Tensor) -> dict:
        """1- vs 2-component Gaussian mixture on a 1-D sample, chosen by BIC.

        Returns a dict with the chosen ``k``, the 2-component params and both
        BICs. Exposed for tests.
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
        fit = self.fit_mixture(self.ema_loss[self.seen])
        if fit["k"] == 2:
            self.mix.copy_(torch.tensor([fit["pi_hi"], fit["mu_lo"], fit["var_lo"], fit["mu_hi"], fit["var_hi"]],
                                        dtype=self.mix.dtype, device=self.mix.device))
            self.mix_active.fill_(True)
            self.last_est_noise_rate = fit["pi_hi"]
        else:
            self.mix_active.fill_(False)
            self.last_est_noise_rate = 0.0

    @torch.no_grad()
    def posterior_high(self, loss: torch.Tensor) -> torch.Tensor:
        """P(high-loss component | ℓ) under the current mixture; 0.5 if inactive."""
        if not bool(self.mix_active):
            return torch.full_like(loss, 0.5)
        pi_hi, mu_lo, var_lo, mu_hi, var_hi = [self.mix[i] for i in range(5)]
        lp_hi = torch.log(pi_hi.clamp_min(1e-12)) + self._log_normal(loss, mu_hi, var_hi)
        lp_lo = torch.log((1.0 - pi_hi).clamp_min(1e-12)) + self._log_normal(loss, mu_lo, var_lo)
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
