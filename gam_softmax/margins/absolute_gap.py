from __future__ import annotations

import torch

from gam_softmax.margins.sample_confidence import SampleConfidenceMargin


class AbsoluteGapMargin(SampleConfidenceMargin):
    """M4 with an *absolute* suspicion statistic (Phase C′ stateless control).

        s_i = σ( (max_{j≠t} p_j − p_{t,i} − offset) / temp )

    A sample is suspicious (s > 0.5) iff some other class currently beats its
    training label by more than ``offset``. Everything downstream is inherited
    from ``SampleConfidenceMargin``: δ_i = clamp(δ_base − β(τ)(2 s_i − 1), floor, ceil).

    Why this fixes noise-rate blindness
    -----------------------------------
    M4's batch-relative z-score flags ~half of every batch whatever the data.
    Here the flagged fraction is the fraction of samples the model currently
    disagrees with: it shrinks on clean data as the model fits, and grows with
    the noise rate, with no estimate of that rate anywhere.

    Rejection geometry (same as M4): with gap g = p_max − p_t, the sample is
    fully rejected iff β(2 s − 1) − δ_base ≥ g. With temp = 0.1 any lead of
    g ≳ 0.03 rejects (aggressive); with temp = 0.3 the lead must be g ≳ 0.3
    (only confidently contradicted labels). ``temp`` is therefore the one
    Phase C′ scale alternative.

    Parameters (beyond the parent's)
    --------------------------------
    offset : float
        Lead another class needs before the sample counts as suspicious.
        0 = "any class ahead of the label".
    """

    def __init__(self, *args, offset: float = 0.0, temp: float = 0.1, **kwargs):
        super().__init__(*args, temp=temp, **kwargs)
        if not -1.0 <= offset <= 1.0:
            raise ValueError(f"offset must be in [-1, 1], got {offset}")
        self.offset = offset

    def rejection_band(self, delta_base: float, beta: float | None = None,
                       n_grid: int = 200_001) -> tuple[float, float] | None:
        """Lead interval [g_lo, g_hi] over which a sample is fully rejected at strength β.

        With gap g = max_{j≠t} p_j − p_t and 2σ(x) − 1 = tanh(x/2), the GAM loss is
        exactly zero with g > 0 iff  β·tanh((g − offset)/(2·temp)) ≥ δ_base + g  and
        g ≤ −delta_floor. The left side is concave in g, so the set is one interval
        (or empty). Leads below g_lo are trained on (too small to distrust); leads
        above g_hi are trained on too, against the leading class only, since the
        margin can fall no lower than δ_base − β. ``beta`` defaults to the full value.
        """
        b = self.beta if beta is None else beta
        g = torch.linspace(0.0, 1.0, n_grid, dtype=torch.float64)
        ok = (b * torch.tanh((g - self.offset) / (2.0 * self.temp)) >= delta_base + g) \
            & (g <= -self.delta_floor)
        if not bool(ok.any()):
            return None
        sel = g[ok]
        return float(sel.min()), float(sel.max())

    @torch.no_grad()
    def _suspicion(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        step_frac: float,
        sample_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)          # (B,)
        others = probs.scatter(1, targets.unsqueeze(1), -1.0)           # drop the target column
        p_max_other = others.max(dim=-1).values                         # (B,)
        gap = p_max_other - p_t - self.offset                           # >0 ⇒ another class leads
        return torch.sigmoid(gap / self.temp)
