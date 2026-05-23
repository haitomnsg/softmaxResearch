# Theoretical Plan

A conference paper at the ICLR/NeurIPS/ACL bar usually needs at least *some*
theory. We don't need to prove an SoTA generalization bound, but we do need
non-trivial mathematical claims that motivate the method and survive review.

This document lists the theorems / lemmas we'd *like* to prove, with proof
sketches and an honest assessment of difficulty.

---

## T1 — Generalized margin bound (must-have, easy)

**Claim.** For GAM-Softmax with bounded margin $\delta_{t,j} \in [0, \delta_{\max}]$,
the logit-space margin condition becomes

$$
o_t - o_{\min} \ge \log\!\left(1 + n \bar\delta_{\text{eff}}\right),
$$

where $\bar\delta_{\text{eff}}$ is the effective average margin defined below.

**Sketch.** For each non-target $j$, the masking condition gives
$p_t - p_j \ge \delta_{t,j}$, hence

$$
\frac{e^{o_t} - e^{o_j}}{S} \ge \delta_{t,j} \quad\Rightarrow\quad e^{o_j} \le e^{o_t} - \delta_{t,j} S,
$$

where $S = \sum_k e^{o_k}$. Summing over $j \ne t$:

$$
S - e^{o_t} \le (n-1) e^{o_t} - S \sum_{j \ne t} \delta_{t,j}.
$$

Solving and bounding $\min_j e^{o_j} \le S/n$ gives the claim with
$\bar\delta_{\text{eff}} = \frac{1}{n}\sum_{j \ne t} \delta_{t,j}$.

For $\bar\delta_{\text{eff}} \to \delta$ this reduces to Lv et al.'s
$\log(n\delta + 1)$. **Difficulty: easy.** This is a 1–2 page proof and the
core motivation for the method.

---

## T2 — Identifiability / capacity of low-rank margin matrix (must-have, moderate)

**Setup.** Restrict to class-pair margins, no sample / time terms:
$\delta_{t,j} = \sigma(u_t^\top v_j / \sqrt k)$ with $u_t, v_j \in \mathbb{R}^k$.

**Claim (identifiability).** The matrix
$M_{t,j} = \sigma(u_t^\top v_j / \sqrt k)$ is identifiable up to a sign
ambiguity and a $k \times k$ orthogonal transformation $u \mapsto Qu, v \mapsto Qv$.

**Claim (capacity).** The set of matrices expressible by this parameterization
is a strict subset of $[0,1]^{n \times n}$ of effective dimension $\le 2nk - k^2$.
For $k = 8$, $n = 150$ (CLINC150), that's $\le 2336$ parameters expressing a
$22{,}500$-entry matrix — a meaningful compression that admits a
generalization bound (T5).

**Sketch.** Identifiability follows from the inverse function theorem on
$\sigma$ (monotonic) plus standard SVD-uniqueness arguments on the linear
factor inside. Capacity counting is direct.

**Difficulty: moderate.** Standard low-rank arguments, but careful with the
sigmoid nonlinearity. A clean writeup is 3–4 pages.

---

## T3 — Convergence under time-adaptive margin (must-have, moderate-hard)

**Setup.** Continuous-time gradient flow $\dot\theta_t = -\nabla_\theta L(\theta_t; \delta(\tau))$
with $\delta(\tau)$ continuous in $\tau$ and bounded.

**Claim.** Under standard assumptions (Lipschitz gradients, bounded
parameters), the trajectory under any continuous schedule $\delta(\tau)$
converges to a stationary point of the *terminal* loss
$L(\theta; \delta(1))$, with the same rate as constant-margin gradient flow
modulo a factor that depends on $\sup_\tau |\dot\delta(\tau)|$.

**Sketch.** Treat the time-varying margin as a slowly-varying parameter,
apply standard arguments for non-autonomous gradient flow (think
Łojasiewicz inequality + bounded perturbation). The key insight: changes in
$\delta$ are perturbations to the gradient, and if $\delta$ varies slowly
enough relative to gradient magnitude, the trajectory tracks the
quasi-stationary path.

**Difficulty: moderate-hard.** Need to be careful about the non-smooth mask
$z$ — the loss is continuous in $\delta$ but not differentiable. May need
to relax to a "soft mask" version for the proof and argue separately that
the hard mask is close in expectation.

**Fallback if proof is too hard:** prove for the soft-mask relaxation only,
and provide empirical evidence (training curves) that the hard mask behaves
the same way.

---

## T4 — Loss-accuracy correlation (nice-to-have, easy)

**Claim.** Under GAM-Softmax, the per-batch loss is a *monotonic* function
of the per-batch error rate when all class-pair margins satisfy
$\delta_{t,j} \ge \delta_{\min} > 0$.

**Sketch.** When a sample is correctly classified with margin, its loss is 0
(the mask kills all non-target terms). When it's misclassified, its loss is
strictly positive. So zero loss ⇔ all-correct. Then argue the loss is
monotonic in the *number* of incorrect samples.

This is the formal counterpart to Lv et al.'s empirical
$\rho \approx -0.95$ finding. **Difficulty: easy.** 1 page.

**Why it matters for the paper.** Reviewers like the "loss finally tracks
accuracy" framing. Making it a theorem (not just an empirical correlation)
is a small but rhetorically large win.

---

## T5 — Generalization bound for low-rank parameterization (stretch)

**Claim.** With probability $1 - \delta'$ over training samples, the test
error of GAM-Softmax with low-rank class-pair margins satisfies

$$
\text{err}_{\text{test}}(\theta) \le \text{err}_{\text{train}}(\theta) + O\!\left(\sqrt{\frac{n k \log m + \log(1/\delta')}{m}}\right),
$$

where $m$ is the training set size and $k$ is the rank.

**Sketch.** Standard PAC-Bayes / Rademacher complexity argument with the
margin matrix as the structured object. The benefit of low-rank: complexity
scales as $nk$ rather than $n^2$.

**Difficulty: stretch.** This is real generalization theory. If we can pull
it off, it's a strong contribution. If not, it gets cut from the paper —
**don't block on it.**

**Fallback:** cite existing bounds for low-rank softmax variants and argue
inheritance.

---

## T6 — Equivalence with weighted cross-entropy (nice-to-have, easy)

**Claim.** GAM-Softmax with sample-dependent margin $\delta(x)$ is equivalent
to a weighted cross-entropy where the per-sample weight is determined by the
margin function.

**Sketch.** The mask is a 0/1 weight on negative-class log-probs. Mapping
between weights and margins is straightforward.

**Why it matters.** Connects GAM-Softmax to the focal-loss / hard-example
mining literature, which helps in related-work positioning.

---

## What to prove first, what to defer

**Phase 1 of theory work (Weeks 8–10):** T1, T4, T6. These are all
straightforward and together justify the method.

**Phase 2 (Weeks 11–13):** T2, T3. These are the meaty contributions.

**Phase 3 (Weeks 14–15, only if on track):** T5. Cut if needed.

---

## Sanity-checking the theory

Every theorem should be **empirically corroborated** by an experiment
plotted in the paper:

| Theorem | Plot in paper |
|---|---|
| T1 | Histogram of $o_t - o_{\min}$ at convergence vs predicted bound |
| T2 | Heatmap of learned $M$ vs rank-$k$ reconstruction |
| T3 | Loss curves under different $\delta(\tau)$ schedules |
| T4 | Scatter of (per-batch loss, per-batch error) — should match Lv et al.'s SST-5 result |
| T5 | Train/test error gap as function of $n, k, m$ |

If the experiment disagrees with the theorem, something's wrong with one
of them. **Run the experiment first** for cheap theorems (T1, T4) — a
half-day of empirical work can save a week of misdirected proof.
