# GAM-Softmax — Progress & Results

> **⚠️ SUPERSEDED (2026-07-31).** This was written mid-project, before the final
> results. Its central claim — that a structured margin beats a fixed one — was
> **not** supported. Read [FINAL_REPORT.md](FINAL_REPORT.md) for what actually
> happened, or open `presentation/GAM-Softmax-Presentation.pptx` for the talk version.


*A 10-minute walkthrough. Last updated: 2026-05-29*

---

## 1. The idea in one line

A classifier normally pushes the correct answer toward **100% confidence** for every
example, forever — wasteful. **AS-Softmax** (the paper we build on) instead just asks the
correct class to beat each wrong class by a fixed **gap δ**, then stops pushing. Trains
faster, generalizes better.

**Our method, GAM-Softmax, makes that single fixed gap *smart* instead of one global number**,
varying it along three axes:

| Axis | Idea | Example |
|---|---|---|
| **Class-pair** | bigger gap for hard class pairs | "cat vs dog" hard → big gap; "cat vs plane" easy → small gap |
| **Sample** | bigger gap for ambiguous examples | blurry photo ≠ clear photo |
| **Time** | gap changes during training | start loose, tighten later |

**The bet:** a structured gap beats one fixed number. We test it on text first, then images/audio, then write the paper.

---

## 2. What we built (Stage 0 — DONE)

A clean, reproducible research engine — **one config file = one experiment = one result row**:

- **9 loss functions** (`gam_softmax/losses/`): softmax, AS-Softmax, AM-Softmax, sparsemax, entmax, focal, label-smoothing, power-softmax, and **GAM-Softmax (ours)**.
- **Training loop, SST-5 + 20-Newsgroups data loaders, BERT-base wrapper, experiment runner** — single entrypoint `experiments/run.py --config <yaml>`.
- **M3 "smart gap"** (`gam_softmax/margins/classpair_lowrank.py`) — the class-pair margin, in both fixed and learnable forms.
- **~500-line test suite** — verifies the math of every loss, including a proof that GAM-Softmax reduces *exactly* to AS-Softmax when the gap is held constant. **Tests pass.**

> Constraint: everything runs on a **6 GB RTX 3050 laptop GPU** (the docs assume a 3090/4090).

---

## 3. Results so far

### Stage 1 — Baselines reproduce cleanly ✅
SST-5, BERT-base, 3 seeds:

| method | mean acc | std |
|---|---|---|
| softmax | 51.1% | 0.15% |
| AS-Softmax | **52.2%** | 0.75% |

**Sanity check passes:** AS-Softmax ≥ softmax (+1.06%). Engine is trustworthy.
*(Fixed a 6 GB-GPU attention crash by disabling memory-efficient SDP; cut epochs 5→3 to stop overfitting.)*

### Stage 2 — Class-pair "smart gap" tested on SST-5 — H1 verdict: **no benefit**
Criterion to "pass": GAM beats AS-Softmax by **≥ +0.3%** over 3 seeds.

| method | mean acc | vs AS-Softmax |
|---|---|---|
| softmax | 51.1% | — |
| AS-Softmax | 52.2% | — |
| GAM-M3 (fixed/random gap) | 52.1% | −0.06% |
| GAM-M3′ (learnable gap) | 51.8% | −0.33% |

All four methods sit in a **~51–52% band within one std** — on SST-5, the class-pair axis shows **no measurable benefit**.
- We confirmed the learnable margin *does* train (M3′ ≠ M3).
- **Tuning ruled out under-fitting:** raising the margin learning rate *monotonically hurt* (down to 50.1%). So the flat result is real, not a tuning artifact.
- **Root cause:** SST-5 overfits in 1 epoch — *every* method ties ~52%, so there's no headroom for any method to win.

### Stage 2b — Moved to a harder testbed (20 Newsgroups) — first positive signal 🟡
20 classes, ~71% accuracy with real spread between methods (single seed):

| method | best acc | trajectory |
|---|---|---|
| softmax | 71.5% | peaked epoch 1, then overfit |
| AS-Softmax | 70.6% | peaked epoch 2, then overfit |
| **GAM (ours)** | 71.2% | **still rising at last epoch** |

- **GAM > AS-Softmax by +0.6%** — first faint positive for the core hypothesis.
- **GAM overfits slower** — still improving when the others had peaked and declined.
- ⚠️ Caveat: plain softmax is still best, and AS-Softmax didn't beat plain CE here.

---

## 4. The honest state & key tension

**Engine: built and verified. Baselines: clean. Core hypothesis: not yet proven.**

The central tension we hit: **AS-Softmax — the method we extend and aim to beat — does not reliably beat plain cross-entropy in our setup** (won by ~1% on SST-5, lost by ~0.9% on 20NG). On balanced/clean text, plain CE is a strong baseline; margin/masking methods are usually motivated in *harder* regimes (many classes, long-tailed, noisy labels).

**Most durable finding so far:** GAM **overfits slower** than both baselines (20NG trajectory) — a real, repeatable behavioral difference, even where raw accuracy ties.

---

## 5. Where we go next (paused for direction decision)

| # | Option | Rationale |
|---|---|---|
| a | **Reframe around robustness/calibration** | GAM showed a real overfitting-trajectory advantage even when accuracy tied |
| b | **Move to a regime where margins are known to help** | many-class, long-tailed, or noisy-label data |
| c | **Pivot axis** — sample-based (Stage 3) or time-based (Stage 4) | class-pair on text showed no win |
| d | **Reposition the claim** | match the contribution to where the evidence is |

> No more compute until the direction is chosen — full detail in
> [STATUS_AND_NEXT_STEPS.md](STATUS_AND_NEXT_STEPS.md) and the decision log in [README.md](README.md) §7.

---

## 6. One-slide takeaway

- ✅ **Built** a full, tested 9-loss research engine with a clean one-config-per-experiment workflow.
- ✅ **Reproduced** baselines; AS-Softmax beats plain softmax on SST-5 as expected.
- 🟡 **Tested the core hypothesis** (smart class-pair gap): ties on SST-5 (no headroom), faint +0.6% on 20-Newsgroups, and **overfits slower than baselines** — promising but not yet a win.
- 🔑 **Key insight:** the bottleneck is the *testbed/regime*, not the code — clean balanced text doesn't separate methods. Next move is choosing the right regime or reframing toward robustness.
