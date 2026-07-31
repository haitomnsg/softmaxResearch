# GAM-Softmax — Final Report

**Project:** Generalized Adaptive Margin Softmax
**Period:** 2026-05-23 → 2026-07-31
**Compute:** one RTX 3050 Laptop, 6 GB VRAM
**Status:** concluded

*This is the single document that answers "what did you do and what did you
find." [README.md](README.md) is the original research plan,
[HOW_WE_VARY_DELTA.md](HOW_WE_VARY_DELTA.md) explains the method in plain
language, and `runs/*/summary.md` hold the raw per-experiment tables.*

---

## 1. The one-paragraph version

We set out to show that AS-Softmax's single scalar margin δ is the wrong
primitive, and that a **margin function** δ_{t,j}(x, τ) — varying by class pair,
by sample, and over time — does better. We built the full apparatus: nine loss
functions, three margin variants, a composable margin/schedule architecture, a
noisy-label testbed, and calibration/memorization instrumentation, all
unit-tested. We ran it on four testbeds.

**On accuracy, the hypothesis did not hold.** No variant beat plain
cross-entropy by more than seed noise on any testbed. The reason is upstream of
our contribution and is the most useful thing this project found: **AS-Softmax
itself — the published method we extend — does not reliably beat plain
cross-entropy in our setup.** A generalization of a scalar cannot help when the
scalar version has no advantage to generalize.

**One thing did work, on a different metric.** Extending δ along the *sample*
axis and allowing it to go **negative** ejects examples the model distrusts from
the loss entirely — which makes the noisy-label literature's "small-loss trick" a
special case of a margin. Under 40% label noise this cut memorization of
corrupted labels roughly **fourfold** (3.1 vs 11.5 pts above the
no-memorization floor) and left the finished model **+2.8 pts** better than
cross-entropy, replicated on both seeds. It did *not* improve peak accuracy. The
benefit is therefore conditional: early-stopping on a clean validation set
discards it entirely — and a clean validation set is precisely what you do not
have when your labels are noisy.

We also found, against the standard story and against our own stated rationale
for the noisy-label pivot, that **AS-Softmax memorizes corrupted labels *more*
than plain cross-entropy.**

What we defend: the negative results, the mechanism analysis explaining them, one
conditional positive finding, and a codebase that makes all of it checkable.

---

## 2. What we were testing

AS-Softmax (Lv et al. 2023) replaces "push p_t → 1" with a margin condition: for
a training example with true class *t*, drop non-target class *j* from the loss
once `p_t − p_j ≥ δ`. It reports better accuracy, better calibration, and ~1.2×
speedup on text classification.

δ is **one scalar** — the same for every class pair, every sample, and every step
of training. Our hypothesis:

> **H (core):** replacing the scalar δ with a structured margin function
> δ_{t,j}(x, τ) yields measurable gains over scalar AS-Softmax.

Broken into the three axes, with the pre-registered success criterion of
**≥ +0.3% accuracy over AS-Softmax, averaged over seeds**:

| | Hypothesis | Variant |
|---|---|---|
| **H1** | class-pair structure helps | M3 — low-rank `σ(u_t·v_j/√k)` |
| **H2** | per-sample structure helps | M4 — batch-relative confidence, δ may go negative |
| **H3** | the axes compose | M6 — M3 × M4 |

There is also a **precondition** that the whole plan silently assumed, and which
turned out to be the crux:

> **H0:** AS-Softmax beats plain cross-entropy in our setup.

---

## 3. What was built

All of it works and is tested — the null result is not a code failure. 78 unit
tests pass in ~20 s.

| Component | Where | Notes |
|---|---|---|
| 9 loss functions | [gam_softmax/losses/](gam_softmax/losses/) | softmax, AS-Softmax, AM-Softmax, sparsemax, entmax-1.5, focal, label-smoothing, power-softmax, GAM-Softmax |
| Margin abstraction | [gam_softmax/margins/](gam_softmax/margins/) | M3 class-pair, M4 sample, composable into M6 |
| Schedules | [gam_softmax/schedules/](gam_softmax/schedules/) | constant, linear-warmup; the time axis |
| Straight-through estimator | [losses/gam_softmax.py](gam_softmax/losses/gam_softmax.py) | forward identical to the hard mask, backward routes gradient into δ's parameters |
| Label-noise testbed | [data/label_noise.py](gam_softmax/data/label_noise.py) | seeded symmetric noise, decoupled from the run seed so every method sees identical corruption |
| Memorization probe | [data/probe.py](gam_softmax/data/probe.py), [training/trainer.py](gam_softmax/training/trainer.py) | measures whether a model predicts the *corrupted* label it was trained on |
| Calibration | [eval/classification.py](gam_softmax/eval/classification.py) | 15-bin ECE |
| Experiment runner | [experiments/run.py](experiments/run.py) | one config = one experiment = one table row |
| Final driver | [experiments/final_experiment.py](experiments/final_experiment.py) | resumable, rebuilds its summary after each run |

Two correctness properties are enforced by tests, because getting them wrong
would have invalidated everything: the mask is computed in **probability** space
(`p_t − p_i`, not logits), and it is **detached**. A test asserts GAM-Softmax
reduces *exactly* to AS-Softmax when the margin is constant — so every GAM result
is a strict generalization of the baseline, not a different loss.

### The one genuinely novel piece

M4 lets **δ go negative**. Nothing in the math requires a margin to be positive:
`p_t − p_j ∈ [−1, 1]`, so δ = −0.4 is a meaningful threshold meaning *"drop class
j even if it currently beats the training label by up to 0.4."*

For a **mislabeled** example the model keeps assigning low `p_t`, so under a
positive δ no competitor is ever masked and the model is pushed to fit the wrong
label forever — that is how networks memorize noise. Under a negative δ every
competitor is masked, the renormalized loss over the single surviving class is
≈0, and the sample drops out of training.

This makes the noisy-label literature's **small-loss trick a special case of a
margin** — no sample-selection heuristic, no auxiliary loss, no second network.
It is the contribution that would have been worth writing up had the empirical
result gone the other way, and it remains, in our view, the correct reframing
even though it did not win.

---

## 4. Every experiment we ran, in order

| # | Date | Testbed | Question | Result |
|---|---|---|---|---|
| 1 | 05-28 | SST-5, clean, 3 seeds | baselines sane? | softmax **51.1%**, AS-Softmax **52.2%** → AS ≥ CE by +1.1% ✓ |
| 2 | 05-28 | SST-5, clean, 3 seeds | **H1** (fixed-random margin) | M3 **52.10%** vs AS **52.17%** = **−0.06%** → FAIL (uninformative: u/v never trained) |
| 3 | 05-29 | SST-5, clean, 3 seeds | **H1** (learnable margin) | M3′ **51.83%** vs AS **52.17%** = **−0.33%** → FAIL |
| 4 | 05-29 | SST-5, tuning screen | is H1 just under-tuned? | higher margin LR *monotonically hurt* (52.5% → 50.6% → 50.1%) → not a tuning artifact |
| 5 | 05-29 | 20NG, clean, 1 seed | testbed with headroom? | GAM **71.18%** > AS **70.59%** (+0.6%) — but plain CE **71.46%** wins, **AS < CE** |
| 6 | 06-13 | SST-5, noise {0,.2,.4}, 2 seeds | does a hard regime rescue it? | all gaps within ±2% seed scatter → INCONCLUSIVE |
| 7 | 07-28/31 | **20NG, noise 40%, 2 seeds** | **the decisive test** | H0/H1 null confirmed; M4 cuts memorization ~4× — see §5 |

Two things repeat across rows 1–6 and are worth stating plainly:

1. **H0 kept failing.** AS-Softmax beat CE on SST-5 (+1.1%) but lost to it on
   20NG (−0.9%) and was a wash under SST-5 noise. Three of four setups gave the
   method we extend no reliable edge over the simplest possible baseline.
2. **The only repeatable GAM signal was stability, not accuracy.** GAM had the
   lowest seed-to-seed variance at every noisy level (±0.3% vs ±1–1.6%) and was
   still *improving* at the final epoch on 20NG while CE and AS-Softmax had
   peaked and started overfitting. That is a robustness signal, and it is what
   experiment 7 was designed to confirm or kill.

---

## 5. E1 — the final experiment

**Design.** 20 Newsgroups (20 classes, headers/footers/quotes stripped — without
that BERT hits ~99% from metadata and nothing separates), BERT-base, 4 epochs,
batch 16, lr 2e-5. **40% symmetric label noise on the training set only;
validation labels stay clean.** The noise seed is fixed per level and decoupled
from the run seed, so every method sees the *identical* corrupted labels — the
comparison is exact, not statistical.

Six configs, all sharing an identical dataset/model/training block so the only
difference is the `loss:` block ([configs/final/](configs/final/)):

| row | δ varies by | notes |
|---|---|---|
| `softmax` | — | plain cross-entropy |
| `as_softmax` | nothing (δ = 0.15) | the published method we extend |
| `gam_m3` | class-pair, time | learnable low-rank margin |
| `gam_m4` | **sample**, time | β = 1.0, δ may fall to −0.85 |
| `gam_m4_b05` | sample, time | β = 0.5 — does the effect scale with the knob? |
| `gam_m6` | class-pair, sample, time | all three axes at once |

**Metrics.** Accuracy alone is what stayed flat in every prior experiment, so E1
measures the mechanism directly:

- **best / final val accuracy** and their gap — the overfitting trajectory
- **val ECE** — calibration; CE keeps pushing p_t → 1 even on corrupted labels
- **mem rate** — of the training examples whose labels were *corrupted*, the
  fraction where the model now predicts **the corrupted label**. This is
  memorization, measured directly rather than inferred.
- **recover rate** — same subset, fraction predicting the **true** label despite
  having been trained on a wrong one.

### 5.1 Results

20 Newsgroups, 40% symmetric label noise, BERT-base, 4 epochs, **2 seeds (42, 43)**.
Run 2026-07-28/31, commit `b2ff750`. Raw records: `runs/final_20ng/*.json`.

| method | peak val acc % | final val acc % | decay | memorized noise (pts over floor) | mem @final % | recovers true label % |
|---|---|---|---|---|---|---|
| softmax (CE) | 67.59 ±0.71 | 64.19 ±0.92 | −3.40 | +11.5 | 33.7 ±0.4 | 41.2 |
| as_softmax | 67.56 ±0.34 | 64.72 ±0.20 | −2.84 | **+13.6** | **39.9 ±1.8** | 44.1 |
| gam_m3 (class-pair) | 67.68 ±0.81 | 65.21 ±1.42 | −2.47 | +11.4 | 35.2 ±1.7 | 45.9 |
| **gam_m4 (sample)** | **68.09 ±0.05** | **66.96 ±0.42** | **−1.13** | **+3.1** | **19.3 ±0.7** | **57.3** |

*"Memorized noise (pts over floor)": with 40% noise where flips never return the
original label, a model that learned the true function and memorized nothing
scores exactly **60.0%** on the labels it was given. Everything above 60 is
memorized noise. "decay" = accuracy lost from the peak epoch to the last.*

**Head-to-head vs plain cross-entropy, reported per seed** — because a
two-seed mean can hide the fact that one seed did all the work:

| | peak (s42 / s43) | final (s42 / s43) | memorization (s42 / s43) |
|---|---|---|---|
| as_softmax − CE | −0.29 / +0.23 | +1.33 / −0.26 | **+7.8 / +4.6** |
| gam_m3 − CE | +0.16 / +0.01 | +0.67 / +1.38 | +0.6 / +2.4 |
| gam_m4 − CE | −0.03 / **+1.04** | **+3.72 / +1.83** | **−14.6 / −14.2** |

![memorization](runs/final_20ng/figures/fig2_memorization_n0.4.png)

### 5.2 What the numbers say

**First, the noise floor — this governs how everything else should be read.**
Plain cross-entropy at two seeds spans **1.0 pt** of peak accuracy (68.09 /
67.09) but only **0.6 pt** of final memorization rate (33.4 / 34.0).
**Memorization is a far quieter measurement than accuracy.** So a 14-point gap on
memorization is ~20× its seed spread and cannot be a lucky seed, while a
half-point gap on accuracy is inside the noise and means nothing. Reading the
right metric is most of this experiment.

**Second, AS-Softmax fails its precondition again.** AS − CE = **−0.03 pts** over
two seeds (−0.29 and +0.23 — a coin flip). That is the fourth testbed in a row;
clean SST-5 was the only one where it won, by +1.06. The method we set out to
generalize does not, in our hands, have an advantage over plain cross-entropy to
generalize.

**Third — and this contradicts the reason we pivoted to label noise at all —
AS-Softmax memorizes corrupted labels *more* than plain cross-entropy.** It ends
**+13.6 pts** above the no-memorization floor against CE's +11.5, and the effect
holds on **both** seeds (+7.8 and +4.6 pts of extra memorization). The argument
in every margin-loss paper, and in our own pivot rationale, is that masking
should *prevent* memorization: "it stops pushing once the margin is met, so it
cannot fit the wrong label." Measured directly, the opposite happens.

The plausible mechanism: masking retires the *easy* negatives first, and those
are precisely what a correctly-labeled example finishes with early. So as
training proceeds an ever-larger share of the surviving gradient budget is spent
on the hard, still-confusable examples — and mislabeled examples are exactly
those. **Scalar masking concentrates effort onto the noise.** This also explains
why the whole noisy-label pivot never produced the expected win.

**Fourth, the sample axis does what it was designed to do.** M4 is the only
variant that breaks the pattern, and it breaks it on every mechanism metric at
once (2-seed means):

| | CE | AS | M3 | **M4** |
|---|---|---|---|---|
| memorized noise above the floor (pts) | 11.5 | 13.6 | 11.4 | **3.1** |
| corrupted examples predicted as their *true* label | 41.2% | 44.1% | 45.9% | **57.3%** |
| accuracy lost from peak to final epoch | 3.40 | 2.84 | 2.47 | **1.13** |
| non-target slots masked at the end | 0% | 56% | 49% | **95%** |

Four independent measurements moving together in the predicted direction, driven
by a mechanism we can point at in the code (δ goes negative → the suspect
sample's competitors are all masked → its loss contribution vanishes). M4 ends
training **3.1 pts** above the theoretical no-memorization floor where the others
sit 11–14 pts above it — it has very nearly stopped memorizing altogether.

**Fifth, the honest limit: this does not translate into a peak-accuracy win.**
M4's mean peak is 68.09 vs CE's 67.59 — nominally **+0.51**, which would clear
the pre-registered +0.3 criterion. **We do not claim it.** Per seed the gap is
**−0.03 and +1.04**: the entire mean comes from seed 43, where cross-entropy
happened to have a bad run. One seed of two is not a result, and this project has
been burned by exactly that pattern before.

What *does* replicate is the benefit at the **end** of training: **+3.72 and
+1.83 pts** over CE, both seeds positive, mean **+2.77**. M4 holds 66.96% where
CE has decayed to 64.19%.

One suggestive extra, offered as an observation rather than a finding: **M4 has
by far the tightest seed-to-seed spread on peak accuracy (±0.05 vs CE's ±0.71)**,
which echoes the "GAM is more stable" thread that showed up in every earlier
experiment. With n=2 a variance claim is not something we can support.

Whether that counts as a win depends on a protocol choice that the earlier
experiments never made explicit:

- **If you have a clean validation set and early-stop on it**, you keep the
  epoch-1 checkpoint and M4 buys you nothing. This is what every experiment in
  this project did, and it is why six weeks of accuracy-only measurements looked
  flat.
- **If you do not** — which is the actual situation whenever labels are noisy,
  since a clean held-out set is exactly what you lack — you keep the final model,
  and M4 is worth **+2.8 pts** over cross-entropy.

**Sixth, a calibration caveat that reads backwards if you only look at the
number.** The margin variants have far worse ECE (0.42–0.47 vs CE's 0.19). This
is **not** overconfidence — it is the opposite. Mean predicted confidence is
0.25 for M4 against ~0.67 accuracy: the margin objective stops pushing p_t
upward by construction, so probabilities stay compressed and the model is
systematically *under*confident. That is a benign failure mode compared to CE's
direction, and it is largely removable by temperature scaling, which we did not
run. **We do not claim a calibration result in either direction.**

---

## 6. Conclusions

### Verdict on each hypothesis

| | Hypothesis | Verdict |
|---|---|---|
| **H0** | AS-Softmax beats plain cross-entropy | **NOT SUPPORTED** — 3 of 4 testbeds show no advantage; here −0.03 pts over 2 seeds |
| **H1** | class-pair structure (M3) helps | **NOT SUPPORTED** — +0.12 pts over AS-Softmax, inside a 1.0 pt noise band; flat on three prior testbeds too, and shown not to be a tuning artifact |
| **H2** | per-sample structure (M4) helps | **SPLIT** — peak accuracy not established (+0.51 mean, but −0.03/+1.04 per seed); memorization cut ~4× and final-epoch accuracy +2.77 pts over CE, both replicated |
| **H3** | the axes compose (M6) | **UNTESTED** — implemented and unit-tested, run cut for compute |

### Confidence, per finding

Graded explicitly, because the findings are not equally solid:

| finding | confidence | basis |
|---|---|---|
| M4 cuts memorization ~4× (−14.4 pts) | **Confirmed** | both seeds, −14.6/−14.2; effect is ~20× the metric's seed spread |
| AS-Softmax memorizes *more* than CE | **Confirmed** | both seeds, +7.8/+4.6 |
| M4 gains at the final epoch (+2.77) | **Confirmed (direction)** | both seeds positive, but magnitude varies 2× (+3.72/+1.83) |
| H0 and H1 null | **Confirmed** | 4 testbeds, 2–3 seeds each, plus a tuning screen that ruled out under-tuning |
| M4 gains at peak accuracy (+0.51) | **Not established** | driven entirely by one seed |
| M4 is more seed-stable (±0.05) | **Suggestive only** | n=2; consistent with every prior experiment but not testable at this sample size |
| Calibration | **No claim** | ECE differences are underconfidence artifacts; temperature scaling not run |

### The one-sentence conclusion

*Generalizing AS-Softmax's scalar margin along the class-pair axis produced no
measurable benefit on any of four testbeds; generalizing it along the sample
axis — by letting the margin go negative, which makes small-loss sample
rejection a special case of a margin — did not reliably improve peak accuracy
either, but cut memorization of corrupted labels roughly fourfold and left the
final-epoch model 2.8 points better than cross-entropy under 40% label noise.*

### What we would and would not claim in writing

**Would claim:**
- A negative result on the class-pair axis, replicated across four testbeds and
  shown not to be a tuning artifact (a dedicated margin LR made it monotonically
  worse).
- The observation that AS-Softmax **increases** memorization under label noise,
  which contradicts the standard rationale for margin losses in that regime, with
  a direct measurement rather than an inference from accuracy.
- The negative-margin reframing (small-loss rejection as a margin) as a modeling
  contribution, with evidence that it works mechanically.

**Would not claim:**
- That GAM-Softmax improves accuracy. It does not.
- Any state-of-the-art claim on noisy-label learning. We never compared against
  the actual baselines in that literature (co-teaching, DivideMix, GCE, symmetric
  CE). M4 beats *cross-entropy and AS-Softmax at the final epoch*; that is a much
  weaker statement and it is the only one the data supports.
- A calibration result, for the underconfidence reason in §5.2.

### Limitations — what would have to be true for this to be wrong

Stated plainly, because the result is modest enough that these matter:

1. **Two seeds, one dataset, one noise level, one architecture, four epochs.** The
   memorization effect is large relative to seed scatter (~20×), but 20 Newsgroups
   at 40% symmetric noise with BERT-base is a single point in a large space. Two
   seeds is enough to kill a hypothesis and not enough to establish a small
   effect — which is exactly why the peak-accuracy gap is reported as
   unestablished while the memorization gap is not.
2. **β was never swept.** The ablation comparing β = 1.0 against β = 0.5 was
   implemented and queued but cut for compute. Without it we cannot show the
   effect scales with the knob, which is the cleanest evidence that a mechanism
   is what we think it is.
3. **Symmetric noise is the easy case.** Real label noise is
   instance-dependent and asymmetric, where "the model disagrees with this label"
   is a much weaker signal of corruption.
4. **The comparison is against weak baselines** — CE and AS-Softmax, not against
   methods designed for noisy labels.
5. **More epochs would likely widen the gap**, since memorization compounds; we
   stopped at 4 and every method was still moving.

---

## 7. Why we stopped here

The project had a decision gate at each phase, and E1 was the last one that
mattered. The honest reasons to stop:

1. **The precondition failed, not just the hypothesis.** Every downstream plan
   (multi-label, binned regression, cross-modality validation on image and
   audio) was premised on margin-based masking being a better objective than
   cross-entropy in the first place. Four testbeds say it isn't — at least not
   for balanced text classification with a strong pretrained encoder, which is
   where all our compute could reach.
2. **The remaining plan was not affordable.** The un-run items (M5, M7–M10,
   image/audio, the theory chapter) assumed an RTX 3090/4090; on a 6 GB laptop
   GPU each 20NG run is ~25 min, so a single properly-seeded sweep is a day of
   wall clock. Extending to new modalities was never within reach.
3. **A clean negative result is worth more than a forced positive one.** With
   enough hyperparameter search on enough testbeds, some configuration would
   eventually clear +0.3% on some seed. That number would not replicate, and we
   would know it. The failure mode this project most needed to avoid was
   reporting one.

---

## 8. What a follow-up should do differently

Not "more of the same." The specific things this project learned that would
change the next attempt:

- **Validate H0 before building anything.** One week reproducing AS-Softmax's
  reported gain over CE on the authors' own setup would have caught the crux in
  week 1 instead of week 8. If the baseline's advantage doesn't reproduce, either
  the setup differs in a way that matters (find it) or the premise is weaker than
  published — and either answer redirects the whole project.
- **Pick a regime where masking is forced to matter.** Balanced text
  classification with a pretrained BERT is close to the worst case: the encoder
  is already good, classes are few, and plain CE is very hard to beat. Extreme
  classification (10k+ classes, where the "skip easy negatives" argument is a
  compute argument, not an accuracy one) or genuinely long-tailed data would give
  the mechanism something to bite on.
- **Instrument the mechanism from day one.** We spent six weeks reading only
  accuracy, which was flat everywhere and told us nothing about *why*. The
  masking rate, calibration, and memorization probes added for E1 should have
  been in the trainer from the first run.
- **Keep the M4 negative-margin framing.** Independently of this project's
  outcome, "small-loss sample rejection is a margin with δ < 0" is a clean idea
  that unifies two literatures, and it is one config line away from being tested
  against real noisy-label baselines (co-teaching, DivideMix) on a proper
  noisy-label benchmark.

---

## 9. Reproducing everything

```bash
conda activate gam        # no `python` on PATH otherwise

python -m pytest tests/ -v                                    # 78 tests, ~20 s
python experiments/final_experiment.py                        # E1, resumable (~6 h)
python experiments/final_experiment.py --only-block 1          # just the decisive block
python experiments/run.py --config configs/final/20ng_gam_m4.yaml --label-noise 0.4
```

Raw artifacts: `runs/final_20ng/` (per-run JSON with full epoch history, plus
`summary.md`), and `runs/*/summary.md` for experiments 1–6.
