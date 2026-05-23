# Risks and Decisions

This is the **failure-recovery doc**. Every research project has things that
will go wrong; the difference between a finished paper and a stalled one is
usually how quickly you recognize the failure and pivot.

Each risk has: a description, an early-warning signal, and a concrete
fallback plan.

---

## R1 — Class-pair margins don't help (H1 fails)

**Description.** Low-rank $\delta_{t,j}$ underperforms or merely matches
scalar AS-Softmax.

**Early warning.** Week 4 H1 quick-check on SST-5 shows ≤ 0.1% gain or a
regression.

**Diagnosis steps:**
1. Check that $u, v$ are actually being learned (track their gradient norms).
2. Check the masked fraction — is the model behaving differently from
   AS-Softmax at all?
3. Plot the learned margin matrix at convergence — does it have any structure,
   or is it ≈ uniform?

**Fallback A.** Try a different parameterization (full-rank with weight
decay, or similarity-init then learn).

**Fallback B.** Pivot the paper's lead from "class-pair" to "sample-dependent
margins improve calibration." This is still a publishable contribution if
H2 holds — the title and intro shift, but the method and infrastructure
are reusable.

**Fallback C.** Reframe as a negative result: "We systematically test
whether class-pair structure helps; on five datasets across three modalities,
the answer is no for $n \le 200$ but yes for $n \ge 1000$." This kind of
paper exists and gets cited. Less impact, but recoverable.

---

## R2 — Sample-dependent margins collapse (H2 fails)

**Description.** Either (a) the learned $\delta(x)$ collapses to a near-constant
value (gradient finds the trivial solution), or (b) it reduces accuracy
without improving calibration.

**Early warning.** Week 8: distribution of $\delta(x)$ has variance < 10% of
mean. Or: ECE doesn't improve.

**Fallback A.** Increase anchor-regularization $\mu$.

**Fallback B.** Try non-learned variants (entropy-based, feature-norm)
which can't collapse.

**Fallback C.** Cut sample-dependence from the main story; treat as an
ablation showing "we tried, here's the limit."

---

## R3 — Theoretical work runs over time

**Description.** T2 and T3 take longer than two weeks.

**Early warning.** End of Week 11, T2 proof has open holes; T3 only proved
for soft-mask relaxation.

**Fallback A.** Submit with T1 + T4 only as main theorems, T2 and T3 sketched
in appendix.

**Fallback B.** Reformulate T2 as identifiability only (skip capacity bound).

**Fallback C.** Reach out to a math/theory colleague for a 1-week
collaboration sprint; offer co-authorship.

**Hard rule:** **do not let theory delay the paper.** A submitted paper
with 2 clean theorems is worth more than an unsubmitted paper with 5
half-proved theorems.

---

## R4 — Cross-modality results don't transfer

**Description.** M3 wins on text but ties or loses on CIFAR-100 / ESC-50.

**Early warning.** Week 12 results show < 0.1% gain or regression on image.

**Fallback A.** Retain image/audio results as honest comparison; reframe
intro to be NLP-first. Mention image/audio in §5.5 with a "modality
dependence" subsection.

**Fallback B.** Investigate why — is it a backbone issue (ViT vs ResNet)?
Is the class structure flat (CIFAR-10) vs hierarchical (CIFAR-100)?
A targeted experiment can turn this into a contribution: "we show GAM-Softmax
helps when class structure is hierarchical, identifies when it doesn't."

**Fallback C.** Submit text-only to ACL/EMNLP and image-only follow-up to
CVPR later. Two papers > one bad one.

---

## R5 — Regression extension doesn't work (H5 fails)

**Description.** Binned-regression + distance margins underperforms direct
regression.

**Early warning.** Week 14, UCI Boston results show MAE worse than MLP+MSE.

**Fallback.** Cut the regression section. It was a stretch goal. The paper
loses one section but loses no quality.

**This is the safest cut.** Don't fight to make regression work; classification
is the main story.

---

## R6 — GPU bottleneck

**Description.** Single GPU + 16-week timeline gets squeezed.

**Early warning.** Week 7, you're ≥ 1 week behind on Phase 2.

**Fallback A.** Cut the long-running datasets: Eurlex, WikiArt, Speech
Commands. Keep the core 4–5 text + CIFAR-100 + ESC-50.

**Fallback B.** Reduce seeds from 5 to 3 (acceptable, but report std bigger).

**Fallback C.** Reduce ablations to the top-5 most important (rank sweep,
schedule, composition, anchor regularization, per-direction).

**Fallback D.** Use cloud credits for a 1-week burst — even free Colab Pro
or Kaggle GPU hours can unstick a logjam.

---

## R7 — AS-Softmax reproduction fails

**Description.** Week 2 ends without hitting the published number.

**Early warning.** AS-Softmax accuracy on SST-5 is more than 1% off paper.

**Diagnosis:**
1. Check $\delta$ value — they used 0.3 for SST-5.
2. Check the warm-up ratio — they used $r = 0.5$ on SST-5.
3. Check the loss is on $p_t - p_j$ (probability space), not $o_t - o_j$
   (logit space).
4. Check AS-Speed isn't broken (try with it off).
5. Check the backbone — BERT-base vs roberta-base produce different numbers.

**Fallback A.** Ask the authors. They responded to email per the paper.

**Fallback B.** Use the official code if released, port piece by piece.

**Fallback C.** If still stuck after a week, use a slightly worse but stable
baseline (e.g., the implementation matches the paper qualitatively but
+/- 1%) and disclose this in the appendix.

**Hard rule:** **do not start GAM-Softmax until AS-Softmax is reproduced.**
Building on an untrustworthy baseline poisons everything.

---

## R8 — Reviewer pushback on novelty

**Description.** Reviewers say "this is just AS-Softmax with three
hyperparameters."

**Pre-emptive mitigation (write this into the paper):**

- The bound (T1) makes it formal that scalar AS-Softmax is the *constant*
  special case of GAM-Softmax. We're a strict generalization, not a tweak.
- The low-rank theory (T2) gives a non-trivial complexity argument.
- The cross-modality results (and the diagnostic figures showing learned
  structure) demonstrate the framework captures real structure, not just
  more knobs.
- The honest negative results in the appendix show we *aren't* claiming
  every axis helps everywhere — which makes the positive claims credible.

If reviewers still push back, the rebuttal points to: (i) the bound is
new and non-trivial; (ii) the empirical gains are statistically significant;
(iii) the framework opens future work (e.g., probabilistic-margin LMs).

---

## R9 — Mentor disagreement on direction

**Description.** Mid-project, mentor pushes toward a different framing or
extra experiments.

**Mitigation.** Schedule weekly 30-min checkpoints. Show the
[06_timeline_and_milestones.md](06_timeline_and_milestones.md) and ask which gates they want to weigh in
on. Disagreements caught early are cheap; disagreements caught at Week 15
are expensive.

---

## R10 — You burn out

**Description.** 16 weeks of focused research is real work.

**Mitigation:**
- Keep weekly hours capped. 25–30 h/week sustainable beats 50 h/week for
  two weeks then nothing.
- Take a buffer Sunday every 3 weeks. The plan accommodates this.
- Celebrate Gate passes. They're hard-won.
- If you hit a wall, work on a different doc (writing, reading) instead of
  more experiments. Writing-heavy weeks save experiment-heavy weeks later.

---

## Decision tree summary

```
After Gate 1 (W3): Baselines reproducible?
├─ Yes → Phase 2
└─ No  → Spend Week 4 fixing; do NOT proceed (see R7)

After Gate 2 (W7): Class-pair helps?
├─ Big win → Phase 3 with class-pair as lead
├─ Marginal → Phase 3, see if sample adds; reframe as "framework" paper
└─ No → Pivot lead to sample/time; class-pair becomes ablation (see R1)

After Gate 3 (W9): Sample improves calibration?
├─ Yes → Phase 4 with sample in headline
├─ Marginal → Sample stays in main table as one of several axes
└─ No → Cut sample from headline; keep as ablation (see R2)

After Gate 4 (W11): Schedules add value?
├─ Yes → M10 is flagship; M3 is "lite version"
└─ No → M3 is flagship; schedule analysis is contribution to community

After Week 13: Cross-modality works?
├─ Yes → Full multi-domain paper
├─ Partial → Highlight where it works, honest about where it doesn't (see R4)
└─ No → Text-only paper; defer modality story (see R4 Fallback C)

After Week 14: Regression works?
├─ Yes → Regression appendix
└─ No → Cut entirely (see R5)
```
