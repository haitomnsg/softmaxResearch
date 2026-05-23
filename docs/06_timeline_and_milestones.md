# Timeline and Milestones

16-week plan, starting **2026-05-23**, targeting a paper draft by
**2026-09-12** (a comfortable margin before NeurIPS / ICLR cycles depending
on which year's deadline). Adjust the start date everywhere if you're
reading this later.

This timeline has **four decision gates** at the end of weeks 3, 7, 11, 13.
At each gate, look at the actual results vs the hypothesis and either
proceed, pivot, or cut scope.

---

## The big picture

```
Weeks: 1 2 3 | 4 5 6 7 | 8 9 | 10 11 | 12 13 | 14 | 15 16
       ─────   ───────   ───   ─────   ─────   ──   ─────
       Setup   Class     Smp   Time    X-mod   Reg  Theory
       +Repro            δ(x)  δ(τ)    +Comb        +Write
                ⬇         ⬇    ⬇        ⬇       ⬇
              Gate1     Gate2 Gate3  Gate4   (cut?)
```

---

## Week 1 — Setup (2026-05-23 → 2026-05-29)

**Goals:** Skeleton, baselines compile, first end-to-end softmax run.

- [ ] Create `gam_softmax/` package structure (see [05_implementation_roadmap.md](05_implementation_roadmap.md))
- [ ] Set up W&B project
- [ ] Implement `Trainer`, `eval/classification.py`, `data/text/sst5.py`
- [ ] Implement softmax + CE loss; train BERT-base on SST-5; verify number
- [ ] Read AS-Softmax paper end-to-end (you've read the notes; read the paper)
- [ ] Read AM-Softmax, Sparsemax (Tier-1, see [09_reading_list.md](09_reading_list.md))

**MS0, MS1.** End-of-week: vanilla softmax on SST-5 within 1% of literature.

---

## Week 2 — Reproduce AS-Softmax (2026-05-30 → 2026-06-05)

**Goals:** Reproduce the anchor paper.

- [ ] Implement `AS-Softmax` loss with scalar $\delta$
- [ ] Implement AS-Speed gradient accumulation
- [ ] Sweep $\delta \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$ on SST-5
- [ ] Reproduce ~58% accuracy headline number
- [ ] Reproduce $\rho \approx -0.95$ correlation
- [ ] Get CLINC150 + CoNLL-2003 dataloaders working

**MS2.** End-of-week: AS-Softmax within ±0.5% of paper. **Do not move on
without this.**

---

## Week 3 — Finish baselines + Gate 1 (2026-06-06 → 2026-06-12)

**Goals:** All baselines coded, integration-tested, reproducible.

- [ ] Implement AM-Softmax, Sparsemax, Entmax-1.5, Label Smoothing, Focal,
      Power Softmax
- [ ] Unit tests for every loss (gradient check)
- [ ] Run all 8 baselines on SST-5, CLINC150, CoNLL-2003 (3 seeds each)
- [ ] Build the first results table — this is the "before" picture

**MS3.** Decision Gate 1:

> *Are baselines reproducible, and is the harness trustworthy?*

If yes → Phase 2. If no → spend Week 4 fixing the harness; do not start
GAM-Softmax development on a shaky foundation.

---

## Weeks 4–5 — Class-pair margins, part 1 (2026-06-13 → 2026-06-26)

**Goals:** Implement and test M1, M2, M3.

- [ ] Implement `MarginFunction` interface
- [ ] Implement `ClassPairSimilarity` (M1), `ClassPairConfusion` (M2),
      `ClassPairLowRank` (M3)
- [ ] **Run H1 quick-check** ([04_experimental_plan.md §10](04_experimental_plan.md))
- [ ] Sweep $(k, \delta_{\min}, \delta_{\max})$ for M3 on SST-5
- [ ] Pick winning M3 config; run on CLINC150, CoNLL-2003

End of Week 5: first headline number for M3 on text core. If it beats
AS-Softmax — celebrate, take a screenshot, continue. If not — see Gate 2.

---

## Weeks 6–7 — Class-pair margins, part 2 + Gate 2 (2026-06-27 → 2026-07-10)

**Goals:** Ablations on the class-pair direction; finalize Phase 2.

- [ ] Rank sweep $k \in \{2, 4, 8, 16, 32, n\}$ — find quality vs param plot
- [ ] Compare M1 vs M2 vs M3 on the same dataset (which parameterization wins?)
- [ ] Diagonal restriction ablation (does pair structure matter, or just
      per-class margin?)
- [ ] Run M3 on CIFAR-100 (first cross-modality probe — does the win transfer?)
- [ ] Start prove T1 and T4 (cheap theorems)

**MS6.** Decision Gate 2:

> *Does class-pair margin help, and is the win robust across datasets?*

- **Yes, ≥ 0.3% gain consistent:** proceed to Phase 3 (sample margins).
- **Marginal, ≤ 0.3% gain:** still proceed but flag it; sample/time may
  compose with it to give bigger wins.
- **No gain or regression:** see [08_risks_and_decisions.md R1](08_risks_and_decisions.md). Likely
  pivot to making sample-margins the lead direction.

---

## Week 8 — Sample margins, part 1 (2026-07-11 → 2026-07-17)

**Goals:** Implement and test M4, M5.

- [ ] Implement `SampleEntropy` (M4), `SampleFeatureNorm`, `SampleHead` (M5)
- [ ] Build the small MLP head architecture; choose normalization
- [ ] Test M4 on SST-5: does entropy-based give any calibration benefit?
- [ ] Test M5 with anchor-regularization $\mu$ sweep

---

## Week 9 — Sample margins, part 2 + Gate 3 (2026-07-18 → 2026-07-24)

**Goals:** Finish sample direction; calibration deep-dive.

- [ ] Run M4, M5 on text core
- [ ] Compute ECE / Brier / reliability diagrams (this is the calibration story)
- [ ] Generate the "δ(x) distribution" figure — what samples get small margins?
- [ ] Composition: try M3 + M5 (M8)

**MS7.** Decision Gate 3:

> *Does sample-dependent margin reduce ECE without hurting accuracy?*

- **Yes:** sample direction stays in the paper.
- **Mixed:** sample becomes an ablation, not a headline.
- **No:** cut sample direction from main results; mention as failed attempt
  in discussion (this is fine for a paper — honest negative results are good).

---

## Weeks 10–11 — Time-adaptive margins + Gate 4 (2026-07-25 → 2026-08-07)

**Goals:** Schedules.

- [ ] Implement constant / linear / cosine / cyclical / gradient-driven
- [ ] Schedule comparison on SST-5 + CLINC150 with M3 backbone
- [ ] Test if a good schedule can *replace* hyperparameter tuning of scalar $\delta$
- [ ] Compose: M3 + M6 (M9); M3 + M5 + M6 (M10)
- [ ] Start T3 proof (convergence under varying $\delta$)

**MS8.** Decision Gate 4:

> *Is the flagship M10 the best, or does a simpler combo (M3 or M9) win?*

- The answer determines which row is the "headline" in the paper.

---

## Weeks 12–13 — Cross-modality + comprehensive comparison (2026-08-08 → 2026-08-21)

**Goals:** Image and audio results.

- [ ] Run M3 and M10 on CIFAR-100 (full), CIFAR-10
- [ ] Run M3 and M10 on ESC-50
- [ ] If time: WikiArt, Speech Commands
- [ ] Generate all main-text figures (calibration plots, $M$ heatmap,
      rank-quality curve, $\delta(\tau)$ trajectory)
- [ ] Fill in the master results table

**MS9.** This is when you know if the paper has legs in non-text domains.

---

## Week 14 — Regression extension (stretch) (2026-08-22 → 2026-08-28)

**Status:** Cut if behind.

- [ ] Implement binned-regression head
- [ ] Implement distance-aware margin
- [ ] Run on UCI Boston / California (small-scale proof)
- [ ] If promising: UTKFace age estimation

If regression works, it's a separate paper section. If not, it's a single
paragraph in future work.

---

## Weeks 15–16 — Theory & writing (2026-08-29 → 2026-09-11)

**Goals:** Paper draft v1.

- [ ] Finalize T1, T2, T3 proofs
- [ ] Write Methods + Theory sections first (they're the most concrete)
- [ ] Write Experiments section (tables already exist)
- [ ] Write Intro + Related Work (these always take longer than expected)
- [ ] Write Abstract + Conclusion last
- [ ] Pass to mentor for read-through

**MS10.** Submission-ready draft.

---

## Buffer week (2026-09-12 → 2026-09-18)

Reserved for: unexpected experiments reviewer questions, fixing one thing
that breaks at the last minute, mentor revisions. **Don't plan work into
this week.**

---

## Decision-gate philosophy

At every gate, write a short note in the [README.md decision log](README.md#7-decision-log):

```
2026-07-10 — Gate 2: class-pair low-rank gained 0.4% on SST-5,
0.7% on CLINC150, 0.2% on CoNLL. Margin matrix heatmap shows
expected structure (off-diagonal block-sparse). Proceed to Phase 3.
```

If a gate gives a "no" answer, that's not failure — that's information. The
paper might look different than you expected. Honest negative results on
one direction sharpens the story for the directions that did work.

---

## What this timeline assumes

- **~25 hours/week** of focused work (reading + coding + writing). If you
  have less, drop the regression stretch first, then scale back ablations.
- **GPU is mostly available.** If it's shared, double the wall-clock
  estimates in [04_experimental_plan.md §8](04_experimental_plan.md).
- **You can ask your mentor** roughly weekly for feedback. Their throughput
  matters as much as yours.
- **No major dataset bugs.** SST-5, CLINC150 are well-known; Eurlex and
  WikiArt have more loader issues — budget extra time if you include them.
