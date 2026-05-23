# SoftMax Research Plan — Master Index

**Project:** Generalized Adaptive Margin Softmax (GAM-Softmax)
**Owner:** haitomns4173@gmail.com
**Started:** 2026-05-23
**Target venue:** ACL / EMNLP / ICLR / NeurIPS
**Compute:** Single consumer GPU (RTX 3090/4090 class)

---

## 1. One-paragraph vision

The Adaptive Sparse Softmax (AS-Softmax) paper showed that replacing the
"push $p_t \to 1$" objective with a margin condition $p_t - p_j \ge \delta$
gives better accuracy, better calibration, and ~1.2× speedup. But the margin
$\delta$ is a **single global scalar** — the same value for every class pair,
every sample, and every training step. This is mathematically convenient but
provably suboptimal once you start asking *which* pairs are hard, *which*
samples are ambiguous, and *when* a strict margin is even useful. **Our project
generalizes $\delta$ along three axes — class-pair, sample, and time — and
develops the theory + empirical evidence that this generalized margin is the
correct primitive for sparse, hard-negative-focused classification.** We extend
this to multi-label and binned regression, and validate across text, image,
and audio so the contribution is not modality-specific.

## 2. The plan, in one picture

```
                  STANDARD SOFTMAX
                 (push p_t → 1, dense)
                          │
                          ▼
                    AS-SOFTMAX (Lv et al. 2023)
                  p_t − p_j ≥ δ   (δ is scalar)
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
        Direction 1   Direction 2   Direction 3
        δ_{t,j}        δ(x)          δ(t)
        class-pair     sample        schedule
            │             │             │
            └─────────────┼─────────────┘
                          ▼
              GAM-SOFTMAX (this project)
        p_t − p_j ≥ δ_{t,j}(x, t)
                          │
                ┌─────────┼─────────┐
                ▼         ▼         ▼
              Theory   Text+CV+Audio  Regression
              (bounds) (main results) (extension)
                          │
                          ▼
                   Conference paper
```

## 3. Document index

| # | File | Purpose |
|---|---|---|
| 0 | [00_research_synthesis.md](00_research_synthesis.md) | What the 3 source notes say + state of the art |
| 1 | [01_problem_and_gaps.md](01_problem_and_gaps.md) | Problem statement, 5 specific gaps, research questions, hypotheses |
| 2 | [02_proposed_method.md](02_proposed_method.md) | GAM-Softmax formulation, parameterizations, defaults |
| 3 | [03_theoretical_plan.md](03_theoretical_plan.md) | Math goals — bounds, identifiability, convergence |
| 4 | [04_experimental_plan.md](04_experimental_plan.md) | Datasets, baselines, metrics, ablations, compute budget |
| 5 | [05_implementation_roadmap.md](05_implementation_roadmap.md) | Repo structure, modules, milestones |
| 6 | [06_timeline_and_milestones.md](06_timeline_and_milestones.md) | 16-week schedule with decision gates |
| 7 | [07_paper_outline.md](07_paper_outline.md) | Section-by-section outline + figure/table list |
| 8 | [08_risks_and_decisions.md](08_risks_and_decisions.md) | What could fail and what to do |
| 9 | [09_reading_list.md](09_reading_list.md) | Prioritized literature |

## 4. The first thing to do (Week 0–1)

Before any new method work:

1. **Set up the repo skeleton** (see [05_implementation_roadmap.md](05_implementation_roadmap.md))
   — `gam_softmax/` package, `experiments/` runner, `configs/` directory.
2. **Reproduce AS-Softmax on SST-5** with a BERT-base backbone. Target accuracy
   from the original paper is ~58.4%. **This is your sanity check.** If you
   can't reproduce, nothing downstream is trustworthy.
3. **Implement the four baselines you'll re-use everywhere**: Softmax,
   AS-Softmax, AM-Softmax, Sparsemax. Lock the training loop, optimizer, seeds.
4. **Read the Tier-1 papers** in [09_reading_list.md](09_reading_list.md).

If step 2 lands within ±0.5% of the reported number, you have a trustworthy
baseline harness and can start writing new losses.

## 5. The narrative for the paper, in one sentence

> *The right primitive for sparse-softmax classification is not a scalar margin
> but a structured margin function $\delta_{t,j}(x, t)$ — and once you give the
> model that structure, you get measurable gains in accuracy, calibration, and
> sample efficiency across text, vision, audio, and binned regression.*

## 6. Status tracker

Update this section as you make progress.

| Phase | Status | Notes |
|---|---|---|
| Phase 0: Repo setup | not started | |
| Phase 1: Reproduce baselines | not started | |
| Phase 2: Class-dependent margins | not started | |
| Phase 3: Sample-dependent margins | not started | |
| Phase 4: Time-adaptive margins | not started | |
| Phase 5: Combined + cross-modality | not started | |
| Phase 6: Regression extension (stretch) | not started | |
| Phase 7: Theory + writing | not started | |

## 7. Decision log

Record every non-obvious choice you make so the paper's "why" is recoverable.

| Date | Decision | Reasoning |
|---|---|---|
| 2026-05-23 | Primary direction: generalized margins (extend AS-Softmax), not new function family | User decision; lower risk, builds on validated idea |
| 2026-05-23 | Text = primary, image/audio = breadth, regression = stretch | Single-GPU constraint; conference paper needs one strong narrative |
| 2026-05-23 | Method name: GAM-Softmax | Distinct from existing AM-Softmax, AS-Softmax; signals "generalized" |

## 8. How to use these docs

- These docs are **a plan, not a contract.** Update them as you learn.
- The timeline in [06_timeline_and_milestones.md](06_timeline_and_milestones.md)
  has **decision gates** at Weeks 4, 7, 11. At each gate, look at what worked
  and re-plan the rest. If a direction is dead, kill it and shift resources.
- The risk doc ([08_risks_and_decisions.md](08_risks_and_decisions.md)) is the
  decision tree — read it when something breaks.
- The paper outline ([07_paper_outline.md](07_paper_outline.md)) is your
  north star. Every experiment should map to a figure, table, or claim in
  that outline. If it doesn't, ask whether it's worth running.
