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
| **PROJECT** | **concluded 2026-07-31** | Core hypothesis not supported. **Read [FINAL_REPORT.md](FINAL_REPORT.md) first** — it supersedes the plan in this file |
| Phase 0: Repo setup | **done** | MS0–MS3 complete; gam_softmax package + 8 baselines coded & unit-tested |
| Phase 1: Reproduce baselines | **done** | SST-5 (3 seeds): softmax 51.1%, AS-Softmax 52.2% (AS ≥ softmax ✓). See `runs/baselines/summary.md` |
| Phase 2: Class-dependent margins | H1 tested, no benefit on clean text | Both fixed (52.10%) and learnable (51.83%) class-pair margins tie/underperform AS-Softmax (52.17%) on SST-5; 20NG gave a faint +0.6%. Root issue: AS-Softmax itself doesn't reliably beat plain CE on clean balanced text. **Direction chosen (2026-06-12): move to a hard regime (noisy labels) where masking is motivated.** |
| Phase 2b: Hard-regime testbed (noisy labels) | **done** | SST-5 sweep inconclusive (all gaps inside seed scatter). Decisive 20NG + 40% noise experiment run as E1 — see below |
| Phase 3: Sample-dependent margins | **done** | M4 `SampleConfidenceMargin` built + tested. δ may go **negative**, ejecting suspect samples from the loss — makes the small-loss trick a special case of a margin. **The project's one positive result:** ~4× less memorization of corrupted labels, +3.7 pts final-epoch accuracy over CE; **no peak-accuracy gain** |
| Phase 4: Time-adaptive margins | **partial** | The `Schedule` (linear warmup of the margin cap) is used by every GAM variant, so the time axis is exercised throughout — but never ablated on its own |
| Phase 5: Combined + cross-modality | **code only** | M6 (class-pair × sample × time) implemented and unit-tested via the nested `base:` config block; run cut for compute. Cross-modality never attempted (single 6 GB GPU) |
| Phase 6: Regression extension (stretch) | not started | Dropped — depended on the core hypothesis holding |
| Phase 7: Theory + writing | **concluded** | [FINAL_REPORT.md](FINAL_REPORT.md) is the write-up. No conference submission: the headline hypothesis did not survive, and the result that did is a conditional robustness finding, not an accuracy one |

## 7. Decision log

Record every non-obvious choice you make so the paper's "why" is recoverable.

| Date | Decision | Reasoning |
|---|---|---|
| 2026-05-23 | Primary direction: generalized margins (extend AS-Softmax), not new function family | User decision; lower risk, builds on validated idea |
| 2026-05-23 | Text = primary, image/audio = breadth, regression = stretch | Single-GPU constraint; conference paper needs one strong narrative |
| 2026-05-23 | Method name: GAM-Softmax | Distinct from existing AM-Softmax, AS-Softmax; signals "generalized" |
| 2026-05-24 | All 8 baselines implemented in one session; AM-Softmax owns its own learnable W | Cleaner than coupling the loss to the model's head; trainer pre-existing optimizer was extended to include loss params |
| 2026-05-24 | torch 2.6.0+cu124 + transformers 5.9 in conda env `gam` | Latest stable PyTorch CUDA wheels at install time; bumped requirements.txt upper bounds to match |
| 2026-05-28 | Disable memory-efficient SDP attention on CUDA in `experiments/run.py` | Its non-deterministic backward intermittently faulted mid-step on the 6 GB RTX 3050 with no Python error; forcing the flash kernel made all runs finish cleanly |
| 2026-05-28 | Baseline epochs 5 → 3 on SST-5 | Val acc peaks at epoch 0–1 then overfits hard (~6 pt drop by epoch 4); 3 epochs captures the peak without wasted compute |
| 2026-05-28 | Baselines reproduced: softmax 51.1%, AS-Softmax 52.2% (3 seeds); AS ≥ softmax holds | Stage 1 sanity check passed. README's ~58.4% target is a different SST-5 setup (open item), not our internal comparison |
| 2026-05-28 | H1 verdict: **FAIL** — gam_m3 52.10% vs AS-Softmax 52.17% (−0.06%, criterion ≥ +0.3%) | gam_m3's class-pair margin is a *fixed random* low-rank matrix (u/v don't learn), so a tie is expected. Not a refutation; next step is making the margin learnable, then re-run H1. See `runs/stage2_gam_m3/summary.md` |
| 2026-05-29 | Added learnable margin (STE flag `learnable_margin` in GAMSoftmaxLoss) so u/v actually train | Fixed-random M3 was an inconclusive test; STE keeps forward identical to AS-Softmax but routes gradient to the margin params. Unit-tested |
| 2026-05-29 | H1 re-run with LEARNABLE margin: **FAIL** — gam_m3' 51.83% vs AS-Softmax 52.17% (−0.33%) | The real test of the core hypothesis on the class-pair axis. Learnable gap did not beat (slightly below) scalar AS-Softmax; all methods tie ~52% on SST-5. Class-pair axis shows no benefit here. Options: tune margin-LR/ste_temp, try a dataset with headroom, or pivot axis. See `runs/stage2_gam_m3_learnable/summary.md` |
| 2026-05-29 | Tuning screen (margin_lr 1e-3/1e-2, ste_temp 0.05): under-tuning ruled out — higher margin LR monotonically **hurt** (down to 50.1%) | Added optional `margin_lr` param group in run.py. The flat H1 result is not a tuning artifact; letting the class-pair gap learn more makes it worse. Bottleneck is the testbed (SST-5 ties ~52% for all methods). Decision: stop class-pair-on-SST-5; switch dataset or pivot axis. See `runs/stage2_tuning/summary.md` |
| 2026-05-29 | Added 20 Newsgroups loader (sklearn, headers stripped) as a discriminating testbed | SST-5 couldn't separate any method; 20NG (20 classes) reaches ~71% with real spread. New loader + 3 configs (softmax/as_softmax/gam-learnable) |
| 2026-05-29 | 20NG screen (1 seed): GAM 71.18% > AS-Softmax 70.59% (+0.6%); softmax best at 71.46% | First positive signal for GAM>AS-Softmax. GAM was still rising at epoch 3 (overfits slower) while others peaked early. One seed, within noise; AS-Softmax < plain CE. Next: 3 seeds + more epochs. See `runs/stage2_20ng/summary.md` |
| 2026-06-12 | After reassessment, **direction = harder testbed (noisy labels)** over reframing/axis-pivot | The blocker isn't a bug, it's that AS-Softmax ≈ plain CE on clean balanced text, so GAM has no floor to build on. Margin/masking losses are *motivated* in hard regimes; symmetric label noise is the cheapest, best-motivated one (masking can't memorize corrupted labels) and reuses existing data. User decision |
| 2026-06-12 | Built seeded symmetric label-noise infra: `gam_softmax/data/label_noise.py` + `label_noise`/`noise_seed` knobs in both loaders + `--label-noise` override + `experiments/noise_sweep.py` | Noise seed is decoupled from run seed so all methods see identical corrupted labels (fair comparison). 11 new unit tests; smoke-tested on GPU at 40% noise. Sweep not yet run (it's the next GPU job) |
| 2026-07-31 | **PROJECT CONCLUDED.** Full write-up in [FINAL_REPORT.md](FINAL_REPORT.md) | H0 (AS-Softmax > CE) and H1 (class-pair) not supported across four testbeds; H2 (sample axis) split — mechanism confirmed, no peak-accuracy gain. Stopping is the right call: every remaining item in the plan was premised on H0 holding, and the compute for cross-modality was never reachable on a 6 GB laptop GPU |
| 2026-07-31 | **E1 result: M4 sample-axis margin cuts memorization ~4× (2.7 pts above the no-memorization floor vs CE's 11.2) and holds +3.7 pts of final-epoch accuracy over CE — but peak accuracy is flat (68.06 vs 68.09).** | The benefit is conditional on the protocol: early-stopping on a *clean* val set discards it, but a clean val set is exactly what you lack under label noise. This is why six weeks of accuracy-only measurement looked flat — we were reading the wrong number. See `runs/final_20ng/summary.md` |
| 2026-07-31 | **E1 also found AS-Softmax memorizes corrupted labels MORE than plain CE** (14.2 vs 11.2 pts above the floor) | Directly contradicts the rationale for the 2026-06-12 noisy-label pivot ("masking stops the push, so it can't memorize"). Likely mechanism: masking retires easy negatives first, concentrating the remaining gradient budget onto hard examples — which are exactly the mislabeled ones. A negative result worth reporting on its own |
| 2026-07-28 | Built the sample axis (M4 `SampleConfidenceMargin`) with **δ allowed to go negative**, plus ECE + a memorization probe in the trainer | `p_t − p_j ∈ [−1,1]`, so nothing requires δ > 0. A negative δ masks classes that are *ahead* of the training label, collapsing a suspect sample's loss to ~0 — i.e. small-loss sample rejection expressed as a margin. Instrumentation added because accuracy alone had been flat and uninformative on every prior experiment |
| 2026-06-13 | **Noisy-label sweep (SST-5, noise {0,0.2,0.4}, 2 seeds): INCONCLUSIVE — no hard-regime win.** Best-val AS−CE = {+1.0%, −0.5%, +1.0%}, GAM−AS = {−0.1%, +2.2%, −0.1%}; all within the ~±2% seed scatter. Final-epoch (memorization) checked too — the seed-42 trace where AS resists noise is cancelled by seed-43 where CE wins. | Third setup where AS-Softmax fails to convincingly beat plain CE (clean SST-5, clean 20NG, now noisy SST-5). The only repeatable thread is **GAM's lower seed-to-seed variance** (±0.3% vs ±1–1.6% at noise>0) — a stability/robustness signal, not an accuracy one. SST-5 (5 classes + clean-val early stopping) likely too easy to bite. See `runs/noise_sweep_sst5/summary.md`. Decision pending: decisive 20NG+high-noise test vs reframe around robustness |

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
