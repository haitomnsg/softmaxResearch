# Where We Are & What To Do Next (plain-English version)

*Last updated: 2026-07-31. This is the simple companion to `README.md` and `docs/`.
No math required to read it.*

---

# ✅ PROJECT CONCLUDED — 2026-07-31

**The full write-up is [FINAL_REPORT.md](FINAL_REPORT.md). Read that. Everything
below this banner is the history that led there, kept so the "why" is recoverable.**

## The whole thing in six lines

- We tried to show that AS-Softmax's **one fixed gap δ** should be a **smart gap**
  that varies by class-pair, by example, and over time.
- **It didn't work on accuracy.** Across four testbeds, no version beat plain
  cross-entropy by more than the run-to-run randomness.
- **The reason is upstream of us:** AS-Softmax — the published method we were
  extending — doesn't reliably beat plain cross-entropy in our setup either. You
  can't improve on an advantage that isn't there.
- **One thing did work.** Letting the gap go **negative** for examples the model
  distrusts kicks those examples out of training entirely. Under 40% wrong labels
  that cut memorization of the bad labels by about **4×** and left the finished
  model **+3.7 points** better than cross-entropy.
- **But** it doesn't improve the *best* score during training, only the *final*
  one. If you can early-stop on clean data you get nothing; if you can't — which
  is the real situation when labels are noisy — you get the +3.7.
- **We stopped** because everything left in the plan assumed the core idea worked.

```
[engine built] -> [baselines] -> [H1 flat on clean text] -> [noisy SST-5 flat]
     (done)         (done)             (done)                   (done)
                                                                   |
                                                                   v
                              [E1: noisy 20NG + sample axis] -> [CONCLUDED]
                                    (done, one real finding)
```

---

## ⏸ Earlier active direction (2026-06-12) — superseded by the banner above

**The real problem isn't the code — it's the testbed.** On clean, balanced text
(SST-5, 20NG) the method we extend, AS-Softmax, barely matches plain cross-entropy,
so our GAM method has no floor to build on. Margin/masking losses are *designed* for
harder regimes. **Decision (user, 2026-06-12): test in a hard regime — noisy labels —
where masking is supposed to win.**

Why noisy labels: AS-Softmax stops pushing once the correct class wins by the margin,
so it can't memorize wrong labels the way plain CE does. It's the cheapest hard regime
(reuses SST-5/20NG, just corrupts a fraction of training labels) and makes a clean figure.

**What's already built (this session):**
- `gam_softmax/data/label_noise.py` — seeded symmetric label-noise utility (11 unit tests).
- `label_noise` / `noise_seed` knobs in both data loaders; `--label-noise` override on `run.py`.
- `experiments/noise_sweep.py` — runs softmax / AS-Softmax / GAM across noise levels × seeds,
  writes a table with two key columns: **AS−CE** (does the foundation hold?) and **GAM−AS**
  (does our margin help?). Smoke-tested on GPU at 40% noise — runs clean.

**✅ RAN (2026-06-13). Verdict: INCONCLUSIVE — no hard-regime win.** SST-5, noise {0,0.2,0.4},
2 seeds, all 18 runs clean (one cosmetic Unicode crash in the final print, since fixed; data intact).

| noise | softmax | as_softmax | gam_m3 | AS−CE | GAM−AS |
|---|---|---|---|---|---|
| 0%  | 0.5104 | 0.5204 | 0.5195 | +1.0% | −0.1% |
| 20% | 0.4869 | 0.4823 | 0.5041 | −0.5% | +2.2% |
| 40% | 0.4677 | 0.4778 | 0.4764 | +1.0% | −0.1% |

- The hoped-for "gaps widen as noise rises" did **not** appear. All gaps (~1–2%) are inside the
  seed-to-seed scatter (±1–1.6% for softmax/AS). No clean separation.
- Checked the **final-epoch** model too (where memorization shows, not just best-on-clean-val):
  the seed-42 trace where AS resists noise is cancelled by seed-43 where plain CE wins. Still a wash.
- **Only repeatable signal:** GAM has the lowest seed-to-seed variance at every noisy level
  (±0.3% vs ±1–1.6%) — echoes the earlier "GAM overfits slower / more stable" finding. A
  *robustness* signal, not an accuracy one.
- **Why SST-5 probably can't show this:** 5 classes + early-stopping on a *clean* val set already
  protects every method from memorizing noise — muting the exact effect masking is meant to provide.

**This is the third setup where AS-Softmax doesn't convincingly beat plain CE** (clean SST-5,
clean 20NG, noisy SST-5). Decision point — pick one (see the chat discussion of 2026-06-13):
  - **(A) decisive test:** move the sweep to 20NG (20 classes) + higher noise (0.4, 0.6) + 3 seeds.
    Give the hypothesis its best shot; if it fails *there*, the pivot is fully justified.
  - **(B) reframe:** stop chasing accuracy, build the paper around GAM's stability/robustness
    (the one repeatable thread) — calibration + train/val memorization gap as the headline metrics.
  - **(C) pivot axis:** drop class-pair, build the sample-based (Stage 3) or time-based (Stage 4) margin.

*Everything below is the earlier history that led here — kept for context.*

---

## What this project is (no math)

A normal classifier is trained to push the correct answer's probability all the way to
**1.0** (100% sure), for every example, forever. That's wasteful.

The paper we build on (**AS-Softmax**) says: don't do that. Just make the correct answer
beat each wrong answer by a fixed **gap** — say 0.15 — and once you're winning by that
gap, **stop pushing**. Think of a student who only needs to pass an exam by a safe
margin, not score 100% on every question. This trains faster and generalizes better.

The catch: their gap is **one single number** used for everything — every pair of
classes, every example, every moment in training.

**Our project (GAM-Softmax) makes the gap smart instead of fixed:**

| Axis | Plain meaning | Example |
|---|---|---|
| **Class-pair** | Different gap for different pairs of classes | "cat vs dog" is hard → big gap; "cat vs airplane" is easy → small gap |
| **Sample** | Different gap for easy vs hard examples | A blurry, ambiguous photo gets a different gap than a clear one |
| **Time** | Gap changes over the course of training | Start loose, tighten up later |

**The bet:** a smart, structured gap beats one fixed number. The whole project is about
proving that, first on text, then on images/audio, then writing it up as a paper.

---

## The 7 stages at a glance

| Stage | Plain description | Status |
|---|---|---|
| 0. Build the engine | Loss functions, trainer, data, runner, tests | **DONE** |
| 1. Reproduce baselines | Get plain softmax + AS-Softmax running with sane numbers | **DONE** (softmax 51.1%, AS-Softmax 52.2%, 3 seeds; `runs/baselines/summary.md`) |
| 2. Class-pair gaps (the first "smart gap") | Variant M3 is coded; the H1 test must pass | **TESTED both ways — H1 FAIL.** Fixed (52.10%) and learnable (51.83%) class-pair margins both tie/underperform AS-Softmax (52.17%) on SST-5. Decision: tune, switch dataset, or pivot axis — see Step 4 |
| 3. Sample-based gaps | Harder examples get different gaps (M4–M5) | not started |
| 4. Time-based gaps | Gap changes on a schedule (M6–M7) | not started |
| 5. Combine + other data types | Best variant on images/audio (M8–M10) | not started |
| 6. Regression extension | Stretch goal | not started |
| 7. Theory + write the paper | Math proofs + the actual paper | not started |

> Note: only **one** smart-gap variant exists so far (M3 = class-pair). The full plan in
> `docs/` lists ~10 of them (M1–M10). Each new one is a small new file — the code is
> already set up to add them easily. But none of that matters until the crash in Stage 2
> is fixed and H1 gives a verdict.

---

## What's built and working (Stage 0 + most of Stage 1)

- **9 loss functions** in `gam_softmax/losses/`: plain softmax, AS-Softmax, AM-Softmax,
  sparsemax, entmax, focal, label-smoothing, power-softmax, and **GAM-Softmax (ours)**.
- **The training loop** (`gam_softmax/training/trainer.py`), the **SST-5 text data
  loader**, the **BERT-base model wrapper**, config loading, and seeding.
- **The experiment runner** (`experiments/run.py`) — one command runs one experiment
  from one config file.
- **The M3 "smart gap"** (`gam_softmax/margins/classpair_lowrank.py`) — the first
  class-pair gap variant.
- **A real test suite** (`tests/test_losses.py`, ~500 lines) that checks the math of
  every loss, including a test proving GAM-Softmax behaves exactly like AS-Softmax when
  the gap is made constant. **The math is verified in isolation — the crash is a
  GPU/runtime problem, not a "the formula is wrong" problem.**
- **Baselines actually ran on the GPU.** From `runs/h1_quick_check/`:
  - plain softmax: 51.2% average (3 seeds)
  - AS-Softmax: ~51.5% (2 of 3 seeds finished; 1 crashed)

---

## What's broken right now (the one blocker)

**GAM-Softmax (our method) crashed on all 3 runs in the H1 experiment and produced no
numbers.** See `runs/h1_quick_check/summary.md` — the `gam_m3` row is all `nan`, and the
H1 verdict line never got written because there was nothing to compare.

What the logs show:
- One AS-Softmax run (seed 44) **died in the middle of a training step with no error
  message** — that signature usually means a **GPU memory blowup** or the known
  **non-deterministic GPU attention path faulting** on this 6 GB laptop card.
- Every GAM run log is **completely empty** — the processes died instantly at startup.
  That strongly suggests the earlier hard crash **left the GPU in a bad state**, so every
  run after it died immediately.

Important: our GAM loss uses **essentially the same GPU memory** as AS-Softmax (the extra
gap-matrix is tiny for a 5-class problem), so this is **probably not** "GAM is too heavy."
It's most likely a one-off GPU fault that poisoned the rest of the batch. **One isolated
test (step 1 below) will tell us for sure.**

Also worth knowing (a real finding, not a bug): **5 epochs overfits SST-5.** In the
AS-Softmax run, validation accuracy peaked at epoch 0 (52.4%) and *dropped* to 46% by
epoch 4 while training loss kept falling. We'll want fewer epochs / early stopping
(Stage 1 cleanup below).

---

## What to do next — step by step

Do these **in order**. Each step has a clear "done when" so you know when to move on.

### Step 1 — Reproduce the crash on its own (diagnosis)
Run **only** the GAM method, fresh GPU, tiny step budget, so you see the real error
instead of an empty log:
```
conda activate gam
python experiments/run.py --config configs/m3_classpair_lowrank/sst5.yaml --max-steps 50
```
- **If it finishes 50 steps cleanly** → GAM is fine on its own; the H1 batch crash was the
  GPU getting wedged by the earlier AS-Softmax fault. Skip to Step 3.
- **If it crashes** → read the actual error now visible in the terminal. Most likely a CUDA
  out-of-memory or the attention backend. Go to Step 2.
- **Done when:** you've seen either 50 clean steps or a real, readable error message.

### Step 2 — Fix the crash (only if Step 1 crashed)
Try these in order, cheapest first. Re-run the Step 1 command after each:
1. Watch GPU memory while it runs (`nvidia-smi -l 1` in another terminal). If it's near
   6 GB → memory. Lower `batch_size` from 16 to 8 in
   `configs/m3_classpair_lowrank/sst5.yaml` (and add gradient accumulation later if you
   want the effective batch back).
2. If it dies in the backward pass with no message → it's the non-deterministic attention
   path. In `experiments/run.py`, near the top of `main()`, add
   `torch.backends.cuda.enable_mem_efficient_sdp(False)` (forces a stabler attention
   kernel), or try `torch.use_deterministic_algorithms(True, warn_only=True)`.
3. As a safety net for the H1 batch, add `torch.cuda.empty_cache()` at the end of a run so
   one run can't starve the next.
- **Done when:** the Step 1 smoke command finishes 50 steps with no NaN and prints a
  `best_val_acc`.

### Step 3 — Make the H1 batch robust, then re-run it
The H1 runner (`experiments/h1_quick_check.py`) already runs each method in its own
process — good. The fix is just making sure one crash can't poison the rest. Once Step 1
is clean:
```
python experiments/h1_quick_check.py
```
- **Done when:** `runs/h1_quick_check/summary.md` has **real numbers in all three rows**
  (softmax, as_softmax, gam_m3) and an `H1 verdict: PASS/FAIL` line at the bottom.

### Step 4 — Read the H1 verdict (the first real decision point)
> **✅ RAN (2026-05-28). Verdict: FAIL (a tie).** gam_m3 **52.10%** vs AS-Softmax **52.17%**
> over 3 seeds = **−0.06%** (criterion was ≥ +0.3%). Crash is fixed; all runs clean. As the
> note below predicted, M3's margin is *fixed and random* (u/v never learn), so a tie is the
> expected, non-informative outcome — **not** a refutation of the idea. Decided next move:
> **make the class-pair margin learnable, then re-run H1** before judging the class-pair axis.
> Full table: `runs/stage2_gam_m3/summary.md`.
>
> **✅ DONE (2026-05-29). Learnable margin ran — verdict still FAIL.** gam_m3' (learnable)
> = **51.83%** vs AS-Softmax **52.17%** = **−0.33%**. The u/v parameters did train (M3' ≠ M3),
> but the learned class-pair gap landed slightly *below* both AS-Softmax and the fixed-random
> M3. All four methods sit in a ~51–52% band within ~1 std — **the class-pair axis shows no
> measurable benefit on SST-5**. Full table + options: `runs/stage2_gam_m3_learnable/summary.md`.
>
> **Option (1) tuning DONE (2026-05-29) — under-tuning ruled out.** A dedicated higher
> `margin_lr` for u/v monotonically *hurt* (1e-3 → 50.6%, 1e-2 → 50.1%, vs 52.5% at the
> original tiny LR; all below AS-Softmax 52.8% on seed 42). So the flat H1 result is real, not
> a tuning artifact — letting the class-pair gap learn *more* makes it *worse*. See
> `runs/stage2_tuning/summary.md`.
>
> **Class-pair-on-SST-5 is closed.** Remaining options: (2) re-test on a dataset with real
> headroom (SST-5 ties ~52% for every method, so nothing can win here); (3) pivot to the
> sample-based (Stage 3) or time-based (Stage 4) axis. The testbed (option 2) is the more
> likely bottleneck, since *no* method separates on SST-5.
>
> **Option (2) tried — 20 Newsgroups screen (1 seed, 2026-05-29).** Real testbed (~71%, methods
> separate). GAM 71.18% > AS-Softmax 70.59% (+0.6%, first positive for the core hypothesis),
> but plain softmax wins at 71.46% and AS-Softmax < plain CE. Most robust finding: GAM was
> still *rising* at epoch 3 while softmax/AS-Softmax peaked at epoch 1 and overfit — GAM
> overfits slower. See `runs/stage2_20ng/summary.md`.
>
> **PAUSED for reassessment (user, 2026-05-29).** Core tension: AS-Softmax (the method we
> extend and aim to beat) does not reliably beat plain cross-entropy in our setup (won by ~1%
> on SST-5, lost by ~0.9% on 20NG). On balanced/clean text classification plain CE is a strong
> baseline — margin/masking methods are usually motivated in harder regimes (very many classes,
> long-tailed, noisy labels). Reassessment directions in this file's companion discussion:
> (a) reframe contribution around overfitting-robustness/calibration (where GAM showed a real
> trajectory difference); (b) move to a regime where margins are known to help; (c) pivot axis;
> (d) reposition the claim. No more compute until direction is chosen.

Open `runs/h1_quick_check/summary.md`. The criterion: **GAM-M3 must beat AS-Softmax by at
least 0.3% accuracy, averaged over 3 seeds.**
- **PASS** → the core idea has a pulse. Continue to Step 5.
- **FAIL** → don't kill the idea yet. The current M3 uses a **fixed, random** smart-gap
  (its `u`/`v` parameters never actually learn — see the long comment in
  `classpair_lowrank.py`). So a FAIL only means "*random* structure didn't help." The next
  thing to try is making the gap **actually learnable** before giving up (this is a known,
  planned follow-up, not a redesign).
- **Done when:** you've written one line in `README.md` §7 (the decision log) recording the
  verdict and what you decided.

### Step 5 — Clean up the baselines (so the final comparison table is trustworthy)
> **✅ DONE (2026-05-28).** Crash fixed by disabling memory-efficient SDP attention in
> `experiments/run.py` (it was faulting mid-backward on the 6 GB GPU). Epochs cut 5 → 3 to
> stop overfitting. All 6 baseline runs finished clean — softmax **51.1%**, AS-Softmax
> **52.2%** (3 seeds), so **AS ≥ softmax holds**. Full table: `runs/baselines/summary.md`.
> One sub-item left (no GPU needed): confirm whether the README's ~58.4% target is the same
> SST-5 setup we use (likely a different split/model). Until then, trust the *internal*
> softmax-vs-AS comparison, which is apples-to-apples.

- Fix the overfitting: reduce `epochs` (try 3) or add early-stopping/best-checkpoint to the
  trainer. Right now it trains past the best point.
- Sanity-check the comparison: AS-Softmax should be **at least as good as** plain softmax.
  Right now they're basically tied (~51.5% vs ~51.2%) — confirm that holds after the epoch
  fix.
- Settle the reproduction-target question: `README.md` cites ~58.4% as the AS-Softmax
  target, but we get ~52%. Figure out whether 58.4% is even the same SST-5 setup (it may
  use a different split/model). Adjust the claim so we're comparing apples to apples.
- **Done when:** baselines are stable, AS-Softmax ≥ softmax, and the README target number
  is one we can actually stand behind.

### Step 6 and beyond — build out the rest of the smart-gap variants (Stages 3–7)
Only after H1 passes and baselines are clean. Each new variant follows the same recipe and
is mostly copy-paste-shaped:
1. Write one new file in `gam_softmax/margins/` (a new gap function).
2. Register it in `MARGIN_REGISTRY` in `experiments/run.py`.
3. Add one config YAML under `configs/`.
4. Add tests in `tests/test_losses.py`.

The planned variants (from `docs/05_implementation_roadmap.md` §4): sample-based gaps
(Stage 3), time-based gaps (Stage 4), combinations + images/audio (Stage 5), then theory
and the paper (Stage 7). **Don't start these until Steps 1–5 are green** — there's no point
building 9 more variants on top of a method that hasn't beaten the baseline once.

---

## Command cheat-sheet

```
conda activate gam        # this machine has no `python` on PATH — always activate first

# run one experiment (one config = one result)
python experiments/run.py --config configs/baselines/softmax_sst5.yaml
python experiments/run.py --config configs/m3_classpair_lowrank/sst5.yaml

# quick smoke test (skips full training — use this to check the pipeline / debug crashes)
python experiments/run.py --config <yaml> --max-steps 50

# run a specific seed (used by the H1 multi-seed batch)
python experiments/run.py --config <yaml> --seed 43

# the H1 decision experiment (softmax vs AS-Softmax vs GAM-M3, 3 seeds each)
python experiments/h1_quick_check.py

# tests (must stay fast, ~30s)
python -m pytest tests/ -v
```

---

## Map of the files that matter

| File | What it is |
|---|---|
| `experiments/run.py` | The one entry point. Runs any experiment from a config. |
| `experiments/h1_quick_check.py` | The make-or-break experiment for the whole idea. |
| `gam_softmax/losses/gam_softmax.py` | **Our loss.** The star of the project. |
| `gam_softmax/losses/as_softmax.py` | The baseline we're trying to beat. |
| `gam_softmax/margins/classpair_lowrank.py` | The one "smart gap" (M3) that exists so far. |
| `gam_softmax/training/trainer.py` | The training loop all experiments share. |
| `configs/` | One YAML per experiment. Baselines + the M3 config. |
| `runs/h1_quick_check/` | Results of the H1 experiment (currently incomplete). |
| `tests/test_losses.py` | Proof the loss math is correct. |
| `README.md` + `docs/` | The full research/math plan (this file is the simple version). |

---

## A note on the docs being out of sync

If you read the other files, you'll see small contradictions — this is normal for an
in-progress project, but here's the honest reconciliation so you trust this file:

- `README.md` §6 status table says "Phase 1 in progress, full SST-5 reproduction TODO."
- The git history says "M2 Completed" and "M4 and M5 Done."
- **Reality:** the *code* for those milestones is written and committed. What's actually
  left is the runtime problem — **getting our method to finish a run on the GPU and reading
  off the H1 verdict.** That's Steps 1–4 above.

When Steps 1–4 are done, update `README.md` §6 (status table) and §7 (decision log) to
match — that keeps the paper's "why" recoverable later.
