# Plan v2 — Negative Margins for Learning with Noisy Labels

**Status:** ACTIVE (started 2026-10-05). Replaces the original plan in files 00–09 as the working plan.
Files 00–09 describe the first hypothesis, which was tested and concluded negative on 2026-07-31.
They stay unchanged as the project's history. See [../FINAL_REPORT.md](../FINAL_REPORT.md) for
why that hypothesis stopped and what survived.

**Compute:** RTX 3080 Ti (12 GB), i7-12700K, 16 GB RAM, runs 24/7. One 20NG/BERT-base run ≈ 7.4 min.

---

## 1. The claim this plan tests

> *Small-loss sample rejection, the core heuristic of the noisy-label literature, is a margin
> loss with a negative, per-sample margin. Written that way (M4), it needs no oracle noise
> rate, no second network and no selection step. It matches or beats explicit small-loss
> selection and standard robust losses at resisting memorization of corrupted labels.*

What already exists (from E1, 2 seeds, 20NG, 40% symmetric noise, BERT-base):
- M4 memorized corrupted labels ~4× less than CE (+3.1 vs +11.5 pts over the no-memorization floor).
- M4's last-epoch accuracy beat CE by +2.8 pts. Its best-epoch accuracy gain was **not**
  established (+0.5, driven by one seed).
- AS-Softmax memorized **more** than CE. This is a separate negative finding worth reporting.

What it was missing (FINAL_REPORT §6 limitations): only 2 seeds; weak baselines (CE and
AS only); one noise level, symmetric noise only; no β sweep; 4 epochs; one dataset.

## 2. Hypotheses and what counts as passing

| ID | Hypothesis | Passes if (≥ 5 seeds, mean ± std, per-seed table reported) |
|---|---|---|
| **N1** | The E1 memorization result replicates | M4 memorizes less than CE on **every** seed, and its mean last-epoch accuracy is above CE by more than 2 std |
| **N2** | M4 is competitive with explicit small-loss selection | M4 last-epoch accuracy ≥ small-loss − 0.5 pts, **without** being told the noise rate |
| **N3** | M4 is competitive with robust losses | M4 ≥ max(GCE, SCE) on last-epoch accuracy at 40% noise |
| **N4** | The effect is caused by the mechanism | Memorization falls steadily as β rises (0 → 0.25 → 0.5 → 1.0), and β = 0 matches AS-Softmax |
| **N5** | It generalizes past synthetic text noise | On CIFAR-10N / CIFAR-100N (real human label noise), M4 beats CE and lands within reach of small-loss / GCE |

**Model-selection protocol (fixed now, before results).** Report three numbers per run:
(a) best epoch picked on the *clean* val set (an oracle, an upper bound only);
(b) last epoch; (c) epoch picked on a *noisy* val set, corrupted with the same noise process.
(c) is the realistic protocol, since a clean val set is exactly what you don't have under
label noise. The paper's headline uses (b) and (c), never (a) alone.

**Baseline fairness.** Each baseline gets the same small tuning budget as M4, chosen on the
noisy val set: GCE q ∈ {0.4, 0.7}; SCE (α, β) ∈ {(0.1, 1), (1, 1)}; small-loss
forget_rate ∈ {oracle, 0.5 × oracle}. M4 gets β ∈ {0.5, 1.0}.

## 3. Phases and decision gates

### Phase A — Confirm (running since 2026-10-05)
- 5 seeds (42–46): CE, AS-Softmax, M4 (β = 1), M4 (β = 0.5) at 40% noise. 20 runs, ~2.5 h.
- **Gate A (N1).** Fail → stop. The one positive result didn't replicate; write that up as
  an addendum to FINAL_REPORT. Pass → Phase B.

### Phase B — Baselines on 20NG (queued behind Phase A)
- GCE, SCE, single-network small-loss (code done, 12 tests): 5 seeds, ~2 h.
- Then the light tuning grid from §2 (~4 h).
- **Gate B (N2, N3).** If small-loss clearly beats M4, the contribution shrinks to "a
  unifying reframing". Decide then whether that's still a paper (workshop) or whether M4
  needs a fix, e.g. an adaptive threshold. If M4 ≥ the baselines → Phase C.

### Phase B′ — Gate-B fair re-test (user decision 2026-10-05, rule fixed before running)
Gate B failed, but the comparison was confounded against M4. Exactly two variants remove the confounds, and nothing else changes:
- `gam_m4_matched`: M4 with small-loss's rejection timing (β ramps 0 → full over the first 30%, then holds).
- `gam_m4_ce`: same rejection decisions, but every non-rejected sample trains with plain CE (`nonneg_margin: 1.0`).
- **Rule:** if the better variant's mean last-epoch accuracy (5 seeds) is more than 1 pt below small-loss (67.80), **stop and
  write up**. If it's within 1 pt, N2 counts as passed and we go to Phase C. No further M4 tuning in either case.

### Phase C — Close the E1 limitations (20NG)
- **Code:** noisy-val model selection in the trainer (protocol (c)); an asymmetric /
  class-conditional noise option in `label_noise.py`.
- β sweep {0, 0.25, 0.5, 1.0} → N4.
- Noise levels {0, 0.2, 0.4, 0.6} × {symmetric, asymmetric} for CE / small-loss / GCE / M4.
- 8 epochs instead of 4 (memorization compounds, so the gap should widen).
- Budget: ~150 runs ≈ 20 h, i.e. one day of wall clock.

### Phase C′ — Self-calibrating negative margin (pre-registered 2026-10-08, before any run)

Phase C showed M4's rejection fraction is **noise-rate-blind** (§6). The claim "no oracle noise rate" is only honest if the
rejected fraction tracks the noise actually present. C′ changes **one thing**: the suspicion statistic ``s_i`` that drives
``δ_i = clamp(δ_base − β·(2·s_i − 1), −1, 0.15)``. Everything else (β = 1, ramp 0 → 30%, AS base δ = 0.15, 8 epochs,
noisy-val selection, 5 seeds) is frozen at the Phase C recipe. Two variants, one file each:

- **`m4v2` (`loss_mixture`)** — per-sample EMA of −log p_t over training, a 1- vs 2-component Gaussian mixture on **log**
  losses, refit every 100 steps and chosen by BIC, **active only if the high component's mean loss is ≥ log C** (the loss of a
  uniform prediction: the model rates those labels below chance). Inactive → s_i = 0.5 for everyone (exactly AS-Softmax,
  nothing rejected). Active → s_i is the posterior of the high-loss component, and the high component's weight is logged as
  `est_noise_rate`, a free, checkable prediction. This is DivideMix's selection statistic expressed as a margin.
  *Design iteration before launch (2026-10-08), from a diagnostic with ground truth on the training set (one CE epoch, 20NG;
  no test accuracy involved): the first design fitted raw losses and gated on BIC alone. Raw-loss fits over-estimated the
  rate (0.59 at a true 0.40, precision 0.67; log fits 0.39–0.40, precision 0.82–0.87, AUC 0.94–0.96). And on **clean** data
  BIC still chose two components with weight 0.44–0.52, because unfitted clean samples form a second mode; since a rejected
  sample is never fitted, that would lock in ~half the clean set. The two modes differ in position: mislabeled ≈ 3.5 ≥
  log 20 = 3.0; unfitted-clean 1.5 → 0.77 and falling. Hence the chance-level anchor, which is not a tuned threshold. The same
  diagnostic predicts the stateless control will flag 21–29% of clean samples, which is what P1 on `m4_absgap` measures.*
- **`m4_absgap` (`abs_gap`)** — stateless control: s_i = σ((max_{j≠t} p_j − p_t) / temp). Suspicious iff some class currently
  beats the given label, so the rejected fraction falls on clean data and rises with noise by itself.

**Runs** (driver `experiments/phase_c.py`, tiers 4–5). Tier 4, sym 40%: m4v2, m4_absgap (temp 0.1), m4_absgap temp 0.3
(the one scale alternative, since temp is a new unit), and the Phase B fairness variants that never ran: GCE q = 0.4,
SCE (1, 1), small-loss at 0.5 × oracle. Tier 5, the other five conditions: the better of m4v2 / m4_absgap at tier 4 on
`sel`, plus the other if within 0.5. No further knobs after that.

**Pass criteria** for a variant V (paired over seeds, `sel` accuracy):

| | Criterion |
|---|---|
| **P1 clean cost** | sym 0%: V ≥ CE − 0.5 |
| **P2 adaptivity** | V's final `train_masked_ratio` increases monotonically over sym {0, .2, .4, .6}; for m4v2, `est_noise_rate` within ±0.10 of the true rate at {.2, .4, .6} |
| **P3 home condition** | sym 40%: V ≥ max(small-loss, small-loss ½, best GCE, best SCE) − 0.5, and memorization ≤ small-loss |
| **P4 asymmetric** | pair 40%: V ≥ CE, and within 2.0 of small-loss |

**Decision rule.** All four pass for some V → V is the method; Phase D starts with it. P3 fails → the method is dead at its
home condition; write up as a reframing (workshop / TMLR). P1 or P2 fails with P3 passing → report the noise-blindness as a
stated limitation and still go to Phase D, with the fairness-grid caveat. §5 kill criteria stay in force.

### Phase D — Real-noise benchmark (N5)
- **Code:** an image path. A CIFAR-10N/100N loader (the human noisy labels are public); a
  PreAct ResNet-18 model; generalize the trainer batch, which is currently
  `(input_ids, attention_mask, labels)`.
- Standard protocol (PreAct ResNet-18, SGD, ~120 epochs). Compare with published CIFAR-N
  numbers for CE / Co-teaching / GCE / ELR. DivideMix is cited, not re-implemented (it is a
  semi-supervised pipeline, not a loss).
- Estimate ~30–50 min/run with AMP. 5 seeds × ~6 methods × 3 label sets ≈ 2 days.

### Phase E — Theory + paper
- Formal statement: for a single sample, M4 with δ ≤ −max_j(p_j − p_t) masks every competitor, giving zero loss and
  zero gradient, so it equals hard rejection. Characterize the implicit rejection threshold
  as a function of (β, temp, batch statistics) and contrast it with small-loss's fixed
  budget (no oracle rate needed).
- Gradient analysis of why AS-Softmax *increases* memorization (it masks easy negatives
  first, so the remaining push concentrates on hard, often mislabeled, samples).
- Draft. Venue depends on Gate B / N5: main track (ICLR / NeurIPS / ACL) if N2 + N5 pass; a
  workshop or TMLR if the result is "a clean reframing that matches but doesn't beat
  small-loss".

## 4. Timeline (adjust at each gate)

| Week | Dates | Work | Gate |
|---|---|---|---|
| 1 | 2026-10-05 → 10-11 | Phase A + B runs; noisy-val selection + asymmetric noise code | **A**, **B** |
| 2 | 10-05 → 10-07 | Phase C sweeps (27 h GPU) | N4 read; C verdict |
| 2′ | 10-08 → 10-14 | Phase C′: index plumbing, `loss_mixture` + `abs_gap` margins, fairness grid, tiers 4–5 (~20 h GPU) | **P1–P4** |
| 3–4 | ~~10-15 → 11-08~~ | ~~Phase D: CIFAR-10N/100N (with the C′ winner)~~ — cut by the C′ rule (P3 failed) | ~~N5~~ |
| 2″–4 | 10-09 → ~10-25 | Phase E, moved up: theory section, mechanism runs, figures, draft (reframing → TMLR / workshop) | Go/no-go on venue |

## 5. Kill criteria (stated in advance)

- **Gate A fails** → stop, write an addendum.
- **M4 loses to small-loss by > 1 pt on 20NG AND on CIFAR-N** → the reframing still stands
  but the method doesn't. Don't tune M4 until it wins; report the result honestly.
- **The effect disappears under noisy-val model selection** → it's a last-epoch artifact,
  not a usable method. Report it as such.

## 6. Status

| Phase | Status | Results |
|---|---|---|
| A — confirm (5 seeds) | **PASSED** (2026-10-05) | M4 − CE last-epoch acc **+1.81** (paired sd 0.92, t = 4.42, positive on 5/5 seeds); memorization 17.2% vs 34.7% (lower on 5/5). Best-epoch acc +0.47 (t = 1.96, not established). β = 0.5 gives almost no effect (mem 34.5%), so the β response is non-linear (watch for N4). `runs/final_20ng/summary.md` |
| B — baselines | **FAILED as specified** (2026-10-05) | Final-epoch acc: GCE 67.90, small-loss 67.80, SCE 67.38 vs M4 65.65. All three beat M4 on 5/5 seeds by +1.7 to +2.3 pts (t ≥ 5.6) and memorize less (9–12% vs 17%), untuned. **Two confounds found, both favoring the baselines:** (1) M4's rejection is off until 30% and reaches full strength only at 100%, while small-loss is at full strength from 30%; (2) M4 sits on AS-Softmax, which memorizes more than CE. → Phase B′ re-test (user decision) |
| B′ — fair re-test | **N2 PASSED by the pre-registered rule; N3 narrowly FAILED** (2026-10-05) | `gam_m4_matched` last-epoch 67.41 ±0.60 vs small-loss 67.80 ±0.27 (−0.39, t = −1.04, a tie) **without the oracle noise rate**, and the lowest memorization of any method (8.1% vs GCE 9.3, SCE 9.1, small-loss 12.4). vs GCE 67.90: −0.49, so N3 (M4 ≥ max(GCE, SCE)) fails narrowly; ties SCE. Timing alone accounted for the gate-B deficit (65.65 → 67.41). `gam_m4_ce` was much worse (63.29), so the AS base is not a handicap. **`gam_m4_matched` is the M4 used from here on.** |
| C — limitations | **DONE** (2026-10-07, 140 runs, 27 h) | `runs/phase_c/summary.md`. **Headline (sym 40%, 8 ep, noisy-val selection):** M4 sel **67.44** ±0.55 is the best of 8 methods (GCE 67.34, CE 66.97, SCE 66.64, small-loss 65.61), final 66.72 (next best GCE 65.55), memorization 16.1% (lowest). **N2 ✔ N3 ✔** at 8 epochs. **N4:** memorization 96 / 96 / 54 / 16 % for β = 0 / .25 / .5 / 1 — monotone but threshold-shaped, not steady; explained by the clamp geometry (a sample is fully rejected iff β(2s−1) − δ_base ≥ p_max − p_t, so β = 0.25 can only reject gaps ≤ 0.10). **Pair-flip 40%: FAIL** (52.71, −6.7 vs small-loss, t = −3.6; −4.6 vs CE). Pair 20%: −0.7 vs small-loss. Sym 60%: −0.9 (ns). Sym 20%: tie, lowest mem (7%). **Clean (0%): FAIL**, −3.83 vs CE (t = −5.2); GCE costs −0.23. **Diagnosis:** M4's per-epoch `train_masked_ratio` is ≈ 0.99 at 0 / 20 / 40 / 60 % noise alike and `delta_eff` settles at −0.04 everywhere: the batch-relative z-score flags ~half of every batch as suspicious whatever the noise, so the rejected fraction is set by β, not by the data. β = 1 is a hidden fixed rate that happens to suit 40% symmetric noise. Small-loss scales 0 → 0.62 only because it is handed the oracle rate. → Phase C′ |
| C′ — self-calibrating margin | **DONE 2026-10-09 (55 runs, 11 h). Verdict for `m4_absgap`: P4 PASS, P1 / P2 / P3 FAIL → by the pre-registered rule, write up as a reframing; Phase D does not start.** Tier 5 (5 conditions): **pair 40%: 62.69 ±1.55, the best method of all** (+5.38 vs CE, t = 3.4; +3.27 vs small-loss, t = 3.3; +4.93 vs GCE; +9.98 vs M4); pair 20%: 68.04 (ties GCE 68.16, +0.42 vs small-loss). Clean: 69.88, −1.51 vs CE (t = −5.1; M4 was −3.83). Sym 20 / 60%: −0.51 / −0.37 vs CE (ns), −1.0 / −1.3 vs GCE. **Memorization 9.3 / 9.5 / 8.0 % at sym 20 / 40 / 60 %, the lowest of every method at 40 / 60 % and under pair noise** (at sym 20 % the noise-blind M4 is lower, 7.0; small-loss 29 / 26 / 13, GCE 16 / 24 / 31; corrected 2026-10-09), and 26% at pair 40% (small-loss 49, GCE 80). P2 as worded fails: the slot-level masked ratio is 0.996 everywhere (it is dominated by the AS base's masking of confident samples, so it was a poor adaptivity measure); the sample-level signal (memorization flat at 8–10% vs. small-loss scaling 13–29%) says the rule rejects *what it disagrees with*, not a fixed fraction. **Reading:** an absolute, rate-free rejection rule ("some class beats the label") is the most memorization-resistant method and the most robust to *structured* noise, at a 0.5–1.5 pt cost on clean / symmetric data where GCE wins. Tier 4 detail: | Sym 40%, `sel`: **m4_absgap 66.39 ±0.65** (−0.59 vs CE, ns; +0.78 vs small-loss, t = 2.3; **memorization 9.5%, the lowest of all 14 methods**); `m4v2` 65.33 ±0.90 (−1.64 vs CE, t = −3.0; mem 23%; **est. noise 0.35 ±0.01 vs true 0.40**, inside P2's ±0.10); `m4_absgap_t03` barely rejects (mem 96%), a non-method. Fairness grid: GCE q=0.4 67.20 < q=0.7 67.34; SCE (1,1) 66.73 ≈ (0.1,1) 66.64; small-loss ½-oracle 66.75 > oracle 65.61 — the untuned defaults were not under-tuned, so **P3's bar is GCE 67.34 − 0.5 = 66.84**. m4_absgap misses it by 0.45, m4v2 by 1.5. By the pre-registered decision rule the self-calibrating variants are not "the method" at the home condition; the noise-blind M4 (67.44) still is, with its known clean-cost / pair-noise failures. Tier 5 (P1, P2, P4 for m4_absgap) still runs: whether self-calibration fixes the clean cost and pair noise is the write-up's question either way |
| D — CIFAR-N | **cut** (2026-10-09) by the pre-registered C′ rule: P3 failed, so no real-noise benchmark | |
| E — theory + paper | **in progress** (started 2026-10-09) | Draft in [paper/](../paper/) (LaTeX, all sections written; not yet compiled, no TeX on the desktop). Theory: (1) δ_i < 0 ⇔ rejection, exactly (zero loss iff g ≤ −δ, g = max_{j≠t} p_j − p_t); (2) batch-relative statistics are affine-invariant → noise-blind; (3) the absolute gap rejects a closed-form, rate-free band [0.038, 0.85] (τ = 0.3: [0.26, 0.64], explains its 96% memorization); (4) **escape region**: no statistic can reject g ≥ β − δ_b (0.85), so the most contradicted labels get a near-maximal gradient toward the wrong class; the same cap explains N4's threshold (β = .25/.5/1 can reject only g < .10/.35/.85). All checked against the code in `tests/test_mechanism.py`. **AS-Softmax memorizes faster than CE** at epochs 3–7 on 5/5 seeds (peak +9.2 pts at epoch 5, t = 13), both saturate by epoch 8. Mechanism runs (`experiments/mechanism.py`, Phase C/C′ configs unchanged + per-sample logit-gradient diagnostics on the probe, 42 runs × 3 seeds) reproduce Phase C bit-exactly; results pending. Not run: β = 1.15 (closes the escape region) — outside the pre-registration, user's call |
