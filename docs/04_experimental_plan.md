# Experimental Plan

Goal: produce a results table that a reviewer at ACL / EMNLP / ICLR / NeurIPS
cannot dismiss. That means strong baselines, multiple seeds, careful
ablations, and breadth across modalities.

This is the operational core of the project. Treat it as authoritative;
[06_timeline_and_milestones.md](06_timeline_and_milestones.md) just sequences
it in time.

---

## 1. Phase structure

| Phase | Weeks | Deliverable |
|---|---|---|
| 0. Infrastructure | 1 | Repo, training loop, eval harness, all baselines coded |
| 1. Reproduce | 2–3 | AS-Softmax reproduced on SST-5, CLINC150, CIFAR-100 |
| 2. Class-pair $\delta_{t,j}$ | 4–7 | RQ1 answered, H1 tested |
| 3. Sample $\delta(x)$ | 8–9 | RQ2 answered, H2 tested |
| 4. Time $\delta(\tau)$ | 10–11 | RQ3 answered, H3 tested |
| 5. Cross-modality + combined | 12–13 | RQ4 answered, H4 tested |
| 6. Regression (stretch) | 14 | RQ5 attempted, H5 tested if time |
| 7. Theory & writing | 15–16 | Paper draft |

Decision gates at end of weeks 3, 7, 11, 13 — see
[06_timeline_and_milestones.md](06_timeline_and_milestones.md).

---

## 2. Datasets (with feasibility on RTX 3090/4090)

### 2.1 Text classification (primary)

| Dataset | $n$ | Train size | Backbone | Wall-clock / run | Why include |
|---|---|---|---|---|---|
| SST-5 | 5 | 8.5k | BERT-base | ~25 min | Fast iteration; original AS-Softmax benchmark |
| CLINC150 | 151 | 15k | BERT-base | ~45 min | Tests scaling to large $n$ |
| CoNLL-2003 | 9 | 14k sent | BERT-base | ~30 min | Sequence labeling (token-level) |
| AG News | 4 | 120k | BERT-base | ~2 h | Larger sample size sanity check |
| 20 Newsgroups | 20 | 11k | BERT-base | ~30 min | Has natural class hierarchy → good for class-pair |
| Eurlex-4K | ~4k (multi) | 11k | BERT-base | ~1 h | Multi-label; tests Eq. §6 of method |

**Cuts if time-pressured:** AG News and Eurlex. Keep SST-5, CLINC150,
CoNLL-2003, 20NG as the text core.

### 2.2 Image classification (secondary)

| Dataset | $n$ | Train size | Backbone | Wall-clock / run | Why |
|---|---|---|---|---|---|
| CIFAR-100 | 100 | 50k | ResNet-50 / ViT-S | ~2 h | Has 20-superclass hierarchy → ideal for class-pair |
| CIFAR-10 | 10 | 50k | ResNet-50 | ~1.5 h | Small-$n$ sanity check |
| WikiArt | 195 (artist) | 80k | ResNet-50 | ~4 h | Matches original AS-Softmax benchmark |

**Cuts if needed:** WikiArt; keep CIFAR-100 + CIFAR-10.

### 2.3 Audio classification (tertiary)

| Dataset | $n$ | Train size | Backbone | Wall-clock | Why |
|---|---|---|---|---|---|
| ESC-50 | 50 | 1.6k | wav2vec2-base | ~30 min | Small, fast |
| Speech Commands v2 | 35 | 85k | wav2vec2-base | ~3 h | Larger, more rigorous |

**Cuts if needed:** Speech Commands; keep ESC-50.

### 2.4 Regression (stretch)

| Dataset | Task | Train size | Backbone | Wall-clock | Why |
|---|---|---|---|---|---|
| UTKFace | Age (0–116) | 20k images | ResNet-50 | ~2 h | Standard ordinal regression benchmark |
| UCI Boston (or California) | Bounded continuous | 500–20k | MLP | <10 min | Cheap proof-of-concept |

---

## 3. Baselines

Every method below gets coded once, frozen, and reused across all datasets.
Implementation correctness validated by reproducing one paper's number per
method.

| Baseline | Reference | What it tests |
|---|---|---|
| Softmax + CE | (standard) | Vanilla floor |
| AS-Softmax | Lv et al. 2023 | Our direct comparator — must reproduce |
| AM-Softmax | Wang et al. 2018 | Angular-margin alternative |
| Sparsemax | Martins & Astudillo 2016 | Sparse alternative to softmax |
| Entmax-1.5 | Peters et al. 2019 | Tunable sparse softmax |
| Label Smoothing | Szegedy 2016 | Regularization-style alternative |
| Focal Loss | Lin 2017 | Per-sample reweighting alternative |
| Power Softmax | (proposal in `research/`) | Sparsity-from-normalization alternative |

**Hyperparameter discipline:** every method gets a small grid search on the
validation set, with the same compute budget per method. Report the best.

---

## 4. Methods we're proposing (the rows that are *us*)

| Row | Configuration |
|---|---|
| M1 | GAM-Softmax, class-pair similarity-based |
| M2 | GAM-Softmax, class-pair confusion-based |
| M3 | GAM-Softmax, class-pair low-rank ($k=8$) — **main entry** |
| M4 | GAM-Softmax, sample (entropy-based) only |
| M5 | GAM-Softmax, sample (learned head) only |
| M6 | GAM-Softmax, time (linear ramp) only |
| M7 | GAM-Softmax, time (cosine) only |
| M8 | GAM-Softmax, class-pair + sample (M3 + M5) |
| M9 | GAM-Softmax, class-pair + time (M3 + M6) |
| M10 | GAM-Softmax, full (M3 + M5 + M6) — **flagship** |

---

## 5. Metrics

For every (method × dataset × seed) cell, log all of:

### 5.1 Classification quality

- **Accuracy** (top-1)
- **Macro-F1** (matters for class-imbalanced data like Eurlex)
- **Top-5** (matters for CLINC150)

### 5.2 Calibration

- **Expected Calibration Error (ECE)** at 15 bins
- **Negative Log-Likelihood (NLL)** on test
- **Brier Score**
- **Reliability diagram** (one figure per main dataset)

### 5.3 Optimization quality

- **Wall-clock training time** to convergence
- **Steps to convergence** (val accuracy plateaus)
- **Pearson $\rho$(train loss, val accuracy)** — Lv et al.'s headline metric
- **Masked sample fraction** over time (does AS-Speed kick in?)

### 5.4 Method-specific diagnostics

- For class-pair: **rank-$k$ approximation quality** of learned $M$ (singular
  value spectrum)
- For sample: **distribution of $\delta(x)$** across val set
- For time: **$\delta(\tau)$ trajectory** vs schedule

### 5.5 Robustness (a small but informative section)

- **Label noise:** flip 10% of training labels, report test acc
- **Few-shot:** train with 10%, 25%, 50% of data
- **Imbalanced:** subsample one class to 10% of others (CIFAR-100)

Robustness experiments are cheap (subset of data) and great for the paper
discussion — usually 1 table, 2 paragraphs.

---

## 6. Statistical rigor

- **5 random seeds** per (method × dataset) cell. Report mean ± std.
- **Paired bootstrap** significance test when the method-vs-AS-Softmax gap
  matters for a claim.
- For the headline table, **bold the best**; **underline** any result not
  significantly worse than the best at $p < 0.05$.
- Hyperparameter search **only on the validation set**. Final numbers on
  test, never tune on test.

---

## 7. Ablations (the section reviewers will read most carefully)

Required ablations (in roughly decreasing priority):

1. **Effect of each direction in isolation.** M3, M5, M6 individually,
   compared to AS-Softmax. (Answers: is each axis useful on its own?)
2. **Composition study.** Multiplicative (default) vs additive composition.
3. **Rank $k$ sweep** for low-rank class-pair: $k \in \{2, 4, 8, 16, 32, n\}$.
   Where does the rank-quality plateau?
4. **$(\delta_{\min}, \delta_{\max})$ sweep** — heatmap.
5. **Schedule comparison** — linear vs cosine vs cyclical vs constant.
6. **Class-pair parameterization comparison** — similarity vs confusion vs
   low-rank.
7. **Sample-head architecture** — 1-layer vs 2-layer vs no normalization.
8. **Anchor regularization strength** $\mu$ in sample variant.
9. **Initialization study** for $u, v$ — small random vs class-embedding init.
10. **AS-Speed on/off** — does the gradient accumulation trick still work?

Each ablation is one figure or one mini-table. Most cost ~1 day each.

---

## 8. Compute budget estimate

Assuming RTX 3090/4090:

| Phase | Runs | Avg run | Total |
|---|---|---|---|
| Phase 1 (reproduce, all baselines × text core) | ~25 | 1 h | 25 h |
| Phase 2 (M1–M3 × text + CIFAR-100) | ~30 | 1.5 h | 45 h |
| Phase 3 (M4–M5 × text core) | ~12 | 1 h | 12 h |
| Phase 4 (M6–M7 × text core) | ~10 | 1 h | 10 h |
| Phase 5 (M8–M10 + cross-modality) | ~25 | 2 h | 50 h |
| Phase 6 (regression, stretch) | ~10 | 1.5 h | 15 h |
| Ablations | ~60 | 1 h | 60 h |
| Re-runs / debugging | 20% buffer | | 40 h |
| **Total** | | | **~260 h** |

At ~16 h/day GPU utilization, that's ~16 days of compute over 16 weeks of
calendar time — plenty of slack. The bottleneck is your time, not the GPU.

---

## 9. Logging and reproducibility

- **W&B project** (`gam-softmax`) — every run, every metric, every config.
- **Config files in YAML** (`configs/<method>_<dataset>.yaml`). One config
  per row in the final table.
- **Seeds:** `{42, 43, 44, 45, 46}` everywhere.
- **Pickle the confusion matrix** of every final model — needed for M2
  confusion-based margins, also useful for paper figures.
- **Save the learned $u, v$** for every M3 / M10 run — needed for T2 plots.

---

## 10. The first concrete experiment to run

Once Phase 0 is done, this is the single experiment that decides whether
the project's central hypothesis (H1) is alive:

```yaml
experiment: H1_quick_check
backbone: bert-base-uncased
dataset: SST-5
losses:
  - softmax
  - as_softmax  # δ = 0.3 (from paper)
  - gam_lowrank  # k = 8, δ_min = 0.05, δ_max = 0.4
seeds: [42, 43, 44]
epochs: 10
log: wandb
```

**Pass criterion:** `gam_lowrank` exceeds `as_softmax` by ≥ 0.3% accuracy on
SST-5 averaged over 3 seeds. If yes → proceed to full Phase 2. If no →
debug; check class-embedding initialization, gradient flow into $u, v$,
masked fraction over training.

Expected time to run: ~3 hours total (3 methods × 3 seeds × 25 min).
