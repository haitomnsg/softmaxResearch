# Implementation Roadmap

> **Superseded as the working plan (2026-10-05).** This file is the original plan, whose core hypothesis concluded negative on 2026-07-31 (see `FINAL_REPORT.md`). The active plan is [10_noisy_label_plan.md](10_noisy_label_plan.md). Kept unchanged below as history.


How the code is organized so experiments are fast to iterate and results are
trustworthy.

---

## 1. Repo layout

```
SoftMax/
├── research/                        # original notes (don't touch)
├── source/                          # PDFs (don't touch)
├── docs/                            # this plan
├── gam_softmax/                     # the package
│   ├── __init__.py
│   ├── losses/
│   │   ├── softmax.py               # baseline cross-entropy
│   │   ├── as_softmax.py            # AS-Softmax (Lv et al.)
│   │   ├── am_softmax.py            # AM-Softmax
│   │   ├── sparsemax.py
│   │   ├── entmax.py
│   │   ├── focal.py
│   │   ├── label_smoothing.py
│   │   ├── power_softmax.py         # Power Softmax baseline
│   │   └── gam_softmax.py           # OUR loss — the star of the show
│   ├── margins/
│   │   ├── base.py                  # MarginFunction abstract class
│   │   ├── scalar.py                # ScalarMargin (≡ AS-Softmax)
│   │   ├── classpair_similarity.py  # similarity-based δ_{t,j}
│   │   ├── classpair_confusion.py   # confusion-based δ_{t,j}
│   │   ├── classpair_lowrank.py     # learnable low-rank δ_{t,j}
│   │   ├── sample_entropy.py
│   │   ├── sample_featurenorm.py
│   │   ├── sample_head.py           # learnable MLP h_φ(x)
│   │   └── composition.py           # multiplicative / additive composer
│   ├── schedules/
│   │   ├── base.py                  # Schedule abstract class
│   │   ├── constant.py
│   │   ├── linear.py
│   │   ├── cosine.py
│   │   ├── cyclical.py
│   │   └── gradient_driven.py
│   ├── data/
│   │   ├── text/                    # SST-5, CLINC150, CoNLL, AGNews, 20NG, Eurlex
│   │   ├── image/                   # CIFAR-10, CIFAR-100, WikiArt
│   │   ├── audio/                   # ESC-50, SpeechCommands
│   │   └── regression/              # UTKFace, UCI tabular
│   ├── models/
│   │   ├── text_classifier.py       # BERT-base wrapper
│   │   ├── image_classifier.py      # ResNet-50 / ViT-S wrapper
│   │   ├── audio_classifier.py      # wav2vec2-base wrapper
│   │   └── regressor.py             # binned regressor
│   ├── training/
│   │   ├── trainer.py               # the only training loop
│   │   ├── as_speed.py              # gradient accumulation trick
│   │   └── callbacks.py             # logging, checkpointing
│   ├── eval/
│   │   ├── classification.py        # acc, F1, top-k
│   │   ├── calibration.py           # ECE, NLL, Brier, reliability
│   │   ├── correlation.py           # Pearson(loss, val acc)
│   │   └── diagnostics.py           # masked fraction, δ trajectory
│   └── utils/
│       ├── config.py                # YAML loader, validation
│       ├── seeding.py
│       └── logging.py               # W&B wrapper
├── configs/                         # one YAML per experiment
│   ├── baselines/
│   ├── m1_classpair_similarity/
│   ├── m3_classpair_lowrank/
│   ├── ...
│   └── m10_full/
├── experiments/                     # entrypoint scripts
│   ├── run.py                       # python experiments/run.py --config <path>
│   ├── sweep.py                     # hyperparam sweep
│   └── reproduce_as_softmax.py      # week-1 sanity check
├── notebooks/                       # analysis + figures
│   ├── 01_baseline_reproduction.ipynb
│   ├── 02_class_pair_margins.ipynb
│   ├── ...
│   └── paper_figures.ipynb
├── tests/                           # unit tests
│   ├── test_losses.py               # gradient checks against pytorch
│   ├── test_margins.py
│   └── test_schedules.py
├── paper/                           # LaTeX
│   ├── main.tex
│   ├── figures/
│   ├── tables/
│   └── refs.bib
├── README.md
├── requirements.txt
└── setup.py
```

---

## 2. Key interface contracts

### 2.1 The loss interface

Every loss takes the same signature so they're interchangeable:

```python
class GAMSoftmaxLoss(nn.Module):
    def __init__(self, margin_fn: MarginFunction, schedule: Schedule, n_classes: int):
        ...

    def forward(
        self,
        logits: torch.Tensor,   # (B, n)
        targets: torch.Tensor,  # (B,) int
        features: torch.Tensor, # (B, d) penultimate, for δ(x)
        step: int,              # current training step
    ) -> dict:
        # returns {"loss": ..., "mask": ..., "delta_eff": ..., "masked_ratio": ...}
```

The dict return is intentional — diagnostics live with the loss, so we don't
have to recompute them.

### 2.2 The margin function interface

```python
class MarginFunction(nn.Module):
    def forward(
        self,
        logits: torch.Tensor,   # (B, n)
        targets: torch.Tensor,  # (B,)
        features: torch.Tensor, # (B, d)
        step_frac: float,       # τ ∈ [0, 1]
    ) -> torch.Tensor:          # (B, n) margin per (sample, non-target class)
```

### 2.3 The composition operator

```python
def compose_margins(
    margins: list[MarginFunction],
    mode: str = "multiplicative",  # or "additive"
) -> MarginFunction:
    ...
```

This is what lets us swap M3 ↔ M8 ↔ M10 from a config file by listing margin
functions, instead of writing 10 different loss classes.

---

## 3. Config-driven experiments

Every experiment is a YAML file. Example for M3 on SST-5:

```yaml
name: m3_classpair_lowrank_sst5
seed: 42
dataset:
  name: sst5
  batch_size: 32
  max_seq_len: 128
model:
  backbone: bert-base-uncased
  head: linear
loss:
  type: gam_softmax
  margins:
    - type: classpair_lowrank
      rank: 8
      init: small_random
  schedule:
    type: linear
    delta_min: 0.05
    delta_max: 0.4
    warmup_frac: 0.3
training:
  optimizer: adamw
  lr: 2e-5
  weight_decay: 0.01
  epochs: 10
  warmup_steps_frac: 0.1
  as_speed: true
  as_speed_lambda: 0.5
logging:
  wandb_project: gam-softmax
  wandb_tags: [phase2, sst5, m3]
```

Run with `python experiments/run.py --config configs/m3_classpair_lowrank/sst5.yaml`.

**One config, one row in the final table.** This is the contract that makes
the paper writeable in 2 weeks at the end.

---

## 4. Milestones

| Milestone | Definition of done | Week |
|---|---|---|
| **MS0: skeleton compiles** | `python -c "import gam_softmax"` works, all stubs in place | 1 |
| **MS1: softmax baseline trains** | Vanilla CE on SST-5 hits ~52% accuracy | 1 |
| **MS2: AS-Softmax reproduced** | Within ±0.5% of Lv et al. on SST-5 | 2 |
| **MS3: all 8 baselines coded** | Unit tests passing, smoke-tested on SST-5 | 3 |
| **MS4: GAM-Softmax M3 runs end-to-end** | Trains without NaN, produces valid masks | 4 |
| **MS5: H1 quick-check passes (or fails honestly)** | See [04_experimental_plan.md §10](04_experimental_plan.md) | 4 |
| **MS6: full Phase 2 table done** | M1–M3 on all text datasets, 5 seeds | 7 |
| **MS7: Phase 3 table done** | M4–M5 added | 9 |
| **MS8: Phase 4 table done** | M6–M7 added | 11 |
| **MS9: Cross-modality results** | M3 / M10 on CIFAR-100, ESC-50 | 13 |
| **MS10: Paper draft v1** | All figures, all tables, intro + method + experiments | 16 |

---

## 5. Code quality bar

For a research project, "production quality" is overkill but "messy notebooks"
is dangerous (irreproducible results).

**Required:**
- Type hints on the loss/margin/schedule interfaces.
- A `pytest` suite that runs in < 30s, checks gradients numerically.
- `numpy.random.seed`, `torch.manual_seed`, `torch.use_deterministic_algorithms(True)`
  on every run.
- `requirements.txt` pinned by minor version.

**Not required:**
- Full docstrings everywhere. Method docstrings only.
- Full coverage. Test the math-heavy parts (losses, margins).
- A CLI beyond `run.py --config`. Notebooks for analysis.

---

## 6. Dependencies

```
torch>=2.1
transformers>=4.40
datasets>=2.18
torchvision>=0.16
torchaudio>=2.1
wandb>=0.16
pyyaml
scikit-learn  # for calibration metrics
matplotlib    # for paper figures
seaborn       # for prettier figures
einops        # for tensor reshaping
entmax        # for entmax baseline (pip package exists)
```

That's the lot. Don't add a dependency without a reason.

---

## 7. Pitfalls to avoid (lessons from the AS-Softmax repro)

- **The mask is on $p_t - p_i$, not $o_t - o_i$.** Compute softmax first,
  then check the condition. Computing in logit space loses the AS-Softmax
  reduction.
- **Mask is recomputed per step, not per epoch.** Easy to forget if you cache.
- **Don't backprop through the mask.** `mask = mask.detach()`. The math
  assumes hard masking; if gradient flows through the threshold, behavior
  is undefined.
- **For Power Softmax baseline, $\gamma < 1$ has unbounded gradient at $o_i = 0^+$.**
  Add a small $\epsilon$ inside the power.
- **For low-rank $u, v$, init scale matters a lot.** Start with
  `nn.init.normal_(std=0.02)`; if $\sigma(u^\top v / \sqrt k)$ starts at
  ~0.5, the margin is at $(\delta_{\min} + \delta_{\max})/2$ at init —
  which means hard masking from step 0, which can stall training. Start
  smaller; let it grow.
- **`torch.compile` breaks with dynamic masks.** Either skip it or use
  `dynamic=True`. AS-Softmax's masked sum has data-dependent shape.

---

## 8. What to do if you're stuck on Day 1

1. Don't write GAM-Softmax first. Write the baseline harness.
2. Don't write the perfect config system. Hardcode SST-5 + softmax, get a
   number on the screen, then refactor.
3. Don't optimize. Get correct first. AS-Speed and `torch.compile` come last.
4. If `pip install` fights you for half a day, use a fresh conda env:
   `conda create -n gam python=3.11 && conda activate gam && pip install -r requirements.txt`.
