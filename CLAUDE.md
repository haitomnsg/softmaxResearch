# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A research codebase for **GAM-Softmax** (Generalized Adaptive Margin Softmax) — extends AS-Softmax (Lv et al. 2023) by replacing its single scalar margin δ with a margin function δ_{t,j}(x, t) parameterized along class-pair, sample, and time axes. Target venue: ACL / EMNLP / ICLR / NeurIPS. The original research plan lives in [docs/](docs/) files 00–09 (navigate via the top-level `README.md`); its core hypothesis concluded negative on 2026-07-31 ([FINAL_REPORT.md](FINAL_REPORT.md)). **The active plan since 2026-10-05 is [docs/10_noisy_label_plan.md](docs/10_noisy_label_plan.md)**: the M4 negative-margin idea as a noisy-label method, with gates and kill criteria. Update its §6 status table as phases land.

## Environment

Python is Miniconda-only on this machine; there is no `python` on `PATH`. Use the project env:

```bash
# env name: gam (Python 3.11, torch 2.6.0+cu124, transformers 5.x)
"/c/Users/MSi/miniconda3/envs/gam/python.exe"     # bash / git-bash
C:\Users\MSi\miniconda3\envs\gam\python.exe        # cmd / powershell
# or activate it:
conda activate gam
```

**Windows Smart App Control blocks unsigned DLLs** ("An Application Control policy has blocked this file"). conda-forge and many pip wheels ship unsigned `.pyd` files; Anaconda's `defaults` channel ships signed ones. So install **compiled packages via `conda install` from `defaults`** (numpy, scipy, pandas, scikit-learn, matplotlib, pyarrow, tokenizers, `python-xxhash`, yarl, ...) and use pip only for torch (cu124 index) and pure-Python packages (transformers, datasets, entmax, einops, wandb). torchaudio is not installed: its DLLs are blocked and nothing imports it.

Hardware: **RTX 3080 Ti (12 GB VRAM)**, i7-12700K (12C/20T), **16 GB system RAM**, desktop that can run 24/7. VRAM is still below the 24 GB RTX 3090/4090 the [docs/](docs/) assume: BERT-base with `batch_size=16, max_seq_len=128` uses ~2.5 GB, so longer sequences / larger batches / BERT-large-with-checkpointing are now possible. System RAM is the tighter limit for big datasets and many dataloader workers. Earlier results (through 2026-07-31) were produced on a 6 GB RTX 3050 laptop; the `runs/` artifacts did not migrate (gitignored).

## Commands

```bash
# install / re-sync deps into the env
"/c/Users/LOQ/.conda/envs/gam/python.exe" -m pip install -r requirements.txt
# (torch is installed separately from PyPI's CUDA index — see pyproject.toml note)

# run an experiment (the only entrypoint)
python experiments/run.py --config configs/baselines/softmax_sst5.yaml
python experiments/run.py --config configs/baselines/as_softmax_sst5.yaml

# smoke override (skips full epochs, useful for pipeline checks)
python experiments/run.py --config <yaml> --max-steps 50

# tests
python -m pytest tests/ -v
python -m pytest tests/test_losses.py::test_as_softmax_mask_matches_threshold -v   # single test
```

## Architecture (the parts that need reading multiple files to understand)

**One config = one experiment = one row in the final paper table.** Every experiment runs as `python experiments/run.py --config <yaml>`. Configs live in [configs/](configs/), grouped by phase (baselines, m1_*, m2_*, ...). This contract is what makes the paper writeable at the end — never run experiments via ad-hoc scripts or notebooks. Notebooks are reserved for post-hoc analysis and paper figures only.

**The loss dict-return contract** is the keystone abstraction. Every loss — vanilla `SoftmaxLoss`, `ASSoftmaxLoss`, future GAM variants — implements:

```python
forward(logits, targets, features=None, step_frac=0.0) -> {
    "loss":         Tensor scalar,
    "mask":         BoolTensor (B, n) | None,   # which (sample, class) slots are kept
    "delta_eff":    Tensor scalar  | None,      # effective δ this step (after schedule)
    "masked_ratio": Tensor scalar,              # fraction of non-target slots masked
}
```

Diagnostics live with the loss (not recomputed by the trainer) so swapping losses doesn't break logging. Plain CE fills mask/delta_eff with `None` and masked_ratio with 0.

**Why `step_frac` (normalized progress in [0, 1]) and not raw `step`?** Schedules (warmup, anneal, cosine) need a horizon-independent input. The trainer computes `step_frac = state.step / total_steps` and passes it through; losses never see raw steps.

**The margin/schedule abstractions** (under [gam_softmax/margins/](gam_softmax/margins/) and [gam_softmax/schedules/](gam_softmax/schedules/)) are how GAM variants compose. Adding M1–M10 from the plan means writing one `MarginFunction` subclass + maybe one `Schedule`, then a YAML — not a new loss class per variant.

## Critical correctness pitfalls (from [docs/05_implementation_roadmap.md §7](docs/05_implementation_roadmap.md))

- **AS-Softmax mask is on `p_t - p_i`, not `o_t - o_i`.** Run softmax first, then check the threshold. Computing on logits loses the AS-Softmax reduction property. [gam_softmax/losses/as_softmax.py](gam_softmax/losses/as_softmax.py) does this correctly.
- **Mask must be detached** (`with torch.no_grad()` around mask computation). The math assumes hard masking; gradient flow through the threshold gives undefined behavior.
- **Mask is recomputed per step, not per epoch.** Never cache.
- **Memory-efficient attention is non-deterministic on CUDA backward.** Seeding gives near-reproducibility, not bit-exact. Acceptable until paper-grade results; then `torch.use_deterministic_algorithms(True, warn_only=False)` + workaround.

## Adding a new baseline or GAM variant

1. Implement `nn.Module` subclass in [gam_softmax/losses/](gam_softmax/losses/) following the dict-return contract.
2. Add to `LOSS_REGISTRY` in [experiments/run.py](experiments/run.py).
3. Add a config YAML under `configs/<phase>/<name>.yaml` (copy an existing one as template).
4. Add unit tests in [tests/test_losses.py](tests/test_losses.py) covering: math-heavy edge cases, dict contract, gradient finiteness, hyperparameter validation. The pytest suite must stay under ~30s.

## Status tracker

The [README.md](README.md) §6 status table and §7 decision log are kept up to date as work lands — update them when finishing a milestone. Milestones are defined in [docs/05_implementation_roadmap.md §4](docs/05_implementation_roadmap.md).

## Source material

- [research/](research/) — original structured markdown notes on AS-Softmax, sparse-softmax variants, and the generalized-margin formulation. Don't modify these; they're the project's intellectual provenance.
- [source/](source/) — PDFs of the underlying papers. Same — read-only.
