"""Figures for the Phase E paper draft (paper/), as PDF (for LaTeX) and PNG (preview).

  fig1_geometry          — analytic, no runs: the margin δ as a function of the gap
                           g = max_{j≠t} p_j − p_t against the zero-loss boundary δ = −g.
                           (a) the absolute-gap margin's rejection band and escape region;
                           (b) the batch-relative M4's lowest reachable margin δ_b − β,
                           which caps the rejectable gap at β − δ_b (the β-sweep threshold).
  figC1/C2               — Phase C noise curves and pair-flip bars (make_phase_c_figures).
  fig4_dynamics          — per-epoch memorization and test accuracy at 40% symmetric noise,
                           with AS-Softmax added (it memorizes faster than CE).
  fig5_mechanism         — from runs/mechanism/: where each loss puts its gradient, by gap,
                           mislabeled vs clean samples (written only once those runs exist).

Usage:
    python experiments/make_paper_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import make_phase_c_figures as pc
from make_phase_c_figures import GRID, INK, INK2, SURFACE, direct_labels, style

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "figures"
MECH = REPO / "runs" / "mechanism"
DELTA_B = 0.15
MUTED = "#898781"

# dataviz reference palette, light mode, in slot order; slots 1–5 are the Phase C
# figures' assignment, so a method keeps its colour across every figure.
COLOR = {
    "softmax": "#2a78d6", "small_loss": "#eb6834", "gce": "#1baf7a", "m4": "#eda100",
    "m4_absgap": "#e87ba4", "m4_b0": "#008300", "absgap_t03": "#4a3aa7",
}
LABEL = {
    "softmax": "Cross-entropy", "small_loss": "Small-loss (oracle rate)", "gce": "GCE",
    "m4": "M4 (batch-relative)", "m4_absgap": "M4 absolute gap", "m4_b0": "AS-Softmax",
}


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _absgap_delta(g: np.ndarray, temp: float, beta: float = 1.0) -> np.ndarray:
    return np.clip(DELTA_B - beta * np.tanh(g / (2 * temp)), -1.0, DELTA_B)


def _band(temp: float, beta: float = 1.0) -> tuple[float, float]:
    g = np.linspace(0, 1, 200_001)
    ok = g[beta * np.tanh(g / (2 * temp)) >= DELTA_B + g]
    return float(ok.min()), float(ok.max())


def fig_geometry() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), facecolor=SURFACE, sharey=True)
    g = np.linspace(-1, 1, 2001)
    for ax in axes:
        style(ax, "Per-sample margin δ", "Gap g = max$_{j≠t}$ p$_j$ − p$_t$   (g > 0: another class beats the label)")
        ax.plot(g, -g, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        ax.axvline(0, color=GRID, linewidth=0.8, zorder=1)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1.05, 0.32)
        ax.axvspan(-1, -DELTA_B, color=MUTED, alpha=0.10, linewidth=0, zorder=0)
        ax.annotate("fitted: zero loss", (-0.58, 0.22), ha="center", fontsize=8.5, color=INK2)
        ax.annotate("dashed: δ = −g. Zero loss iff δ ≤ −g", (0.05, -0.99), ha="left", va="bottom",
                    fontsize=8, color=INK2)

    # (a) absolute gap
    ax = axes[0]
    lo, hi = _band(0.1)
    ax.axvspan(lo, hi, color=COLOR["m4_absgap"], alpha=0.13, linewidth=0, zorder=0)
    ax.plot(g, np.full_like(g, DELTA_B), color=COLOR["m4_b0"], linewidth=2, label="AS-Softmax (δ = 0.15)", zorder=3)
    ax.plot(g, _absgap_delta(g, 0.3), color=COLOR["absgap_t03"], linewidth=2,
            label="absolute gap, τ = 0.3", zorder=3)
    ax.plot(g, _absgap_delta(g, 0.1), color=COLOR["m4_absgap"], linewidth=2,
            label="absolute gap, τ = 0.1 (Phase C′)", zorder=4)
    lo3, hi3 = _band(0.3)
    for x in (lo, hi):
        ax.plot([x], [-x], marker="o", markersize=6, color=COLOR["m4_absgap"], markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=5)
    ax.annotate(f"rejected: g ∈ [{lo:.3f}, {hi:.2f}]", ((lo + hi) / 2, 0.2), ha="center",
                fontsize=8.5, color=INK)
    ax.annotate(f"escape: g > {hi:.2f}\ntrained against\nthe leading class", (0.93, -0.47), ha="center",
                fontsize=8, color=INK)
    ax.annotate(f"τ = 0.3 rejects only\ng ∈ [{lo3:.2f}, {hi3:.2f}]", (0.45, -0.12), ha="left",
                fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="center left")
    ax.set_title("(a) Absolute gap, s = σ(g / τ), β = 1", color=INK2, fontsize=9.5, loc="left")

    # (b) batch-relative M4: δ ranges over [δ_b − β, δ_b] with the batch z-score
    ax = axes[1]
    mem = {0.25: 96, 0.5: 54, 1.0: 16}      # Phase C, sym 40%, last epoch, 5 seeds
    styles = {0.25: (0, (1, 2)), 0.5: (0, (5, 2)), 1.0: "solid"}
    ax.plot(g, np.full_like(g, DELTA_B), color=COLOR["m4_b0"], linewidth=2, zorder=3)
    ax.annotate("AS-Softmax (β = 0): memorization 96%", (0.98, DELTA_B + 0.03), ha="right", fontsize=8,
                color=INK)
    for beta in (0.25, 0.5, 1.0):
        floor = DELTA_B - beta
        ax.plot(g, np.full_like(g, floor), color=COLOR["m4"], linewidth=2, linestyle=styles[beta], zorder=3)
        x = beta - DELTA_B
        ax.plot([x], [-x], marker="o", markersize=6, color=COLOR["m4"], markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=5)
        ax.annotate(f"β = {beta:g}: rejects g < {x:.2f}, memorized {mem[beta]}%",
                    (-0.97, floor - 0.025), ha="left", va="top", fontsize=8, color=INK)
    ax.set_title("(b) Batch-relative M4: the floor δ$_b$ − β caps the rejectable gap",
                 color=INK2, fontsize=9.5, loc="left")
    fig.tight_layout()
    save(fig, "fig1_geometry")


def _load_phase_c(cond: tuple[str, float]) -> dict:
    res = {}
    kind, noise = cond
    for p in (REPO / "runs" / "phase_c").glob(f"*__{kind}{noise}__seed*.json"):
        res.setdefault(p.stem.split("__")[0], []).append(json.loads(p.read_text(encoding="utf-8")))
    return res


def fig_dynamics() -> None:
    rows = _load_phase_c(("symmetric", 0.4))
    keys = ["softmax", "m4_b0", "small_loss", "gce", "m4", "m4_absgap"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    panels = [("mem_rate", "Corrupted labels predicted as given (%)"),
              ("val_accuracy", "Clean test accuracy (%)")]
    for ax, (hk, ylab) in zip(axes, panels):
        style(ax, ylab, "Epoch")
        ends = []
        for k in keys:
            recs = rows.get(k, [])
            if not recs:
                continue
            n_ep = min(len(r["history"]) for r in recs)
            ys = [mean(r["history"][e][hk] * 100 for r in recs) for e in range(n_ep)]
            ax.plot(range(1, n_ep + 1), ys, color=COLOR[k], linewidth=2, marker="o", markersize=4.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABEL[k], zorder=3)
            ends.append((n_ep, ys[-1], LABEL[k]))
        direct_labels(ax, ends)
        ax.set_xticks(range(1, 9))
        ax.set_xlim(0.7, 11.8)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    fig.suptitle("40% symmetric noise, 20 Newsgroups, BERT-base, mean of 5 seeds",
                 color=INK2, fontsize=9.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig4_dynamics")


def _mech_runs(kind: str, noise: float) -> dict:
    res = {}
    for p in MECH.glob(f"*__{kind}{noise}__seed*.json"):
        res.setdefault(p.stem.split("__")[0], []).append(json.loads(p.read_text(encoding="utf-8")))
    return res


def fig_mechanism() -> None:
    """Three per-epoch mechanism readings from the training probe (runs/mechanism, 3 seeds):
    (a) mislabeled samples rejected, (b) share of the gradient on mislabeled samples,
    (c) clean samples locked out when there is no noise at all."""
    sym = _mech_runs("symmetric", 0.4)
    clean = _mech_runs("symmetric", 0.0)
    if not sym:
        print("no runs/mechanism data yet: skipping fig5_mechanism")
        return
    panels = [
        (sym, ["small_loss", "m4", "m4_absgap"], "diag_rejected_flipped",
         "Mislabeled samples rejected (%)", "(a) 40% symmetric: rejected mislabeled samples"),
        (sym, ["softmax", "m4_b0", "gce", "m4_absgap"], "diag_grad_share_flipped",
         "Gradient share on mislabeled (%)", "(b) 40% symmetric: where the gradient goes"),
        (clean, ["m4", "m4_absgap"], "diag_rejected_clean",
         "Clean samples rejected (%)", "(c) No noise: clean samples locked out"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), facecolor=SURFACE)
    for ax, (rows, keys, hk, ylab, title) in zip(axes, panels):
        style(ax, ylab, "Epoch")
        ends = []
        for k in keys:
            recs = rows.get(k, [])
            if not recs:
                continue
            n_ep = min(len(r["history"]) for r in recs)
            ys = [mean(r["history"][e][hk] * 100 for r in recs) for e in range(n_ep)]
            ax.plot(range(1, n_ep + 1), ys, color=COLOR[k], linewidth=2, marker="o", markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=LABEL[k], zorder=3)
            ends.append((n_ep, ys[-1], LABEL[k]))
        ax.set_ylim(0, 100 if hk != "diag_rejected_clean" else 30)
        direct_labels(ax, ends, min_gap_frac=0.07)
        ax.set_xticks(range(1, 9))
        ax.set_xlim(0.7, 12.6)
        ax.set_title(title, color=INK2, fontsize=9.5, loc="left")
    n = min(len(v) for v in sym.values())
    fig.suptitle(f"Training probe, mean of {n} seeds. CE, AS-Softmax and GCE never reject; CE fits 99.7% "
                 f"of clean training labels without noise", color=INK2, fontsize=9.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig5_mechanism")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_geometry()
    pc.FIGS = OUT
    pc.EXTS = ("pdf", "png")
    res = pc.load()
    pc.fig_noise_curves(res)
    pc.fig_pair_noise(res)
    fig_dynamics()
    fig_mechanism()
    for p in sorted(OUT.glob("*.png")):
        print("wrote", p.relative_to(REPO))


if __name__ == "__main__":
    main()
