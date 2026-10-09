"""Figures for Phases C and C′ (docs/10), built from runs/phase_c/*.json.

  figC1_noise_curves.png   — two panels over the symmetric-noise rate {0, .2, .4, .6}:
                             (left) test accuracy at the epoch picked on noisy held-out
                             labels (the realistic protocol); (right) memorization of
                             corrupted labels at the last epoch. Mean ± sd over 5 seeds.
  figC2_pair_noise.png     — the same two quantities under pair-flip noise {.2, .4},
                             as grouped bars: the structured-noise case where the
                             absolute rejection rule separates from everything else.
  figC3_masking_dynamics.png — per-epoch memorization at 40% symmetric noise for the
                             rejection methods: the mechanism, not the outcome.

Usage:
    python experiments/make_phase_c_figures.py            # writes runs/phase_c/figures/
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "phase_c"
FIGS = RUNS / "figures"

# Categorical slots in fixed order (dataviz skill reference palette, light mode,
# validated 2026-10-09: adjacent CVD ΔE ≥ 8, normal-vision ≥ 15). Color follows the
# method, never its rank; every series is also direct-labeled.
METHODS = [  # (key, label, hex)
    ("softmax",   "Cross-entropy",            "#2a78d6"),
    ("small_loss", "Small-loss (oracle rate)", "#eb6834"),
    ("gce",       "GCE",                      "#1baf7a"),
    ("m4",        "M4 (batch-relative, β=1)", "#eda100"),
    ("m4_absgap", "M4 absolute gap",          "#e87ba4"),
]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SYM = [0.0, 0.2, 0.4, 0.6]
PAIR = [0.2, 0.4]
EXTS = ("png",)   # make_paper_figures.py redirects FIGS and adds "pdf"


def save(fig, name: str) -> None:
    for ext in EXTS:
        fig.savefig(FIGS / f"{name}.{ext}", dpi=160, facecolor=SURFACE)


def load() -> dict:
    """res[(kind, noise)][method] = list of records (one per seed)."""
    res: dict = {}
    for p in RUNS.glob("*.json"):
        method, cond, _seed = p.stem.split("__")
        kind = "symmetric" if cond.startswith("symmetric") else "pair"
        noise = float(cond[len(kind):])
        res.setdefault((kind, noise), {}).setdefault(method, []).append(json.loads(p.read_text(encoding="utf-8")))
    return res


def stat(recs: list[dict], key: str, scale: float = 100.0):
    xs = [r[key] * scale for r in recs if r.get(key) is not None]
    if not xs:
        return None, None
    return mean(xs), (stdev(xs) if len(xs) > 1 else 0.0)


def style(ax, ylabel: str, xlabel: str):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)


def direct_labels(ax, items: list[tuple[float, float, str]], min_gap_frac: float = 0.055, dx_pt: float = 6):
    """Place end-of-line labels, pushing stacked ones apart so none overlap.

    ``items`` are (x, y, text). Labels keep their order by y; each is moved by at
    least ``min_gap_frac`` of the y-range from its neighbour. A short leader ties a
    moved label back to its point.
    """
    if not items:
        return
    y0, y1 = ax.get_ylim()
    gap = (y1 - y0) * min_gap_frac
    order = sorted(range(len(items)), key=lambda i: items[i][1])
    placed = []
    for i in order:
        y = items[i][1]
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
    # if the stack ran past the top, shift the whole block down
    over = placed[-1] - (y1 - gap * 0.5)
    if over > 0:
        placed = [p - over for p in placed]
    for i, yl in zip(order, placed):
        x, y, text = items[i]
        ax.annotate(text, (x, yl), xytext=(dx_pt, 0), textcoords="offset points", va="center",
                    ha="left", fontsize=8.5, color=INK, annotation_clip=False)
        if abs(yl - y) > 1e-9:
            ax.plot([x, x], [y, yl], color=GRID, linewidth=0.8, zorder=2)


def fig_noise_curves(res: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    panels = [("sel_val_acc", "Test accuracy, epoch picked on noisy held-out labels (%)"),
              ("final_mem_rate", "Corrupted labels predicted as given, last epoch (%)")]
    for ax, (key, ylab) in zip(axes, panels):
        style(ax, ylab, "Symmetric label-noise rate")
        ends = []
        for mkey, label, color in METHODS:
            xs, ys, es = [], [], []
            for n in SYM:
                recs = res.get(("symmetric", n), {}).get(mkey, [])
                m, s = stat(recs, key)
                if m is None:
                    continue
                xs.append(n); ys.append(m); es.append(s)
            if not xs:
                continue
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
            ax.fill_between(xs, [y - e for y, e in zip(ys, es)], [y + e for y, e in zip(ys, es)],
                            color=color, alpha=0.12, linewidth=0)
            ends.append((xs[-1], ys[-1], label))
        direct_labels(ax, ends)
        ax.set_xticks(SYM)
        ax.set_xticklabels([f"{int(n * 100)}%" for n in SYM])
        ax.set_xlim(-0.02, 0.86)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")
    fig.suptitle("20 Newsgroups, BERT-base, 8 epochs, 5 seeds (mean ± sd). Right: no corrupted labels at 0%.",
                 color=INK2, fontsize=9.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "figC1_noise_curves")
    plt.close(fig)


def fig_pair_noise(res: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    panels = [("sel_val_acc", "Test accuracy, epoch picked on noisy held-out labels (%)"),
              ("final_mem_rate", "Corrupted labels predicted as given, last epoch (%)")]
    width = 0.15
    for ax, (key, ylab) in zip(axes, panels):
        style(ax, ylab, "Pair-flip label-noise rate")
        for i, (mkey, label, color) in enumerate(METHODS):
            xs, ys, es = [], [], []
            for j, n in enumerate(PAIR):
                recs = res.get(("pair", n), {}).get(mkey, [])
                m, s = stat(recs, key)
                if m is None:
                    continue
                xs.append(j + (i - 2) * (width + 0.02)); ys.append(m); es.append(s)
            if not xs:
                continue
            bars = ax.bar(xs, ys, width=width, color=color, label=label, zorder=3,
                          yerr=es, error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
            for b, y in zip(bars, ys):
                ax.annotate(f"{y:.1f}", (b.get_x() + b.get_width() / 2, y), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(PAIR)))
        ax.set_xticklabels([f"{int(n * 100)}%" for n in PAIR])
        lo = min(b.get_height() for b in ax.patches)
        ax.set_ylim(max(0, lo - 12), None)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")
    fig.suptitle("Pair-flip noise (each corrupted label goes to one fixed wrong class). 5 seeds, mean ± sd.",
                 color=INK2, fontsize=9.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "figC2_pair_noise")
    plt.close(fig)


def fig_masking_dynamics(res: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor=SURFACE)
    style(ax, "Corrupted labels predicted as given (%)", "Epoch")
    rows = res.get(("symmetric", 0.4), {})
    ends = []
    for mkey, label, color in METHODS:
        recs = rows.get(mkey, [])
        if not recs:
            continue
        n_ep = min(len(r["history"]) for r in recs)
        ys = [mean(r["history"][e]["mem_rate"] * 100 for r in recs) for e in range(n_ep)]
        ax.plot(range(1, n_ep + 1), ys, color=color, linewidth=2, marker="o", markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        ends.append((n_ep, ys[-1], label))
    direct_labels(ax, ends)
    ax.set_xticks(range(1, 9))
    ax.set_xlim(0.7, 11.5)
    ax.set_title("40% symmetric noise: how fast each method fits the wrong labels",
                 color=INK2, fontsize=9.5, loc="left")
    fig.tight_layout()
    save(fig, "figC3_masking_dynamics")
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    res = load()
    fig_noise_curves(res)
    fig_pair_noise(res)
    fig_masking_dynamics(res)
    for p in sorted(FIGS.glob("figC*.png")):
        print("wrote", p.relative_to(REPO))


if __name__ == "__main__":
    main()
