"""Figures for the final report, built from the E1 per-run JSON records.

Two panels, because the two things worth showing are a trajectory and a
mechanism:

  fig1_accuracy_trajectory.png — validation accuracy per epoch. Under 40% label
      noise the interesting shape is the *rise then fall*: a method that resists
      memorization should decay more slowly after its peak.
  fig2_memorization.png — of the training examples whose labels were corrupted,
      the fraction the model now predicts AS the corrupted label. This is the
      quantity every margin-masking argument is really about.

Usage:
    python experiments/make_figures.py                 # reads runs/final_20ng/
    python experiments/make_figures.py --noise 0.0
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "final_20ng"
FIGS = REPO / "runs" / "final_20ng" / "figures"

LABELS = {
    "softmax": "Softmax (plain CE)",
    "as_softmax": "AS-Softmax (scalar δ)",
    "gam_m3": "GAM M3 (class-pair)",
    "gam_m4": "GAM M4 (sample)",
    "gam_m4_b05": "GAM M4 (sample, β=0.5)",
    "gam_m6": "GAM M6 (combined)",
}
ORDER = ["softmax", "as_softmax", "gam_m3", "gam_m4", "gam_m4_b05", "gam_m6"]
COLORS = {
    "softmax": "#666666",
    "as_softmax": "#1f77b4",
    "gam_m3": "#2ca02c",
    "gam_m4": "#d62728",
    "gam_m4_b05": "#ff9896",
    "gam_m6": "#9467bd",
}


def load(noise: float) -> dict[str, list[dict]]:
    """method -> list of per-seed records at this noise level."""
    out: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(RUNS.glob(f"*_n{noise}_seed*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        method = path.stem.rsplit("_n", 1)[0]
        out[method].append(rec)
    return out


def _mean_curve(recs: list[dict], key: str) -> tuple[list[int], list[float]]:
    """Average a per-epoch series over seeds, keeping only epochs every seed reached."""
    per_epoch: dict[int, list[float]] = defaultdict(list)
    for rec in recs:
        for h in rec["history"]:
            if h.get(key) is not None and "epoch" in h:
                per_epoch[h["epoch"]].append(float(h[key]))
    epochs = sorted(e for e in per_epoch if len(per_epoch[e]) == len(recs))
    return epochs, [sum(per_epoch[e]) / len(per_epoch[e]) for e in epochs]


def plot_series(data: dict, key: str, title: str, ylabel: str,
                outfile: Path, noise: float, pct: bool = True) -> bool:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    plotted = False
    for method in ORDER:
        if method not in data:
            continue
        epochs, vals = _mean_curve(data[method], key)
        if not epochs:
            continue
        scale = 100.0 if pct else 1.0
        ax.plot(epochs, [v * scale for v in vals], marker="o", linewidth=2,
                label=f"{LABELS[method]}  (n={len(data[method])})",
                color=COLORS.get(method))
        plotted = True
    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n20 Newsgroups, {noise:.0%} symmetric label noise, BERT-base")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_xticks(sorted({e for m in data for e in _mean_curve(data[m], key)[0]}))
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"wrote {outfile}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.4)
    args = ap.parse_args()

    data = load(args.noise)
    if not data:
        print(f"no runs found for noise={args.noise} in {RUNS}")
        return
    tag = f"n{args.noise}"

    plot_series(data, "val_accuracy",
                "Validation accuracy (clean labels)",
                "val accuracy (%)",
                FIGS / f"fig1_accuracy_{tag}.png", args.noise)
    plot_series(data, "mem_rate",
                "Memorization of corrupted labels\n(lower = more resistant)",
                "corrupted examples predicted as their wrong label (%)",
                FIGS / f"fig2_memorization_{tag}.png", args.noise)
    plot_series(data, "val_ece",
                "Calibration error (lower = better)",
                "val ECE",
                FIGS / f"fig3_ece_{tag}.png", args.noise, pct=False)
    plot_series(data, "train_masked_ratio",
                "Fraction of non-target class slots masked out",
                "masked ratio",
                FIGS / f"fig4_masked_{tag}.png", args.noise, pct=False)


if __name__ == "__main__":
    main()
