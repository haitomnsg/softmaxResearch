"""Phase E mechanism runs (docs/10_noisy_label_plan.md §3, Phase E): where each loss sends
its gradient, measured per epoch on the memorization probe (`gam_softmax.eval.mechanism`).

These are the Phase C / C′ configs, unchanged, re-run on 3 seeds with the probe's new
gradient diagnostics. No method or knob is new: the runs only instrument the paper's
mechanism claims (rejection by negative margin, AS-Softmax's gradient concentration,
M4's noise-blind clean cost, the `abs_gap` escape region above g_hi ≈ 0.85). They also
replicate Phase C's accuracies on seeds 42–44, up to CUDA non-determinism.

Usage:
    python experiments/mechanism.py              # 42 runs, ~8 h
    python experiments/mechanism.py --dry-run
    python experiments/mechanism.py --summary    # rebuild runs/mechanism/summary.md only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import phase_c

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "runs" / "mechanism"
PHASE_C_DIR = REPO / "runs" / "phase_c"   # for the replication column
SEEDS = (42, 43, 44)
# condition-major, most informative first; seed-major inside, so a partial sweep is usable
CONDITIONS = [
    ("symmetric", 0.4, ("softmax", "m4_b0", "m4", "m4_absgap", "small_loss", "gce")),
    ("symmetric", 0.0, ("softmax", "m4", "m4_absgap")),
    ("pair", 0.4, ("softmax", "small_loss", "gce", "m4", "m4_absgap")),
]
ESCAPE_BIN = 18   # gap bins [0.8, 0.9) and [0.9, 1.0]: above abs_gap's g_hi ≈ 0.85


def plan() -> list[tuple[str, float, int, str]]:
    return [(k, n, s, m) for k, n, ms in CONDITIONS for s in SEEDS for m in ms]


def _ms(xs: list[float]) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    return f"{mean(xs):.1f}" if len(xs) == 1 else f"{mean(xs):.1f} ±{stdev(xs):.1f}"


def _escape_share(h: dict) -> float | None:
    """Share of the flipped samples' gradient mass at gaps ≥ 0.8."""
    fl = h.get("diag_grad_hist_flipped")
    if not fl or sum(fl) == 0:
        return None
    return sum(fl[ESCAPE_BIN:]) / sum(fl)


def write_summary() -> str:
    lines = ["# Phase E mechanism runs — 20NG, Phase C recipe, seeds 42–44", "",
             "Driver: `experiments/mechanism.py`. Diagnostics are read on the 1 200-example training probe at the "
             "**last epoch** (`gam_softmax/eval/mechanism.py`). *grad share (noisy)*: share of the probe's gradient "
             "mass carried by mislabeled samples. *rejected*: zero gradient while another class beats the label. "
             "*zero grad (clean)*: clean samples with zero gradient, fitted or rejected. *escape*: share of the "
             "mislabeled samples' gradient mass at gaps ≥ 0.8, where `abs_gap` cannot reject (g_hi ≈ 0.85).", ""]
    for kind, noise, methods in CONDITIONS:
        lines += [f"## {kind} {noise:.0%}", "",
                  "| method | sel % | final % | mem @final % | grad share (noisy) % | rejected noisy % | "
                  "rejected clean % | zero grad clean % | escape % | Phase C sel (same seeds) | n |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for m in methods:
            recs = []
            for s in SEEDS:
                p = OUT_DIR / f"{m}__{kind}{noise}__seed{s}.json"
                if p.exists():
                    recs.append((s, json.loads(p.read_text(encoding="utf-8"))))
            if not recs:
                continue
            last = [r["history"][-1] for _, r in recs]
            pct = lambda key: [h[key] * 100 if h.get(key) is not None else None for h in last]
            ref = []
            for s, _ in recs:
                q = PHASE_C_DIR / f"{m}__{kind}{noise}__seed{s}.json"
                if q.exists():
                    ref.append(json.loads(q.read_text(encoding="utf-8"))["sel_val_acc"] * 100)
            esc = [_escape_share(h) for h in last]
            lines.append(
                f"| {m} | {_ms([r['sel_val_acc'] * 100 for _, r in recs])} | "
                f"{_ms([r['final_val_acc'] * 100 for _, r in recs])} | {_ms(pct('mem_rate'))} | "
                f"{_ms(pct('diag_grad_share_flipped'))} | {_ms(pct('diag_rejected_flipped'))} | "
                f"{_ms(pct('diag_rejected_clean'))} | {_ms(pct('diag_zero_clean'))} | "
                f"{_ms([e * 100 if e is not None else None for e in esc])} | {_ms(ref)} | {len(recs)} |")
        lines.append("")
    s = "\n".join(lines) + "\n"
    (OUT_DIR / "summary.md").write_text(s, encoding="utf-8")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.summary:
        print(write_summary())
        return
    phase_c.OUT_DIR = OUT_DIR          # phase_c.run_one / job_path write here
    jobs = plan()
    todo = [j for j in jobs if not phase_c.job_path(j[3], j[0], j[1], j[2]).exists()]
    print(f"[E] {len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run", flush=True)
    if args.dry_run:
        for j in todo:
            print("    ", j)
        return
    t0 = time.time()
    for i, (kind, noise, seed, m) in enumerate(todo, 1):
        print(f"\n[E] === job {i}/{len(todo)} === elapsed {(time.time() - t0) / 60:.0f} min", flush=True)
        phase_c.run_one(m, kind, noise, seed, smoke=False)
        write_summary()
    print("\n" + write_summary())
    print(f"[E] all done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
