"""Noisy-label sweep: softmax vs as_softmax vs gam_m3 across label-noise levels.

The harder-testbed experiment. The project paused because AS-Softmax did not
reliably beat plain cross-entropy on clean, balanced text. Symmetric label noise
is the textbook regime where masking losses *should* pull ahead: AS-Softmax stops
pushing once the margin is met, so it cannot memorize corrupted labels the way
plain CE does.

Two questions this answers:
  Q1 (foundation): does AS-Softmax beat plain softmax as noise rises?
  Q2 (our method):  does GAM-M3 beat AS-Softmax in the noisy regime?

If Q1 turns YES at higher noise, the project is back on its feet and H1 can be
re-run here. If even Q1 stays NO, that is itself a clear signal to pivot.

Usage:
    python experiments/noise_sweep.py                       # SST-5, noise {0.0,0.2,0.4}, seeds 42 43
    python experiments/noise_sweep.py --dataset 20ng        # 20 Newsgroups (slower, more headroom)
    python experiments/noise_sweep.py --noise 0.0 0.2 0.4 0.6 --seeds 42 43 44
    python experiments/noise_sweep.py --smoke               # 50-step pipeline check, no real training

Live logs: runs/noise_sweep_<dataset>/<method>_n<noise>_seed<N>.log
Summary table: runs/noise_sweep_<dataset>/summary.md
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parents[1]

CONFIGS = {
    "sst5": {
        "softmax":    "configs/baselines/softmax_sst5.yaml",
        "as_softmax": "configs/baselines/as_softmax_sst5.yaml",
        "gam_m3":     "configs/m3_classpair_lowrank/sst5_learnable.yaml",
    },
    "20ng": {
        "softmax":    "configs/baselines/softmax_20ng.yaml",
        "as_softmax": "configs/baselines/as_softmax_20ng.yaml",
        "gam_m3":     "configs/m3_classpair_lowrank/20ng_learnable.yaml",
    },
}

BEST_RE = re.compile(r"best_val_acc=([0-9.]+)")


def run_one(method: str, config: str, noise: float, seed: int,
            out_dir: Path, smoke: bool) -> float:
    log_path = out_dir / f"{method}_n{noise}_seed{seed}.log"
    cmd = [
        sys.executable, "experiments/run.py",
        "--config", config,
        "--seed", str(seed),
        "--label-noise", str(noise),
    ]
    if smoke:
        cmd += ["--max-steps", "50"]
    t0 = time.time()
    print(f"[sweep] start {method} noise={noise} seed={seed} -> {log_path.name}", flush=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# cmd: {' '.join(cmd)}\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"[sweep] FAILED {method} noise={noise} seed={seed} after {dt:.0f}s — see {log_path}")
        return float("nan")
    matches = BEST_RE.findall(log_path.read_text(encoding="utf-8", errors="ignore"))
    if not matches:
        print(f"[sweep] WARN no best_val_acc in {log_path}")
        return float("nan")
    acc = float(matches[-1])
    print(f"[sweep] done  {method} noise={noise} seed={seed}  acc={acc:.4f}  ({dt/60:.1f} min)", flush=True)
    return acc


def _mu(vals: list[float]) -> float:
    finite = [v for v in vals if v == v]
    return mean(finite) if finite else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["sst5", "20ng"], default="sst5")
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.2, 0.4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="50-step runs to validate the pipeline without full training")
    args = ap.parse_args()

    # The summary table uses Unicode minus/em-dash; Windows' default cp1252 console
    # can't encode them and crashes the final print *after* all runs finished. Force
    # UTF-8 stdout so the (already-saved) results also print cleanly.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    methods = CONFIGS[args.dataset]
    out_dir = REPO / (args.out_dir or f"runs/noise_sweep_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # results[method][noise][seed] = acc
    results: dict = {m: {n: {} for n in args.noise} for m in methods}
    for noise in args.noise:
        for method, config in methods.items():
            for seed in args.seeds:
                results[method][noise][seed] = run_one(
                    method, config, noise, seed, out_dir, args.smoke
                )

    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    # One table per method (rows = noise levels), then the two verdict columns.
    lines = [f"# Noisy-label sweep — {args.dataset}", "",
             f"Seeds: {args.seeds}  | noise levels: {args.noise}  | backbone: bert-base-uncased",
             f"{'(SMOKE — 50 steps, not real training)' if args.smoke else ''}",
             "",
             "Mean val acc over seeds at each noise level:", "",
             "| noise | softmax | as_softmax | gam_m3 | AS−CE | GAM−AS |",
             "|---|---|---|---|---|---|"]
    for noise in args.noise:
        ce = _mu([results["softmax"][noise][s] for s in args.seeds])
        asx = _mu([results["as_softmax"][noise][s] for s in args.seeds])
        gam = _mu([results["gam_m3"][noise][s] for s in args.seeds])
        lines.append(
            f"| {noise} | {ce:.4f} | {asx:.4f} | {gam:.4f} | "
            f"{asx - ce:+.4f} | {gam - asx:+.4f} |"
        )
    lines += ["",
              "**AS−CE** > 0 means the foundation holds (AS-Softmax beats plain CE) at that "
              "noise level. **GAM−AS** > 0 means our class-pair margin helps on top.",
              "Look for both gaps turning positive as noise rises — that is the hard-regime "
              "win the project needs."]
    summary = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
