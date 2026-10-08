"""E1 — the project's final experiment: robustness under symmetric label noise.

Why this one and not another sweep
----------------------------------
Three earlier testbeds (clean SST-5, clean 20NG, noisy SST-5) all came back flat:
AS-Softmax never convincingly beat plain cross-entropy, so GAM-Softmax had no
floor to build on. The one signal that repeated across all of them was that GAM
was *more stable* (lower seed variance, slower overfitting) — a robustness
signal, not an accuracy one.

E1 gives that thread its decisive test on the hardest testbed we have (20
Newsgroups, 20 classes, real headroom) at 40% symmetric label noise, and adds
the axis the regime actually motivates: the per-sample margin (M4), which can
drop below zero and eject a suspect sample from the loss entirely.

Metrics beyond accuracy — because accuracy alone was what stayed flat before:
  * val ECE        — calibration; CE keeps pushing p_t → 1 even on wrong labels
  * mem_rate       — of the training examples whose labels were corrupted, the
                     fraction where the model now predicts the corrupted label.
                     This is memorization, measured directly.
  * recover_rate   — same subset, fraction where it predicts the TRUE label
                     despite having been trained on a wrong one.
  * best vs final  — the overfitting trajectory.

Job ordering is deliberate: the most decisive block runs first, so partial
results are still a usable answer if the machine is needed. Every run writes its
own JSON and the summary is rebuilt after each one, so this is safe to
interrupt — re-running skips whatever already finished.

Usage:
    python experiments/final_experiment.py                # full plan (~6 h)
    python experiments/final_experiment.py --only-block 1 # just the decisive block
    python experiments/final_experiment.py --smoke        # 30-step pipeline check
    python experiments/final_experiment.py --dry-run      # print the plan, run nothing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "runs" / "final_20ng"

CONFIGS = {
    "softmax":     "configs/final/20ng_softmax.yaml",
    "as_softmax":  "configs/final/20ng_as_softmax.yaml",
    "gam_m3":      "configs/final/20ng_gam_m3.yaml",
    "gam_m4":      "configs/final/20ng_gam_m4.yaml",
    "gam_m4_b05":  "configs/final/20ng_gam_m4_beta05.yaml",
    "gam_m6":      "configs/final/20ng_gam_m6.yaml",
    # standard noisy-label baselines (added 2026-10-05)
    "gce":         "configs/final/20ng_gce.yaml",
    "sce":         "configs/final/20ng_sce.yaml",
    "small_loss":  "configs/final/20ng_small_loss.yaml",
    # gate-B fair re-test (2026-10-05)
    "gam_m4_matched": "configs/final/20ng_gam_m4_matched.yaml",
    "gam_m4_ce":      "configs/final/20ng_gam_m4_ce.yaml",
}

MAIN = ["softmax", "as_softmax", "gam_m3", "gam_m4"]

# (block, method, noise, seed) — ordered by how much each run buys us
PLAN: list[tuple[int, str, float, int]] = (
    [(1, m, 0.4, 42) for m in MAIN]                      # decisive read
    + [(2, m, 0.4, 43) for m in MAIN]                    # is it seed noise?
    + [(3, m, 0.4, s) for s in (42, 43) for m in ("gam_m4_b05", "gam_m6")]   # ablations
    + [(4, m, 0.0, 42) for m in MAIN]                    # clean reference
)


def job_path(method: str, noise: float, seed: int) -> Path:
    return OUT_DIR / f"{method}_n{noise}_seed{seed}.json"


def run_one(method: str, noise: float, seed: int, smoke: bool) -> dict | None:
    out_json = job_path(method, noise, seed)
    log_path = out_json.with_suffix(".log")
    cmd = [
        sys.executable, "-u", "experiments/run.py",   # -u: live epoch lines in the .log
        "--config", CONFIGS[method],
        "--seed", str(seed),
        "--label-noise", str(noise),
        "--out-json", str(out_json),
    ]
    if smoke:
        cmd += ["--max-steps", "30", "--probe-size", "200"]

    t0 = time.time()
    print(f"[E1] start {method} noise={noise} seed={seed}", flush=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# cmd: {' '.join(cmd)}\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT)
    dt = (time.time() - t0) / 60

    if proc.returncode != 0 or not out_json.exists():
        print(f"[E1] FAILED {method} noise={noise} seed={seed} after {dt:.1f} min — see {log_path}",
              flush=True)
        return None
    rec = json.loads(out_json.read_text(encoding="utf-8"))
    print(f"[E1] done  {method} noise={noise} seed={seed}  "
          f"best={rec['best_val_acc']:.4f} final={rec['final_val_acc']:.4f} "
          f"ece={rec['best_val_ece']:.4f} mem={rec['best_mem_rate']}  ({dt:.1f} min)",
          flush=True)
    return rec


def load_all() -> dict:
    """results[method][noise][seed] = record — whatever is on disk right now."""
    results: dict = {}
    for path in sorted(OUT_DIR.glob("*.json")):
        if path.name == "summary.json":
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        stem = path.stem                        # <method>_n<noise>_seed<seed>
        method, rest = stem.rsplit("_n", 1)
        noise, seed = rest.split("_seed")
        results.setdefault(method, {}).setdefault(float(noise), {})[int(seed)] = rec
    return results


def _agg(records: list[dict], key: str) -> tuple[float, float, int]:
    vals = [r[key] for r in records if r.get(key) is not None]
    if not vals:
        return float("nan"), float("nan"), 0
    return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def _fmt(mu: float, sd: float, n: int, pct: bool = True) -> str:
    if n == 0:
        return "—"
    scale, unit = (100.0, "") if pct else (1.0, "")
    if n == 1:
        return f"{mu * scale:.2f}{unit}"
    return f"{mu * scale:.2f} ±{sd * scale:.2f}"


def write_summary() -> str:
    results = load_all()
    noises = sorted({n for m in results.values() for n in m})
    lines: list[str] = [
        "# E1 — Final experiment: 20 Newsgroups under symmetric label noise", "",
        "BERT-base, 4 epochs, batch 16, lr 2e-5. Validation labels are always clean;",
        "only training labels are corrupted. Noise seed is fixed per level, so every",
        "method sees the *same* corrupted labels. `±` is std over seeds.", "",
        "Regenerate: `python experiments/final_experiment.py` (skips finished runs)", "",
    ]

    for noise in noises:
        rows = [(m, list(results[m][noise].values()))
                for m in CONFIGS if m in results and noise in results.get(m, {})]
        if not rows:
            continue
        lines += [
            f"## Label noise: {noise:.0%}", "",
            "| method | best val acc % | final val acc % | decay | ECE @best | mem @best % | mem @final % | recover @best % | seeds |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for method, recs in rows:
            best_mu, best_sd, n = _agg(recs, "best_val_acc")
            fin_mu, fin_sd, _ = _agg(recs, "final_val_acc")
            ece_mu, ece_sd, _ = _agg(recs, "best_val_ece")
            memb_mu, memb_sd, n_memb = _agg(recs, "best_mem_rate")
            memf_mu, memf_sd, n_memf = _agg(recs, "final_mem_rate")
            rec_mu, rec_sd, n_rec = _agg(recs, "best_recover_rate")
            drop = (best_mu - fin_mu) * 100 if n else float("nan")
            lines.append(
                f"| {method} | {_fmt(best_mu, best_sd, n)} | {_fmt(fin_mu, fin_sd, n)} | "
                f"−{drop:.2f} | {_fmt(ece_mu, ece_sd, n, pct=False)} | "
                f"{_fmt(memb_mu, memb_sd, n_memb)} | {_fmt(memf_mu, memf_sd, n_memf)} | "
                f"{_fmt(rec_mu, rec_sd, n_rec)} | {n} |"
            )
        lines += [
            "",
            "**decay** = accuracy lost between the peak epoch and the last one (overfitting). "
            "**mem** = of the training examples whose labels were corrupted, the fraction the "
            "model predicts AS their corrupted label — direct memorization, at the best epoch "
            "and at the end. **recover** = same examples, fraction predicted as their TRUE label.",
        ]

        # the two comparisons the project actually turns on
        def mu(method: str, key: str) -> float:
            if method not in results or noise not in results.get(method, {}):
                return float("nan")
            return _agg(list(results[method][noise].values()), key)[0]

        ce, asx = mu("softmax", "best_val_acc"), mu("as_softmax", "best_val_acc")
        lines += [
            f"- **AS − CE** = {(asx - ce) * 100:+.2f} pts "
            "(does the method we extend beat plain cross-entropy here?)",
        ]
        for variant, label in (("gam_m3", "M3 class-pair"), ("gam_m4", "M4 sample"),
                               ("gam_m6", "M6 combined"), ("gam_m4_b05", "M4 β=0.5")):
            g = mu(variant, "best_val_acc")
            if g == g:
                lines.append(f"- **{label} − AS** = {(g - asx) * 100:+.2f} pts | "
                             f"**− CE** = {(g - ce) * 100:+.2f} pts")
        lines.append("")

    summary = "\n".join(lines) + "\n"
    (OUT_DIR / "summary.md").write_text(summary, encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-block", type=int, nargs="+", default=None,
                    help="Run only these plan blocks (1=decisive, 2=2nd seed, 3=ablations, 4=clean)")
    ap.add_argument("--smoke", action="store_true", help="30-step runs; validates the plan, not the science")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Re-run jobs that already have results")
    # custom grid instead of the fixed PLAN — e.g. a multi-seed confirmation sweep
    ap.add_argument("--methods", nargs="+", default=None, choices=list(CONFIGS),
                    help="With --seeds: run this method grid instead of PLAN")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.4])
    args = ap.parse_args()
    if (args.methods is None) != (args.seeds is None):
        ap.error("--methods and --seeds go together")

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.seeds is not None:
        # seed-major, so every finished seed is a complete method comparison
        plan = [(0, m, n, s) for n in args.noise for s in args.seeds for m in args.methods]
    else:
        plan = [j for j in PLAN if args.only_block is None or j[0] in args.only_block]
    todo = [j for j in plan if args.force or not job_path(j[1], j[2], j[3]).exists()]

    print(f"[E1] {len(plan)} jobs in plan, {len(plan) - len(todo)} already done, {len(todo)} to run")
    for block, method, noise, seed in todo:
        print(f"      block {block}: {method} noise={noise} seed={seed}")
    if args.dry_run:
        return

    t0 = time.time()
    for i, (block, method, noise, seed) in enumerate(todo, 1):
        print(f"\n[E1] === job {i}/{len(todo)} (block {block}) === "
              f"elapsed {(time.time() - t0) / 60:.0f} min", flush=True)
        run_one(method, noise, seed, args.smoke)
        write_summary()          # keep the table current after every single run

    print("\n" + write_summary())
    print(f"[E1] all done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
