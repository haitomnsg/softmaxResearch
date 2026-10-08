"""Phase C of docs/10_noisy_label_plan.md: close the E1 limitations on 20NG.

Every config in configs/phase_c/ trains 8 epochs and holds out 10% of the
training set WITH its noisy labels. Each run therefore reports three accuracies:
  best  - epoch picked on the clean test split (an oracle; upper bound only)
  sel   - epoch picked on the held-out NOISY split (the realistic protocol)
  final - last epoch

Tiers are ordered by how much each buys, and seed-major inside a tier, so an
interrupted sweep still leaves complete method comparisons. Resumable: finished
runs are skipped, and the summary is rebuilt after every run.

Usage:
    python experiments/phase_c.py               # everything (~30 h)
    python experiments/phase_c.py --tier 1      # main condition + beta sweep
    python experiments/phase_c.py --dry-run
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
OUT_DIR = REPO / "runs" / "phase_c"
SEEDS = (42, 43, 44, 45, 46)
CORE = ("softmax", "small_loss", "gce", "m4")
MAIN = ("softmax", "small_loss", "gce", "sce", "m4", "m4_b0", "m4_b025", "m4_b05")

# (tier, noise_type, noise, methods)
TIERS = [
    (1, "symmetric", 0.4, MAIN),       # headline + beta sweep (N2, N3, N4)
    (2, "pair", 0.4, CORE),            # asymmetric noise: the hard case
    (2, "pair", 0.2, CORE),
    (3, "symmetric", 0.6, CORE),       # noise-level curve
    (3, "symmetric", 0.2, CORE),
    (3, "symmetric", 0.0, CORE),       # clean: what does robustness cost?
]


def plan() -> list[tuple[int, str, float, int, str]]:
    return [(t, kind, n, s, m) for t, kind, n, methods in TIERS for s in SEEDS for m in methods]


def job_path(method: str, kind: str, noise: float, seed: int) -> Path:
    return OUT_DIR / f"{method}__{kind}{noise}__seed{seed}.json"


def run_one(method: str, kind: str, noise: float, seed: int, smoke: bool) -> None:
    out_json = job_path(method, kind, noise, seed)
    cmd = [sys.executable, "-u", "experiments/run.py",
           "--config", f"configs/phase_c/{method}.yaml",
           "--seed", str(seed), "--label-noise", str(noise), "--noise-type", kind,
           "--out-json", str(out_json)]
    if smoke:
        cmd += ["--max-steps", "30", "--probe-size", "100"]
    t0 = time.time()
    with out_json.with_suffix(".log").open("w", encoding="utf-8") as f:
        f.write(f"# cmd: {' '.join(cmd)}\n\n"); f.flush()
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT)
    dt = (time.time() - t0) / 60
    if proc.returncode != 0 or not out_json.exists():
        print(f"[C] FAILED {method} {kind}{noise} seed={seed} after {dt:.1f} min", flush=True)
        return
    r = json.loads(out_json.read_text(encoding="utf-8"))
    print(f"[C] done  {method} {kind}{noise} seed={seed} best={r['best_val_acc']:.4f} "
          f"sel={r['sel_val_acc']:.4f} final={r['final_val_acc']:.4f} ({dt:.1f} min)", flush=True)


def load_all() -> dict:
    """res[(kind, noise)][method][seed] = record"""
    res: dict = {}
    for p in OUT_DIR.glob("*.json"):
        method, cond, seed = p.stem.split("__")
        kind = "symmetric" if cond.startswith("symmetric") else "pair"
        noise = float(cond[len(kind):])
        res.setdefault((kind, noise), {}).setdefault(method, {})[int(seed[4:])] = \
            json.loads(p.read_text(encoding="utf-8"))
    return res


def _ms(xs: list[float]) -> str:
    if not xs:
        return "—"
    return f"{mean(xs):.2f}" if len(xs) == 1 else f"{mean(xs):.2f} ±{stdev(xs):.2f}"


def _paired(a: dict, b: dict, key: str) -> str:
    """mean of a − b over shared seeds, with paired t."""
    seeds = sorted(set(a) & set(b))
    d = [(a[s][key] - b[s][key]) * 100 for s in seeds if a[s].get(key) is not None and b[s].get(key) is not None]
    if len(d) < 2:
        return "—"
    sd = stdev(d)
    t = mean(d) / (sd / len(d) ** 0.5) if sd > 0 else float("inf")
    return f"{mean(d):+.2f} (t={t:.1f}, n={len(d)})"


def write_summary() -> str:
    res = load_all()
    lines = ["# Phase C — 20NG, 8 epochs, model selection on held-out noisy labels", "",
             "Plan: `docs/10_noisy_label_plan.md` §3. **sel** (epoch chosen on the held-out NOISY 10% "
             "of train) is the realistic number; **best** (chosen on clean test) is an oracle upper bound. "
             "`m4_b0` is AS-Softmax exactly. Regenerate: `python experiments/phase_c.py`.", ""]
    order = [(k, n) for _, k, n, _ in TIERS]
    for cond in order:
        if cond not in res:
            continue
        rows = res[cond]
        lines += [f"## {cond[0]} noise {cond[1]:.0%}", "",
                  "| method | best (oracle) % | **sel (noisy val)** % | final % | mem @final % | sel − small_loss | sel − CE | seeds |",
                  "|---|---|---|---|---|---|---|---|"]
        for m in MAIN:
            if m not in rows:
                continue
            r = rows[m]
            g = lambda k: [v[k] * 100 for v in r.values() if v.get(k) is not None]
            lines.append(
                f"| {m} | {_ms(g('best_val_acc'))} | **{_ms(g('sel_val_acc'))}** | {_ms(g('final_val_acc'))} | "
                f"{_ms(g('final_mem_rate'))} | "
                f"{_paired(r, rows['small_loss'], 'sel_val_acc') if 'small_loss' in rows and m != 'small_loss' else '—'} | "
                f"{_paired(r, rows['softmax'], 'sel_val_acc') if 'softmax' in rows and m != 'softmax' else '—'} | {len(r)} |")
        lines.append("")
    s = "\n".join(lines) + "\n"
    (OUT_DIR / "summary.md").write_text(s, encoding="utf-8")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, nargs="+", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    global OUT_DIR
    if args.smoke:
        OUT_DIR = REPO / "runs" / "phase_c_smoke"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in plan() if args.tier is None or j[0] in args.tier]
    todo = [j for j in jobs if not job_path(j[4], j[1], j[2], j[3]).exists()]
    print(f"[C] {len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run", flush=True)
    if args.dry_run:
        for j in todo:
            print("    ", j)
        return
    t0 = time.time()
    for i, (tier, kind, noise, seed, m) in enumerate(todo, 1):
        print(f"\n[C] === job {i}/{len(todo)} (tier {tier}) === elapsed {(time.time() - t0) / 60:.0f} min", flush=True)
        run_one(m, kind, noise, seed, args.smoke)
        write_summary()
    print("\n" + write_summary())
    print(f"[C] all done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
