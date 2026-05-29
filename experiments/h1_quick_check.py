"""H1 quick-check (MS5): softmax vs as_softmax vs gam_m3 on SST-5, 3 seeds each.

The single experiment that decides whether the project's central hypothesis
(class-pair margin structure helps over scalar AS-Softmax) is alive — see
docs/04_experimental_plan.md §10.

Pass criterion: gam_m3 exceeds as_softmax by ≥ 0.3% accuracy on SST-5,
averaged over 3 seeds.

Usage:
    python experiments/h1_quick_check.py
    python experiments/h1_quick_check.py --seeds 42 43 44 --epochs 5

Results are streamed live to runs/h1_quick_check/<method>_seed<N>.log and the
final mean ± std table is written to runs/h1_quick_check/summary.md.
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

METHODS = {
    "softmax":    "configs/baselines/softmax_sst5.yaml",
    "as_softmax": "configs/baselines/as_softmax_sst5.yaml",
    "gam_m3":     "configs/m3_classpair_lowrank/sst5.yaml",
}

BEST_RE = re.compile(r"best_val_acc=([0-9.]+)")


def run_one(method: str, config: str, seed: int, out_dir: Path) -> float:
    log_path = out_dir / f"{method}_seed{seed}.log"
    cmd = [
        sys.executable,
        "experiments/run.py",
        "--config", config,
        "--seed", str(seed),
    ]
    t0 = time.time()
    print(f"[h1] starting {method} seed={seed} -> {log_path.name}", flush=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# cmd: {' '.join(cmd)}\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"[h1] FAILED {method} seed={seed} after {dt:.0f}s — see {log_path}")
        return float("nan")
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = BEST_RE.findall(text)
    if not matches:
        print(f"[h1] WARN no best_val_acc in {log_path}")
        return float("nan")
    acc = float(matches[-1])
    print(f"[h1] done    {method} seed={seed}  acc={acc:.4f}  ({dt/60:.1f} min)", flush=True)
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out-dir", default="runs/h1_quick_check")
    args = ap.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[int, float]] = {m: {} for m in METHODS}
    for method, config in METHODS.items():
        for seed in args.seeds:
            results[method][seed] = run_one(method, config, seed, out_dir)

    (out_dir / "results.json").write_text(
        json.dumps({m: {str(s): v for s, v in d.items()} for m, d in results.items()}, indent=2),
        encoding="utf-8",
    )

    # summary table
    lines = ["# H1 quick-check (MS5)", "",
             f"Seeds: {args.seeds}  | dataset: SST-5  | backbone: bert-base-uncased",
             "",
             "| method | " + " | ".join(f"seed {s}" for s in args.seeds) + " | mean | std |",
             "|---|" + "---|" * (len(args.seeds) + 2)]
    finite = {}
    for method in METHODS:
        accs = [results[method][s] for s in args.seeds]
        finite_accs = [a for a in accs if a == a]  # filter NaN
        finite[method] = finite_accs
        mu = mean(finite_accs) if finite_accs else float("nan")
        sd = stdev(finite_accs) if len(finite_accs) > 1 else 0.0
        row = "| " + method + " | " + " | ".join(f"{a:.4f}" for a in accs) + f" | {mu:.4f} | {sd:.4f} |"
        lines.append(row)
    lines.append("")

    if finite.get("gam_m3") and finite.get("as_softmax"):
        gap = mean(finite["gam_m3"]) - mean(finite["as_softmax"])
        verdict = "PASS" if gap >= 0.003 else "FAIL"
        lines.append(f"## H1 verdict: **{verdict}**")
        lines.append(f"gam_m3 − as_softmax = **{gap:+.4f}**  "
                     f"(criterion: ≥ +0.003 = +0.3% acc)")

    summary = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
