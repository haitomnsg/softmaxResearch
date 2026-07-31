"""Print the E1 comparison for one (noise, seed) slice — the working view while runs land.

Usage:
    python experiments/show_block.py                 # noise 0.4, all seeds present
    python experiments/show_block.py --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs" / "final_20ng"
ORDER = ["softmax", "as_softmax", "gam_m3", "gam_m4", "gam_m4_b05", "gam_m6"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    pattern = f"*_n{args.noise}_seed{args.seed or '*'}.json"
    recs: dict[tuple[str, int], dict] = {}
    for path in RUNS.glob(pattern):
        method, rest = path.stem.rsplit("_n", 1)
        recs[(method, int(rest.split("_seed")[1]))] = json.loads(path.read_text(encoding="utf-8"))
    if not recs:
        print(f"no runs matching {pattern}")
        return

    seeds = sorted({s for _, s in recs})
    for seed in seeds:
        present = [m for m in ORDER if (m, seed) in recs]
        print(f"\n=== 20NG, {args.noise:.0%} label noise, seed {seed} ===")
        print(f"{'method':12s} {'best%':>7} {'final%':>7} {'decay':>6} {'ECE':>6} {'conf':>6}")
        for m in present:
            r = recs[(m, seed)]
            print(f"{m:12s} {r['best_val_acc']*100:>7.2f} {r['final_val_acc']*100:>7.2f} "
                  f"{(r['best_val_acc']-r['final_val_acc'])*100:>6.2f} "
                  f"{r['best_val_ece']:>6.3f} {r['history'][-1]['val_confidence']:>6.3f}")

        print(f"\n{'memorization of corrupted labels (%)':40s}{'masked@end':>12}")
        print(f"{'method':12s}" + "".join(f"{'ep'+str(i):>8}" for i in range(4)) + f"{'':>4}")
        for m in present:
            h = recs[(m, seed)]["history"]
            print(f"{m:12s}" + "".join(f"{e['mem_rate']*100:>8.1f}" for e in h)
                  + f"{h[-1]['train_masked_ratio']:>12.3f}")

        print(f"\n{'train acc on the labels it was GIVEN (fitting the noise, %)':40s}")
        for m in present:
            h = recs[(m, seed)]["history"]
            print(f"{m:12s}" + "".join(f"{e['train_acc_given']*100:>8.1f}" for e in h))


if __name__ == "__main__":
    main()
