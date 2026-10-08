"""Single entry point for every experiment.

Usage:
    python experiments/run.py --config configs/baselines/softmax_sst5.yaml
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import torch

# allow `python experiments/run.py` from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gam_softmax.losses import (
    AMSoftmaxLoss,
    ASSoftmaxLoss,
    Entmax15Loss,
    FocalLoss,
    GAMSoftmaxLoss,
    GCELoss,
    LabelSmoothingLoss,
    PowerSoftmaxLoss,
    SCELoss,
    SmallLossLoss,
    SoftmaxLoss,
    SparsemaxLoss,
)
from gam_softmax.data.probe import build_probe
from gam_softmax.margins import ClassPairLowRankMargin, SampleConfidenceMargin
from gam_softmax.models import TextClassifier
from gam_softmax.schedules import ConstantSchedule, LinearSchedule
from gam_softmax.training import Trainer
from gam_softmax.utils import load_config, seed_everything


LOSS_REGISTRY = {
    "softmax": SoftmaxLoss,
    "as_softmax": ASSoftmaxLoss,
    "am_softmax": AMSoftmaxLoss,
    "sparsemax": SparsemaxLoss,
    "entmax15": Entmax15Loss,
    "focal": FocalLoss,
    "label_smoothing": LabelSmoothingLoss,
    "power_softmax": PowerSoftmaxLoss,
    "gam_softmax": GAMSoftmaxLoss,
    "gce": GCELoss,
    "sce": SCELoss,
    "small_loss": SmallLossLoss,
}

SCHEDULE_REGISTRY = {
    "constant": ConstantSchedule,
    "linear": LinearSchedule,
}

MARGIN_REGISTRY = {
    "classpair_lowrank": ClassPairLowRankMargin,
    "sample_confidence": SampleConfidenceMargin,
}


def build_schedule(cfg: dict):
    cfg = dict(cfg)
    cls = SCHEDULE_REGISTRY[cfg.pop("type")]
    return cls(**cfg)


def build_margin(cfg: dict, n_classes: int):
    cfg = dict(cfg)
    cls = MARGIN_REGISTRY[cfg.pop("type")]
    schedule = build_schedule(cfg.pop("schedule"))
    # a nested `base:` block composes margins (e.g. sample axis over class-pair axis)
    base_cfg = cfg.pop("base", None)
    if base_cfg is not None:
        cfg["base_margin"] = build_margin(base_cfg, n_classes)
    return cls(n_classes=n_classes, schedule=schedule, **cfg)


def build_data(cfg: dict, tokenizer):
    name = cfg["dataset"]["name"]
    if name == "sst5":
        from gam_softmax.data.text import load_sst5

        return load_sst5(
            tokenizer=tokenizer,
            batch_size=cfg["dataset"]["batch_size"],
            max_seq_len=cfg["dataset"]["max_seq_len"],
            num_workers=cfg["dataset"].get("num_workers", 0),
            limit_train=cfg["dataset"].get("limit_train"),
            label_noise=cfg["dataset"].get("label_noise", 0.0),
            noise_seed=cfg["dataset"].get("noise_seed", 0),
            eval_batch_size=cfg["dataset"].get("eval_batch_size"),
        )
    if name == "20newsgroups":
        from gam_softmax.data.text import load_20newsgroups

        return load_20newsgroups(
            tokenizer=tokenizer,
            batch_size=cfg["dataset"]["batch_size"],
            max_seq_len=cfg["dataset"]["max_seq_len"],
            num_workers=cfg["dataset"].get("num_workers", 0),
            limit_train=cfg["dataset"].get("limit_train"),
            label_noise=cfg["dataset"].get("label_noise", 0.0),
            noise_seed=cfg["dataset"].get("noise_seed", 0),
            eval_batch_size=cfg["dataset"].get("eval_batch_size"),
            noisy_val_frac=cfg["dataset"].get("noisy_val_frac", 0.0),
            noise_type=cfg["dataset"].get("noise_type", "symmetric"),
        )
    raise ValueError(f"Unknown dataset: {name}")


def build_loss(cfg: dict, n_classes: int, feature_dim: int) -> torch.nn.Module:
    loss_cfg = dict(cfg["loss"])
    cls = LOSS_REGISTRY[loss_cfg.pop("type")]
    # small-loss's oracle setting: forget exactly the true noise rate of this run
    if loss_cfg.get("forget_rate") == "noise_rate":
        loss_cfg["forget_rate"] = float(cfg.get("dataset", {}).get("label_noise", 0.0))
    # GAM-Softmax composes a margin_fn from a sub-config; build it first.
    if cls is GAMSoftmaxLoss:
        margin_cfg = loss_cfg.pop("margin")
        margin_fn = build_margin(margin_cfg, n_classes=n_classes)
        return cls(n_classes=n_classes, margin_fn=margin_fn, **loss_cfg)
    # pass feature_dim only to losses that declare it (e.g. AM-Softmax owns its own W)
    if "feature_dim" in inspect.signature(cls.__init__).parameters:
        loss_cfg.setdefault("feature_dim", feature_dim)
    return cls(n_classes=n_classes, **loss_cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps for smoke tests")
    ap.add_argument("--seed", type=int, default=None, help="Override cfg.seed (used by H1 multi-seed runs)")
    ap.add_argument("--label-noise", type=float, default=None,
                    help="Override dataset.label_noise (fraction of train labels to corrupt; used by the noise sweep)")
    ap.add_argument("--noise-type", default=None, choices=["symmetric", "pair"],
                    help="Override dataset.noise_type")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-json", default=None,
                    help="Write the full result record (history, calibration, memorization) here")
    ap.add_argument("--probe-size", type=int, default=1200,
                    help="Training examples held in the memorization probe; 0 disables it")
    args = ap.parse_args()
    t_start = time.time()

    cfg = load_config(args.config)
    if args.label_noise is not None:
        cfg.setdefault("dataset", {})["label_noise"] = args.label_noise
    if args.noise_type is not None:
        cfg.setdefault("dataset", {})["noise_type"] = args.noise_type
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    seed_everything(seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    noise = cfg.get("dataset", {}).get("label_noise", 0.0)
    print(f"[run] config={args.config}  device={device}  label_noise={noise}")

    # The memory-efficient SDP attention kernel has a non-deterministic backward
    # that intermittently faults mid-step on 6 GB laptop GPUs, killing the process
    # with no Python error. Disable it so PyTorch uses the stabler flash kernel.
    if device == "cuda":
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    tokenizer = TextClassifier.tokenizer_for(cfg["model"]["backbone"])
    data = build_data(cfg, tokenizer)
    n_classes = data["n_classes"]

    model = TextClassifier(
        backbone_name=cfg["model"]["backbone"],
        n_classes=n_classes,
        dropout=cfg["model"].get("dropout", 0.1),
    )
    feature_dim = model.encoder.config.hidden_size
    loss_fn = build_loss(cfg, n_classes=n_classes, feature_dim=feature_dim)

    optim_cfg = cfg["training"]
    wd = optim_cfg.get("weight_decay", 0.01)
    loss_params = list(loss_fn.parameters())
    margin_lr = optim_cfg.get("margin_lr")
    if margin_lr is not None and loss_params:
        # give the loss/margin params their own (typically higher) LR and no weight
        # decay, so structural params like M3's u/v can actually move at BERT's tiny LR
        optimizer = torch.optim.AdamW([
            {"params": list(model.parameters()), "lr": optim_cfg["lr"], "weight_decay": wd},
            {"params": loss_params, "lr": float(margin_lr), "weight_decay": 0.0},
        ])
    else:
        # include loss params so AM-Softmax / future learnable-margin losses get optimized
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + loss_params,
            lr=optim_cfg["lr"],
            weight_decay=wd,
        )

    # Held-in slice of the training set: separates "learned the task" from
    # "memorized the corrupted labels". Same probe seed for every method so the
    # memorization numbers are comparable.
    probe = build_probe(data, size=args.probe_size, seed=1234) if args.probe_size > 0 else None

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=data["train"],
        val_loader=data["val"],
        optimizer=optimizer,
        device=device,
        max_epochs=optim_cfg["epochs"],
        max_steps=args.max_steps if args.max_steps is not None else optim_cfg.get("max_steps"),
        log_every=optim_cfg.get("log_every", 25),
        grad_clip=optim_cfg.get("grad_clip", 1.0),
        probe=probe,
        noisy_val_loader=data.get("noisy_val"),
    )
    state = trainer.fit()
    print(f"[run] done. seed={seed} best_val_acc={state.best_val_acc:.4f}")

    if args.out_json:
        best = max(state.history, key=lambda h: h.get("val_accuracy", -1.0)) if state.history else {}
        # realistic protocol: pick the epoch on held-out NOISY labels (earliest on ties),
        # report its clean test accuracy
        nv_hist = [h for h in state.history if "noisyval_accuracy" in h]
        sel = max(nv_hist, key=lambda h: h["noisyval_accuracy"]) if nv_hist else {}
        record = {
            "config": args.config,
            "name": cfg.get("name", Path(args.config).stem),
            "dataset": cfg.get("dataset", {}).get("name"),
            "loss_type": cfg["loss"]["type"],
            "seed": seed,
            "label_noise": noise,
            "noise_type": cfg.get("dataset", {}).get("noise_type", "symmetric"),
            "epochs": optim_cfg["epochs"],
            "best_val_acc": state.best_val_acc,
            "best_epoch": state.best_epoch,
            "final_val_acc": state.final_val_acc,
            "sel_val_acc": sel.get("val_accuracy"),
            "sel_epoch": sel.get("epoch"),
            "sel_mem_rate": sel.get("mem_rate"),
            # calibration/memorization are read at the *best* epoch so they
            # describe the model you would actually keep
            "best_val_ece": best.get("val_ece"),
            "best_mem_rate": best.get("mem_rate"),
            "best_recover_rate": best.get("recover_rate"),
            "final_val_ece": state.history[-1].get("val_ece") if state.history else None,
            "final_mem_rate": state.history[-1].get("mem_rate") if state.history else None,
            "wall_time_s": round(time.time() - t_start, 1),
            "history": state.history,
        }
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"[run] wrote {out_path}")


if __name__ == "__main__":
    main()
