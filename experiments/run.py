"""Single entry point for every experiment.

Usage:
    python experiments/run.py --config configs/baselines/softmax_sst5.yaml
"""

from __future__ import annotations

import argparse
import inspect
import sys
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
    LabelSmoothingLoss,
    PowerSoftmaxLoss,
    SoftmaxLoss,
    SparsemaxLoss,
)
from gam_softmax.margins import ClassPairLowRankMargin
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
}

SCHEDULE_REGISTRY = {
    "constant": ConstantSchedule,
    "linear": LinearSchedule,
}

MARGIN_REGISTRY = {
    "classpair_lowrank": ClassPairLowRankMargin,
}


def build_schedule(cfg: dict):
    cfg = dict(cfg)
    cls = SCHEDULE_REGISTRY[cfg.pop("type")]
    return cls(**cfg)


def build_margin(cfg: dict, n_classes: int):
    cfg = dict(cfg)
    cls = MARGIN_REGISTRY[cfg.pop("type")]
    schedule = build_schedule(cfg.pop("schedule"))
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
        )
    raise ValueError(f"Unknown dataset: {name}")


def build_loss(cfg: dict, n_classes: int, feature_dim: int) -> torch.nn.Module:
    loss_cfg = dict(cfg["loss"])
    cls = LOSS_REGISTRY[loss_cfg.pop("type")]
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
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] config={args.config}  device={device}")

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
    # include loss params so AM-Softmax / future learnable-margin losses get optimized
    trainable_params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=optim_cfg["lr"],
        weight_decay=optim_cfg.get("weight_decay", 0.01),
    )

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
    )
    state = trainer.fit()
    print(f"[run] done. best_val_acc={state.best_val_acc:.4f}")


if __name__ == "__main__":
    main()
