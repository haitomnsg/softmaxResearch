"""Single entry point for every experiment.

Usage:
    python experiments/run.py --config configs/baselines/softmax_sst5.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# allow `python experiments/run.py` from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gam_softmax.losses import ASSoftmaxLoss, SoftmaxLoss
from gam_softmax.models import TextClassifier
from gam_softmax.training import Trainer
from gam_softmax.utils import load_config, seed_everything


LOSS_REGISTRY = {
    "softmax": SoftmaxLoss,
    "as_softmax": ASSoftmaxLoss,
}


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


def build_loss(cfg: dict, n_classes: int) -> torch.nn.Module:
    loss_cfg = cfg["loss"]
    cls = LOSS_REGISTRY[loss_cfg["type"]]
    kwargs = {k: v for k, v in loss_cfg.items() if k != "type"}
    return cls(n_classes=n_classes, **kwargs)


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
    loss_fn = build_loss(cfg, n_classes=n_classes)

    optim_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
