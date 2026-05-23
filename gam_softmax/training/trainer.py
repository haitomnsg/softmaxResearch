from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from gam_softmax.eval.classification import classification_metrics


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_val_acc: float = 0.0
    history: list[dict] = field(default_factory=list)


class Trainer:
    """The single training loop every experiment runs.

    Loss objects must follow the dict-return contract (`{"loss": ..., ...}`).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        max_epochs: int = 10,
        max_steps: Optional[int] = None,
        log_every: int = 25,
        eval_every_steps: Optional[int] = None,
        grad_clip: Optional[float] = 1.0,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device) if isinstance(loss_fn, nn.Module) else loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.log_every = log_every
        self.eval_every_steps = eval_every_steps
        self.grad_clip = grad_clip
        self.state = TrainState()

        steps_per_epoch = len(train_loader)
        self.total_steps = max_steps if max_steps is not None else steps_per_epoch * max_epochs

    def _move(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            batch.input_ids.to(self.device, non_blocking=True),
            batch.attention_mask.to(self.device, non_blocking=True),
            batch.labels.to(self.device, non_blocking=True),
        )

    def fit(self) -> TrainState:
        for epoch in range(self.max_epochs):
            self.state.epoch = epoch
            self._train_one_epoch()
            val = self.evaluate(self.val_loader)
            self.state.history.append({"epoch": epoch, "step": self.state.step, **{"val_" + k: v for k, v in val.items()}})
            if val["accuracy"] > self.state.best_val_acc:
                self.state.best_val_acc = val["accuracy"]
            print(f"[epoch {epoch}] step={self.state.step} val_acc={val['accuracy']:.4f} best={self.state.best_val_acc:.4f}")
            if self.max_steps is not None and self.state.step >= self.max_steps:
                break
        return self.state

    def _train_one_epoch(self) -> None:
        self.model.train()
        t0 = time.time()
        running = 0.0
        seen = 0
        for batch in self.train_loader:
            input_ids, attention_mask, labels = self._move(batch)
            out = self.model(input_ids, attention_mask)
            step_frac = self.state.step / max(1, self.total_steps)
            res = self.loss_fn(out.logits, labels, features=out.features, step_frac=step_frac)
            loss = res["loss"]

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            self.state.step += 1

            if self.state.step % self.log_every == 0:
                avg = running / max(1, seen)
                dt = time.time() - t0
                print(f"  step {self.state.step}/{self.total_steps}  loss={avg:.4f}  step_frac={step_frac:.3f}  ({dt:.1f}s)")
                running = 0.0
                seen = 0
                t0 = time.time()

            if self.eval_every_steps and self.state.step % self.eval_every_steps == 0:
                val = self.evaluate(self.val_loader)
                self.state.history.append({"step": self.state.step, **{"val_" + k: v for k, v in val.items()}})
                if val["accuracy"] > self.state.best_val_acc:
                    self.state.best_val_acc = val["accuracy"]
                self.model.train()

            if self.max_steps is not None and self.state.step >= self.max_steps:
                return

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_correct = 0
        total_n = 0
        total_loss = 0.0
        step_frac = self.state.step / max(1, self.total_steps)
        for batch in loader:
            input_ids, attention_mask, labels = self._move(batch)
            out = self.model(input_ids, attention_mask)
            res = self.loss_fn(out.logits, labels, features=out.features, step_frac=step_frac)
            m = classification_metrics(out.logits, labels)
            total_correct += m["n_correct"]
            total_n += m["n"]
            total_loss += res["loss"].item() * m["n"]
        return {
            "accuracy": total_correct / max(1, total_n),
            "loss": total_loss / max(1, total_n),
            "n": total_n,
        }
