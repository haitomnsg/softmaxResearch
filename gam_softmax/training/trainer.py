from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from gam_softmax.eval.classification import (
    classification_metrics,
    expected_calibration_error,
)
from gam_softmax.eval.mechanism import label_gap, per_sample_grad_l1, summarize


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_val_acc: float = 0.0
    best_epoch: int = -1
    final_val_acc: float = 0.0
    history: list[dict] = field(default_factory=list)


class Trainer:
    """The single training loop every experiment runs.

    Loss objects must follow the dict-return contract (`{"loss": ..., ...}`).

    Optional ``probe`` argument
    ---------------------------
    A held-in sample of the *training* set used to measure memorization rather
    than generalization. Under label noise, val accuracy alone can't tell you
    *why* a method wins — you need to see whether it fitted the corrupted
    labels. ``probe`` is a dict::

        {"loader": DataLoader,          # shuffle=False, labels = the (noisy) training labels
         "clean_labels": LongTensor,    # ground truth, aligned to loader order
         "flipped": BoolTensor}         # True where the training label was corrupted

    Producing ``train_acc_given`` / ``train_acc_clean`` and, over the flipped
    subset only, ``mem_rate`` (predicted the corrupted label — memorization) and
    ``recover_rate`` (predicted the true label despite being trained on a wrong
    one).
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
        probe: Optional[dict] = None,
        noisy_val_loader: Optional[DataLoader] = None,
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
        self.probe = probe
        self.noisy_val_loader = noisy_val_loader
        self.state = TrainState()

        steps_per_epoch = len(train_loader)
        self.total_steps = max_steps if max_steps is not None else steps_per_epoch * max_epochs

    def _move(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            batch.input_ids.to(self.device, non_blocking=True),
            batch.attention_mask.to(self.device, non_blocking=True),
            batch.labels.to(self.device, non_blocking=True),
        )

    def _sample_idx(self, batch) -> Optional[torch.Tensor]:
        """Dataset positions of the batch, for stateful per-sample losses; None if
        the loader doesn't carry them or the loss doesn't ask for them."""
        if not getattr(self.loss_fn, "wants_sample_idx", False):
            return None
        idx = getattr(batch, "idx", None)
        return idx.to(self.device, non_blocking=True) if idx is not None else None

    def fit(self) -> TrainState:
        for epoch in range(self.max_epochs):
            self.state.epoch = epoch
            train_stats = self._train_one_epoch()
            val = self.evaluate(self.val_loader)
            entry = {
                "epoch": epoch,
                "step": self.state.step,
                **train_stats,
                **{"val_" + k: v for k, v in val.items()},
            }
            if self.noisy_val_loader is not None:
                # held-out *noisy* labels: the realistic model-selection signal
                nv = self.evaluate(self.noisy_val_loader)
                entry["noisyval_accuracy"] = nv["accuracy"]
            if self.probe is not None:
                entry.update(self.run_probe())
            self.state.history.append(entry)
            self.state.final_val_acc = val["accuracy"]
            if val["accuracy"] > self.state.best_val_acc:
                self.state.best_val_acc = val["accuracy"]
                self.state.best_epoch = epoch
            extra = ""
            if "mem_rate" in entry and entry["mem_rate"] is not None:
                extra = f" mem_rate={entry['mem_rate']:.4f}"
            print(
                f"[epoch {epoch}] step={self.state.step} val_acc={val['accuracy']:.4f} "
                f"val_ece={val['ece']:.4f} best={self.state.best_val_acc:.4f}{extra}"
            )
            if self.max_steps is not None and self.state.step >= self.max_steps:
                break
        return self.state

    def _train_one_epoch(self) -> dict:
        self.model.train()
        t0 = time.time()
        running = 0.0
        seen = 0
        # epoch-level averages of the loss's own diagnostics (dict-return contract)
        masked_ratio_sum = 0.0
        delta_eff_sum = 0.0
        delta_eff_n = 0
        est_noise_sum = 0.0
        est_noise_n = 0
        n_batches = 0
        for batch in self.train_loader:
            input_ids, attention_mask, labels = self._move(batch)
            out = self.model(input_ids, attention_mask)
            step_frac = self.state.step / max(1, self.total_steps)
            sample_idx = self._sample_idx(batch)
            if sample_idx is not None:
                # training step of a stateful per-sample loss: let it update its
                # running statistics. Eval calls (below) never pass sample_idx.
                res = self.loss_fn(out.logits, labels, features=out.features,
                                   step_frac=step_frac, sample_idx=sample_idx)
            else:
                res = self.loss_fn(out.logits, labels, features=out.features, step_frac=step_frac)
            loss = res["loss"]

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            n_batches += 1
            masked_ratio_sum += float(res.get("masked_ratio", 0.0))
            if res.get("delta_eff") is not None:
                delta_eff_sum += float(res["delta_eff"])
                delta_eff_n += 1
            if res.get("est_noise_rate") is not None:
                est_noise_sum += float(res["est_noise_rate"])
                est_noise_n += 1
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
                    self.state.best_epoch = self.state.epoch
                self.model.train()

            if self.max_steps is not None and self.state.step >= self.max_steps:
                break
        return {
            "train_masked_ratio": masked_ratio_sum / max(1, n_batches),
            "train_delta_eff": (delta_eff_sum / delta_eff_n) if delta_eff_n else None,
            # mean over the epoch of the loss's own noise-rate estimate, if it has one
            "train_est_noise_rate": (est_noise_sum / est_noise_n) if est_noise_n else None,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_correct = 0
        total_n = 0
        total_loss = 0.0
        confs: list[torch.Tensor] = []
        corrects: list[torch.Tensor] = []
        step_frac = self.state.step / max(1, self.total_steps)
        for batch in loader:
            input_ids, attention_mask, labels = self._move(batch)
            out = self.model(input_ids, attention_mask)
            res = self.loss_fn(out.logits, labels, features=out.features, step_frac=step_frac)
            m = classification_metrics(out.logits, labels)
            total_correct += m["n_correct"]
            total_n += m["n"]
            total_loss += res["loss"].item() * m["n"]
            probs = torch.softmax(out.logits, dim=-1)
            conf, preds = probs.max(dim=-1)
            confs.append(conf.cpu())
            corrects.append((preds == labels).float().cpu())
        conf_all = torch.cat(confs) if confs else torch.empty(0)
        correct_all = torch.cat(corrects) if corrects else torch.empty(0)
        return {
            "accuracy": total_correct / max(1, total_n),
            "loss": total_loss / max(1, total_n),
            "ece": expected_calibration_error(conf_all, correct_all),
            "confidence": float(conf_all.mean()) if conf_all.numel() else float("nan"),
            "n": total_n,
        }

    @torch.no_grad()
    def run_probe(self) -> dict:
        """Memorization diagnostics over a fixed sample of the training set.

        Also records where the loss sends its gradient (``gam_softmax.eval.mechanism``):
        one autograd call on the detached probe logits per batch, so no model backward
        and no RNG use; the training trajectory is unchanged.
        """
        self.model.eval()
        preds: list[torch.Tensor] = []
        given: list[torch.Tensor] = []
        gaps: list[torch.Tensor] = []
        grads: list[torch.Tensor] = []
        step_frac = self.state.step / max(1, self.total_steps)
        for batch in self.probe["loader"]:
            input_ids, attention_mask, labels = self._move(batch)
            out = self.model(input_ids, attention_mask)
            preds.append(out.logits.argmax(dim=-1).cpu())
            given.append(labels.cpu())
            gaps.append(label_gap(out.logits, labels).cpu())
            grads.append(per_sample_grad_l1(self.loss_fn, out.logits, labels, step_frac,
                                            features=out.features).cpu())
        if not preds:
            return {}
        preds_t = torch.cat(preds)
        given_t = torch.cat(given)
        clean_t = self.probe["clean_labels"][: preds_t.numel()]
        flipped = self.probe["flipped"][: preds_t.numel()]

        out = {
            "train_acc_given": float((preds_t == given_t).float().mean()),
            "train_acc_clean": float((preds_t == clean_t).float().mean()),
            "mem_rate": None,
            "recover_rate": None,
            "n_probe": int(preds_t.numel()),
            "n_probe_flipped": int(flipped.sum()),
        }
        if bool(flipped.any()):
            out["mem_rate"] = float((preds_t[flipped] == given_t[flipped]).float().mean())
            out["recover_rate"] = float((preds_t[flipped] == clean_t[flipped]).float().mean())
        out.update(summarize(torch.cat(gaps), torch.cat(grads), flipped))
        self.model.train()
        return out
