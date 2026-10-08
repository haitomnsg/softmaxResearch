from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


NEWSGROUPS20_NUM_CLASSES = 20


@dataclass
class TextBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    # position of each example in its Dataset. Stateful per-sample losses
    # (e.g. the `loss_mixture` margin) key their running statistics on it.
    idx: Optional[torch.Tensor] = None


class _ListTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_seq_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "idx": torch.tensor(int(idx), dtype=torch.long),
        }


def _collate(batch: list[dict]) -> TextBatch:
    return TextBatch(
        input_ids=torch.stack([b["input_ids"] for b in batch]),
        attention_mask=torch.stack([b["attention_mask"] for b in batch]),
        labels=torch.stack([b["labels"] for b in batch]),
        idx=torch.stack([b["idx"] for b in batch]),
    )


def _fetch(subset: str):
    """20 Newsgroups with headers/footers/quotes removed — without this the task
    is trivially solvable from metadata (BERT ~99%) and no loss can separate."""
    from sklearn.datasets import fetch_20newsgroups

    d = fetch_20newsgroups(
        subset=subset,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )
    texts, labels = [], []
    for t, y in zip(d.data, d.target):
        if t and t.strip():            # drop docs that are empty after stripping
            texts.append(t)
            labels.append(int(y))
    return texts, labels


def load_20newsgroups(
    tokenizer,
    batch_size: int = 16,
    max_seq_len: int = 128,
    num_workers: int = 0,
    limit_train: Optional[int] = None,
    label_noise: float = 0.0,
    noise_seed: int = 0,
    eval_batch_size: Optional[int] = None,
    noisy_val_frac: float = 0.0,
    noise_type: str = "symmetric",
) -> dict:
    """Load 20 Newsgroups (sklearn) and return train/val DataLoaders.

    20NG ships only train/test splits; we use the official test split as the
    validation set (no separate val exists). 20 classes with genuinely varying
    pairwise difficulty — chosen as a testbed where methods can actually separate.

    Set ``label_noise`` > 0 to corrupt that fraction of *training* labels
    (symmetric noise; val stays clean) — the hard regime where masking losses
    are expected to beat plain cross-entropy.

    Set ``noisy_val_frac`` > 0 to hold out that fraction of the training set,
    **with its noisy labels**, as a ``noisy_val`` loader for model selection.
    That is the realistic protocol: under label noise there is no clean val set,
    so picking the epoch on the clean test split is an oracle. The split is
    seeded by ``noise_seed`` so every method holds out the same examples.
    """
    tr_texts, tr_labels = _fetch("train")
    va_texts, va_labels = _fetch("test")
    if limit_train is not None:
        tr_texts = tr_texts[:limit_train]
        tr_labels = tr_labels[:limit_train]
    clean_labels = list(tr_labels)
    if label_noise and label_noise > 0.0:
        from gam_softmax.data.label_noise import inject_label_noise

        tr_labels = inject_label_noise(
            tr_labels, NEWSGROUPS20_NUM_CLASSES, label_noise, seed=noise_seed, kind=noise_type
        )

    held = None
    if noisy_val_frac and noisy_val_frac > 0.0:
        from gam_softmax.data.label_noise import split_heldout

        held_idx, keep_idx = split_heldout(len(tr_texts), noisy_val_frac, seed=noise_seed)
        held = _ListTextDataset([tr_texts[i] for i in held_idx], [tr_labels[i] for i in held_idx],
                                tokenizer, max_seq_len)
        tr_texts = [tr_texts[i] for i in keep_idx]
        tr_labels = [tr_labels[i] for i in keep_idx]
        clean_labels = [clean_labels[i] for i in keep_idx]

    train_ds = _ListTextDataset(tr_texts, tr_labels, tokenizer, max_seq_len)
    out = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=_collate,
        ),
        "val": DataLoader(
            _ListTextDataset(va_texts, va_labels, tokenizer, max_seq_len),
            batch_size=eval_batch_size or batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate,
        ),
        "n_classes": NEWSGROUPS20_NUM_CLASSES,
        # extras the memorization probe needs (see gam_softmax/data/probe.py)
        "train_dataset": train_ds,
        "train_clean_labels": clean_labels,
        "train_noisy_labels": list(tr_labels),
        "collate": _collate,
    }
    if held is not None:
        out["noisy_val"] = DataLoader(
            held,
            batch_size=eval_batch_size or batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate,
        )
    return out
