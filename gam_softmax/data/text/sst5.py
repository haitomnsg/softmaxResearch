from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


SST5_HF_NAME = "SetFit/sst5"
SST5_NUM_CLASSES = 5


@dataclass
class SST5Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class _SST5Dataset(Dataset):
    def __init__(self, hf_split, tokenizer, max_seq_len: int):
        self.examples = hf_split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        row = self.examples[idx]
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def _collate(batch: list[dict]) -> SST5Batch:
    return SST5Batch(
        input_ids=torch.stack([b["input_ids"] for b in batch]),
        attention_mask=torch.stack([b["attention_mask"] for b in batch]),
        labels=torch.stack([b["labels"] for b in batch]),
    )


def load_sst5(
    tokenizer,
    batch_size: int = 16,
    max_seq_len: int = 128,
    num_workers: int = 0,
    limit_train: Optional[int] = None,
) -> dict:
    """Load SST-5 from HuggingFace and return train/val/test DataLoaders."""
    from datasets import load_dataset

    ds = load_dataset(SST5_HF_NAME)
    train = ds["train"]
    val = ds["validation"]
    test = ds["test"]
    if limit_train is not None:
        train = train.select(range(min(limit_train, len(train))))

    return {
        "train": DataLoader(
            _SST5Dataset(train, tokenizer, max_seq_len),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=_collate,
        ),
        "val": DataLoader(
            _SST5Dataset(val, tokenizer, max_seq_len),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate,
        ),
        "test": DataLoader(
            _SST5Dataset(test, tokenizer, max_seq_len),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate,
        ),
        "n_classes": SST5_NUM_CLASSES,
    }
