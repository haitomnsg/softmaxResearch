from __future__ import annotations

import random
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset


def build_probe(
    data: dict,
    size: int = 1200,
    seed: int = 0,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Optional[dict]:
    """Build the memorization probe the Trainer consumes.

    Takes a fixed random sample of the *training* set (in a fixed order, no
    shuffling) so every epoch measures the same examples, and carries the
    ground-truth labels alongside so the trainer can separate "fitted the label
    it was given" from "got the answer right".

    ``seed`` is deliberately independent of the run seed: every method at a
    given noise level probes the same examples, which is what makes the
    memorization numbers comparable across methods.

    Returns ``None`` if the loader dict doesn't carry the fields needed
    (older loaders), so callers can treat the probe as optional.
    """
    ds = data.get("train_dataset")
    clean = data.get("train_clean_labels")
    if ds is None or clean is None:
        return None

    n = len(ds)
    size = min(size, n)
    idx = random.Random(seed).sample(range(n), size)
    idx.sort()  # keep a stable, reproducible order

    clean_labels = torch.tensor([int(clean[i]) for i in idx], dtype=torch.long)
    noisy = data.get("train_noisy_labels")
    if noisy is None:
        flipped = torch.zeros(size, dtype=torch.bool)
    else:
        noisy_labels = torch.tensor([int(noisy[i]) for i in idx], dtype=torch.long)
        flipped = noisy_labels != clean_labels

    loader = DataLoader(
        Subset(ds, idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data["collate"],
    )
    return {"loader": loader, "clean_labels": clean_labels, "flipped": flipped}
