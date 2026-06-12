from __future__ import annotations

import random
from typing import Iterable, List


def inject_label_noise(
    labels: Iterable[int],
    n_classes: int,
    noise_rate: float,
    seed: int = 0,
) -> List[int]:
    """Return a copy of ``labels`` with symmetric label noise applied.

    Each label is, independently with probability ``noise_rate``, replaced by a
    class drawn uniformly from the ``n_classes - 1`` *other* classes (a true flip
    — never back to the original). This is the standard "symmetric / uniform"
    noise model used in the noisy-label literature.

    The ``seed`` is deliberately **decoupled from the training/run seed** so that
    every method (softmax / AS-Softmax / GAM) at a given noise level sees the
    *same* corrupted labels — that is what makes the method comparison fair. Vary
    the run seed for model init / batch order; keep ``seed`` fixed per noise level.

    Only training labels should ever be passed here; validation/test labels stay
    clean so reported accuracy is measured against ground truth.

    Args:
        labels: ground-truth integer class ids.
        n_classes: total number of classes (>= 2).
        noise_rate: fraction of labels to corrupt, in [0, 1].
        seed: RNG seed controlling which labels flip and where to.

    Returns:
        A new list of (possibly corrupted) integer labels. The input is not
        mutated. ``noise_rate <= 0`` returns an unchanged copy.
    """
    if not 0.0 <= noise_rate <= 1.0:
        raise ValueError(f"noise_rate must be in [0, 1], got {noise_rate}")
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2 to inject noise, got {n_classes}")

    labels = [int(y) for y in labels]
    if noise_rate <= 0.0:
        return labels

    rng = random.Random(seed)
    out: List[int] = []
    for y in labels:
        if rng.random() < noise_rate:
            # uniform over the other n_classes - 1 classes (shift trick avoids y)
            wrong = rng.randrange(n_classes - 1)
            if wrong >= y:
                wrong += 1
            out.append(wrong)
        else:
            out.append(y)
    return out
