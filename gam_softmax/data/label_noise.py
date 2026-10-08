from __future__ import annotations

import random
from typing import Iterable, List


def inject_label_noise(
    labels: Iterable[int],
    n_classes: int,
    noise_rate: float,
    seed: int = 0,
    kind: str = "symmetric",
) -> List[int]:
    """Return a copy of ``labels`` with symmetric (default) or pair-flip label noise.

    ``kind="pair"`` is the asymmetric pair-flip model of Co-teaching (Han et al.
    2018): a corrupted label always becomes ``(y + 1) mod n_classes``. The wrong
    class is then systematic rather than random, which is the harder case: the
    model can learn the flip as a pattern. It needs ``noise_rate < 0.5``, or the
    flipped class becomes the majority label and the true class is unidentifiable.

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
    if kind not in ("symmetric", "pair"):
        raise ValueError(f"kind must be 'symmetric' or 'pair', got {kind!r}")
    if kind == "pair" and noise_rate >= 0.5:
        raise ValueError(f"pair noise needs noise_rate < 0.5, got {noise_rate}")

    labels = [int(y) for y in labels]
    if noise_rate <= 0.0:
        return labels

    rng = random.Random(seed)
    out: List[int] = []
    for y in labels:
        # one rng.random() per label in both modes, so the symmetric stream (and
        # every earlier run's corrupted labels) is unchanged by the pair option
        if rng.random() < noise_rate:
            if kind == "pair":
                out.append((y + 1) % n_classes)
                continue
            # uniform over the other n_classes - 1 classes (shift trick avoids y)
            wrong = rng.randrange(n_classes - 1)
            if wrong >= y:
                wrong += 1
            out.append(wrong)
        else:
            out.append(y)
    return out


def split_heldout(n: int, frac: float, seed: int = 0) -> tuple[List[int], List[int]]:
    """Deterministically split ``range(n)`` into (held_out, kept) index lists, both sorted.

    Used to carve a noisy validation split out of the training set. Seeded from
    the noise seed (not the run seed) so every method holds out the same examples.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError(f"frac must be in (0, 1), got {frac}")
    idx = list(range(n))
    random.Random(seed + 10_007).shuffle(idx)
    n_held = int(round(n * frac))
    return sorted(idx[:n_held]), sorted(idx[n_held:])
