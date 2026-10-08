from __future__ import annotations

import pytest

from gam_softmax.data.label_noise import inject_label_noise


def test_zero_noise_returns_unchanged_copy():
    labels = [0, 1, 2, 3, 4, 0, 1, 2]
    out = inject_label_noise(labels, n_classes=5, noise_rate=0.0, seed=0)
    assert out == labels
    assert out is not labels  # a copy, never the same object


def test_does_not_mutate_input():
    labels = [0, 1, 2, 3, 4]
    original = list(labels)
    inject_label_noise(labels, n_classes=5, noise_rate=1.0, seed=0)
    assert labels == original


def test_deterministic_for_same_seed():
    labels = list(range(5)) * 200
    a = inject_label_noise(labels, n_classes=5, noise_rate=0.3, seed=7)
    b = inject_label_noise(labels, n_classes=5, noise_rate=0.3, seed=7)
    assert a == b


def test_different_seed_changes_corruption():
    labels = list(range(5)) * 200
    a = inject_label_noise(labels, n_classes=5, noise_rate=0.3, seed=1)
    b = inject_label_noise(labels, n_classes=5, noise_rate=0.3, seed=2)
    assert a != b


def test_full_noise_flips_every_label_to_a_different_class():
    # rate=1.0 must flip every label, and a "flip" is never back to itself
    labels = list(range(5)) * 100
    out = inject_label_noise(labels, n_classes=5, noise_rate=1.0, seed=0)
    assert all(o != y for o, y in zip(out, labels))


def test_all_outputs_are_valid_class_ids():
    labels = list(range(10)) * 50
    out = inject_label_noise(labels, n_classes=10, noise_rate=0.5, seed=3)
    assert all(0 <= o < 10 for o in out)


def test_corruption_fraction_is_approximately_noise_rate():
    n = 10000
    labels = [i % 5 for i in range(n)]
    out = inject_label_noise(labels, n_classes=5, noise_rate=0.4, seed=0)
    flipped = sum(o != y for o, y in zip(out, labels))
    assert abs(flipped / n - 0.4) < 0.02  # within 2% of the target rate


def test_binary_flips_to_the_other_class():
    labels = [0, 1, 0, 1, 0, 1]
    out = inject_label_noise(labels, n_classes=2, noise_rate=1.0, seed=0)
    assert out == [1, 0, 1, 0, 1, 0]


@pytest.mark.parametrize("bad_rate", [-0.1, 1.5])
def test_rejects_out_of_range_noise_rate(bad_rate):
    with pytest.raises(ValueError):
        inject_label_noise([0, 1], n_classes=2, noise_rate=bad_rate)


def test_rejects_too_few_classes():
    with pytest.raises(ValueError):
        inject_label_noise([0, 0], n_classes=1, noise_rate=0.5)


def _symmetric_reference(labels, n_classes, noise_rate, seed):
    """Frozen copy of the original (pre-2026-10-05) algorithm: guards that adding
    pair noise didn't change the corrupted labels every earlier run trained on."""
    import random
    rng, out = random.Random(seed), []
    for y in labels:
        if rng.random() < noise_rate:
            wrong = rng.randrange(n_classes - 1)
            out.append(wrong + 1 if wrong >= y else wrong)
        else:
            out.append(y)
    return out


def test_symmetric_stream_unchanged_by_pair_option():
    labels = [i % 20 for i in range(2000)]
    assert inject_label_noise(labels, 20, 0.4, seed=7) == _symmetric_reference(labels, 20, 0.4, 7)


def test_pair_noise_flips_to_next_class_only():
    labels = [i % 20 for i in range(4000)]
    noisy = inject_label_noise(labels, 20, 0.3, seed=1, kind="pair")
    flipped = [(y, z) for y, z in zip(labels, noisy) if y != z]
    assert all(z == (y + 1) % 20 for y, z in flipped)
    assert abs(len(flipped) / len(labels) - 0.3) < 0.03


def test_pair_noise_rejects_rate_at_or_above_half():
    with pytest.raises(ValueError):
        inject_label_noise([0, 1, 2], 3, 0.5, kind="pair")
    with pytest.raises(ValueError):
        inject_label_noise([0, 1, 2], 3, 0.2, kind="bogus")


def test_split_heldout_is_deterministic_disjoint_and_complete():
    from gam_softmax.data.label_noise import split_heldout
    held, kept = split_heldout(1000, 0.1, seed=3)
    assert len(held) == 100 and len(kept) == 900
    assert set(held).isdisjoint(kept) and set(held) | set(kept) == set(range(1000))
    assert split_heldout(1000, 0.1, seed=3) == (held, kept)
    assert split_heldout(1000, 0.1, seed=4)[0] != held
    with pytest.raises(ValueError):
        split_heldout(10, 0.0)
