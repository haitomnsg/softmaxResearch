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
