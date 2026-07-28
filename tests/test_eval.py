from __future__ import annotations

import torch

from gam_softmax.eval.classification import expected_calibration_error


def test_ece_is_zero_for_perfectly_calibrated_predictions():
    """Half the predictions right at exactly 50% stated confidence — perfectly
    calibrated, so ECE must be ~0 even though accuracy is only 50%."""
    conf = torch.full((100,), 0.5)
    correct = torch.zeros(100)
    correct[:50] = 1.0
    assert expected_calibration_error(conf, correct) < 1e-6


def test_ece_flags_overconfidence():
    """The failure mode we are measuring: the model claims 99% and is right half
    the time. ECE must be close to the 0.49 gap."""
    conf = torch.full((100,), 0.99)
    correct = torch.zeros(100)
    correct[:50] = 1.0
    assert abs(expected_calibration_error(conf, correct) - 0.49) < 1e-3


def test_ece_is_symmetric_for_underconfidence():
    conf = torch.full((100,), 0.6)
    correct = torch.ones(100)
    assert abs(expected_calibration_error(conf, correct) - 0.4) < 1e-3


def test_ece_weights_bins_by_population():
    """A tiny badly-calibrated bin must not dominate a large well-calibrated one."""
    conf = torch.cat([torch.full((990,), 0.9), torch.full((10,), 0.1)])
    correct = torch.cat([torch.ones(990), torch.ones(10)])
    ece = expected_calibration_error(conf, correct)
    # 0.99 * 0.1 + 0.01 * 0.9
    assert abs(ece - (0.99 * 0.1 + 0.01 * 0.9)) < 1e-3


def test_ece_handles_empty_input():
    assert expected_calibration_error(torch.empty(0), torch.empty(0)) != \
        expected_calibration_error(torch.empty(0), torch.empty(0))  # nan != nan
