import numpy as np
import pytest

from traffic_vision.evaluation import (block_bootstrap_means, calibrate, default_block,
                                       diff_in_diff)


def test_block_bootstrap_shapes_and_constant_series():
    rng = np.random.default_rng(0)
    means = block_bootstrap_means([5.0] * 12, 200, 3, rng)
    assert means.shape == (200,) and np.allclose(means, 5.0)
    assert default_block(27) == 3 and default_block(1) == 1


def test_did_point_estimate_and_relative_effect():
    tb, td = [10, 12, 10, 12], [8, 8, 8, 8]       # treated: 11 -> 8
    cb, cd = [20, 20, 22, 22], [23, 23, 23, 23]   # control: 21 -> 23, city-wide +2
    res = diff_in_diff(tb, td, cb, cd, n_boot=2000)
    assert res.effect == pytest.approx(-5.0)       # (8 - 11) - (23 - 21)
    assert res.relative == pytest.approx(-5.0 / 13.0)  # counterfactual 11 + 2
    assert res.ci[0] <= -5.0 <= res.ci[1]


def test_calibration_factor_and_its_use_in_did():
    cal = calibrate(manual=[10, 12, 9, 11], auto=[8, 10, 7, 9])  # detector finds about 81%
    assert cal.factor == pytest.approx(42 / 34)
    assert cal.relative_error == pytest.approx((34 - 42) / 42)
    tb, td, cb, cd = [10, 10, 10], [6, 6, 6], [10, 10, 10], [10, 10, 10]
    raw = diff_in_diff(tb, td, cb, cd, n_boot=1000)
    corrected = diff_in_diff(tb, td, cb, cd, n_boot=1000, calibration=cal)
    assert corrected.effect == pytest.approx(raw.effect * cal.factor)
    assert corrected.relative == pytest.approx(raw.relative)   # a ratio is not rescaled
    assert corrected.ci[1] - corrected.ci[0] > 0                # factor uncertainty widens the CI


def test_inputs_are_checked():
    with pytest.raises(ValueError):
        diff_in_diff([1], [1, 2], [1, 2], [1, 2])
    with pytest.raises(ValueError):
        calibrate([1, 2], [0, 0])


def test_t_quantile_matches_tables():
    from traffic_vision.evaluation import t_quantile

    assert t_quantile(0.975, 3) == pytest.approx(3.182, abs=1e-3)
    assert t_quantile(0.975, 30) == pytest.approx(2.042, abs=1e-3)


def test_batch_and_block_methods_agree_on_the_estimate():
    rng = np.random.default_rng(1)
    s = [rng.poisson(m, 24) for m in (60, 78, 50, 56)]
    batch = diff_in_diff(*s, method="batch")
    block = diff_in_diff(*s, method="block", n_boot=2000)
    assert batch.effect == pytest.approx(block.effect)
    assert batch.method == "batch" and "4 batches" in batch.detail
    assert batch.ci[0] < batch.effect < batch.ci[1]
    assert batch.relative_ci[0] < batch.relative < batch.relative_ci[1]
    with pytest.raises(ValueError):
        diff_in_diff(*s, method="magic")
