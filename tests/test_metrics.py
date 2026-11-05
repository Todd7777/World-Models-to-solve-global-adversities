"""Metrics are checked against hand-computed values."""
import numpy as np

from carbon_world_model.eval import metrics as M


def test_rmse_mae_known():
    pred = np.array([1.0, 2.0, 3.0])
    true = np.array([1.0, 2.0, 5.0])
    assert M.rmse(pred, true) == np.sqrt(4 / 3)
    assert abs(M.mape(pred, true) - (0 + 0 + 2 / 5) / 3) < 1e-12


def test_r2_perfect_and_mean():
    true = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(M.r2(true, true) - 1.0) < 1e-12
    # predicting the mean gives R2 = 0
    assert abs(M.r2(np.full_like(true, true.mean()), true)) < 1e-12


def test_skill_sign():
    true = np.array([10.0, 20.0, 30.0])
    good = true + 1.0
    bad = true + 5.0
    assert M.skill(good, bad, true) > 0  # good beats bad -> positive skill


def test_coverage_and_interval_score():
    true = np.array([0.0, 5.0, 10.0])
    lo = np.array([-1.0, 4.0, 20.0])
    hi = np.array([1.0, 6.0, 30.0])
    assert abs(M.coverage(lo, hi, true) - 2 / 3) < 1e-12
    # interval score is positive and penalises the missed point
    assert M.interval_score(lo, hi, true, alpha=0.2) > 0


def test_mask_applies():
    pred = np.array([1.0, 100.0])
    true = np.array([1.0, 1.0])
    mask = np.array([1.0, 0.0])
    assert M.rmse(pred, true, mask) == 0.0
