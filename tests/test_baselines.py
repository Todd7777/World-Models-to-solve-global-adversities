"""Baseline forecasters produce valid, sensible outputs."""
import numpy as np

from carbon_world_model.eval.baselines import BASELINES, frozen_kaya, persistence


def test_baselines_shapes_and_finite(panel):
    for name, fn in BASELINES.items():
        pred = fn(panel, 2015)
        assert pred.shape == panel.co2.shape
        assert np.isfinite(pred).all()
        assert (pred >= -1e-6).all()


def test_baselines_dont_touch_training_years(panel):
    k = int(np.where(panel.years == 2015)[0][0])
    for name, fn in BASELINES.items():
        pred = fn(panel, 2015)
        assert np.allclose(pred[:, : k + 1], panel.co2[:, : k + 1], atol=1e-6)


def test_persistence_is_flat(panel):
    k = int(np.where(panel.years == 2015)[0][0])
    pred = persistence(panel, 2015)
    assert np.allclose(pred[:, k + 1:], panel.co2[:, k][:, None])
