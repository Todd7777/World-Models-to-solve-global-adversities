"""Leak-freedom: nothing used to fit a backtest model may depend on post-cutoff data.

The strongest check: scramble every value after the cutoff and assert that the model
construction (emission factors, normalisation, static features), the multiple-shooting
training batch, the training loss and the fitted parameters are bit-for-bit unchanged.
"""
import dataclasses

import jax
import numpy as np

from carbon_world_model.model.world_model import WorldModel
from carbon_world_model.eval.configs import make_config
from carbon_world_model.train.fit import fit, shooting_batch, year_window_mask, _targets
from carbon_world_model.train.losses import loss_terms

CUT = 2012


def _scrambled(panel, cut_year):
    k = int(np.where(panel.years == cut_year)[0][0])
    rng = np.random.default_rng(0)
    def scr(a):
        a = a.copy()
        a[:, k + 1:] = a[:, k + 1:] * rng.uniform(0.3, 3.0, size=a[:, k + 1:].shape)
        return a
    energy = scr(panel.energy)
    return dataclasses.replace(
        panel, energy=energy, primary_energy=energy.sum(axis=2), co2=scr(panel.co2),
        co2_by_source=scr(panel.co2_by_source), process_co2=scr(panel.process_co2),
        gdp=scr(panel.gdp), population=scr(panel.population))


def test_model_build_ignores_future(panel, cfg):
    a = WorldModel.build(panel, cfg, upto_year=CUT)
    b = WorldModel.build(_scrambled(panel, CUT), cfg, upto_year=CUT)
    assert np.array_equal(a.ef, b.ef)
    assert np.array_equal(a.static, b.static)
    assert a.norm == b.norm


def test_shooting_batch_ignores_future(panel, cfg):
    k = int(np.where(panel.years == CUT)[0][0])
    wm = WorldModel.build(panel, cfg, upto_year=CUT)
    ba = shooting_batch(wm, panel, k, H=8, stride=1)
    bb = shooting_batch(wm, _scrambled(panel, CUT), k, H=8, stride=1)
    for key in ("gdp", "pop", "process"):
        assert np.array_equal(np.asarray(ba["drivers"][key]), np.asarray(bb["drivers"][key]))
    for key in ("co2", "D", "shares"):
        assert np.array_equal(np.asarray(ba["targets"][key]), np.asarray(bb["targets"][key]))
    assert np.array_equal(np.asarray(ba["mask"]), np.asarray(bb["mask"]))


def test_loss_mask_excludes_future(panel):
    m = np.asarray(year_window_mask(panel, CUT))
    k = int(np.where(panel.years == CUT)[0][0])
    assert (m[:, k + 1:] == 0).all() and (m[:, : k + 1] > 0).any()


def test_forecast_starts_from_observed_state(panel, cfg, wm, init_params):
    w = WorldModel.build(panel, cfg, upto_year=2015)
    fc = w.forecast(w.init_params(0), panel, 2015)
    k = int(np.where(panel.years == 2015)[0][0])
    assert np.allclose(np.asarray(fc["energy"][:, 0, :]), panel.energy[:, k, :], rtol=1e-3)
    # process CO2 is held at the cutoff value (no future information)
    co2_0 = np.asarray(fc["co2"][:, 0])
    assert np.isfinite(co2_0).all()


def _same_tree(a, b) -> bool:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(np.array_equal(np.asarray(x), np.asarray(y))
                                      for x, y in zip(la, lb))


def test_fit_ignores_future(panel):
    """Training itself, end to end: a few optimiser steps give bit-for-bit the same
    parameters when every value after the cutoff is scrambled. Covers the full-run loss
    (whose free rollout runs past the cutoff on realised drivers) and, for ms_full, the
    multiple-shooting loss."""
    scr = _scrambled(panel, CUT)
    for name in ("orig", "ms_full"):
        cfg = make_config(name, steps=2)
        a = fit(panel, cfg, train_year_max=CUT, verbose=False).params
        b = fit(scr, cfg, train_year_max=CUT, verbose=False).params
        assert _same_tree(a, b), f"{name}: fitted parameters depend on post-cutoff data"


def test_loss_window_excludes_everything_outside_it(panel, cfg):
    """With a fit window (some countries, years <= cutoff), the full-run loss value and
    gradient ignore every value outside it: later years and the other countries' rows."""
    k = int(np.where(panel.years == CUT)[0][0])
    rows = np.zeros(panel.n_countries)
    rows[::2] = 1.0
    window = (panel.years[None, :] <= CUT) * rows[:, None]
    wm = WorldModel.build(panel, cfg, upto_year=CUT)
    p = wm.init_params(0)
    rng = np.random.default_rng(1)

    def outside(a):
        a = a.copy()
        noise = rng.uniform(0.3, 3.0, size=a.shape)
        keep = window.astype(bool) if a.ndim == 2 else window.astype(bool)[..., None]
        return np.where(keep, a, a * noise)

    energy = outside(panel.energy)
    other = dataclasses.replace(
        panel, energy=energy, primary_energy=energy.sum(axis=2), co2=outside(panel.co2),
        gdp=outside(panel.gdp), population=outside(panel.population),
        process_co2=outside(panel.process_co2))
    assert np.array_equal(panel.gdp[:, : k + 1][rows > 0], other.gdp[:, : k + 1][rows > 0])

    def loss(pp, pan):
        mask = year_window_mask(pan, CUT) * rows[:, None]
        return loss_terms(wm.rollout(pp, pan), _targets(pan), mask, cfg, window=window)[0]

    # Rows outside the window still start from (scrambled) observed states and run on
    # their own drivers; only the loss must not see them.
    la, ga = jax.value_and_grad(loss)(p, panel)
    lb, gb = jax.value_and_grad(loss)(p, other)
    assert float(la) == float(lb)
    assert _same_tree(ga, gb)
