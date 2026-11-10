"""Leak-freedom: nothing used to fit a backtest model may depend on post-cutoff data.

The strongest check: scramble every value after the cutoff and assert that the model
construction (emission factors, normalisation, static features) and the
multiple-shooting training batch are bit-for-bit unchanged.
"""
import dataclasses

import numpy as np

from carbon_world_model.model.world_model import WorldModel
from carbon_world_model.train.fit import shooting_batch, year_window_mask

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


def test_stability_batch_ignores_future(panel, cfg):
    from carbon_world_model.train.fit import stability_batch
    k = int(np.where(panel.years == CUT)[0][0])
    wm = WorldModel.build(panel, cfg, upto_year=CUT)
    ba = stability_batch(wm, panel, k, H=30)
    bb = stability_batch(wm, _scrambled(panel, CUT), k, H=30)
    for key in ("gdp", "pop", "process"):
        assert np.array_equal(np.asarray(ba["drivers"][key]), np.asarray(bb["drivers"][key]))
    for key in ba["inits"]:
        assert np.array_equal(np.asarray(ba["inits"][key]), np.asarray(bb["inits"][key]))
    assert np.array_equal(np.asarray(ba["tfrac"]), np.asarray(bb["tfrac"]))
