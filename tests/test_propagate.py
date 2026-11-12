"""Backtest-error propagation: exact properties."""
import numpy as np

from carbon_world_model.eval.propagate import correct_path, drift_rate


def test_drift_rate_recovers_an_exact_trend_error():
    true = 30000 * np.exp(0.02 * np.arange(1, 18))
    pred = true * np.exp(0.011 * np.arange(1, 18))
    assert abs(drift_rate(pred, true) - 0.011) < 1e-12
    assert drift_rate(true, true) == 0.0


def test_correct_path_inverts_the_error_and_keeps_the_base_year():
    path = np.linspace(35000, 40000, 29)
    biased = path * np.exp(0.01 * np.arange(29))
    np.testing.assert_allclose(correct_path(biased, 0.01), path, rtol=1e-12)
    assert correct_path(biased, 0.05)[0] == biased[0]


def test_capped_correction_stops_growing():
    path = np.full(29, 1.0)
    c = correct_path(path, 0.01, h_cap=19)
    np.testing.assert_allclose(c[19:], np.exp(-0.19))
    assert np.all(np.diff(c[:20]) < 0)


def test_long_phases_end_on_fixed_years_with_long_horizons(panel):
    from carbon_world_model.eval.rolling import (
        LONG_TEST_END, LONG_VAL_END, PHASES, TEST_CUTOFFS, horizon_for,
    )
    for phase, end in (("longval", LONG_VAL_END), ("longtest", LONG_TEST_END)):
        cutoffs, horizon_of = PHASES[phase]
        for c in cutoffs:
            assert c + horizon_of(c) == end
            assert horizon_of(c) >= 14
            assert horizon_for(panel, c, horizon_of(c)) == horizon_of(c)  # data reach the end year
    # long validation outcomes stop before the test folds' outcomes begin
    assert LONG_VAL_END <= min(TEST_CUTOFFS)


def test_naive_driver_error_is_zero_for_constant_growth(panel):
    import dataclasses
    from carbon_world_model.eval.propagate import driver_error_vectors
    from carbon_world_model.eval.rolling import PHASES
    t = np.arange(panel.n_years)[None, :]
    rates = np.linspace(0.0, 0.06, panel.n_countries)[:, None]
    synth = dataclasses.replace(panel, gdp=panel.gdp[:, :1] * np.exp(rates * t),
                                population=panel.population[:, :1] * np.exp(0.5 * rates * t))
    vecs = driver_error_vectors(synth, "naive_long")
    assert len(vecs) == len(PHASES["longval"][0]) + len(PHASES["longtest"][0])
    for v in vecs:
        np.testing.assert_allclose(v["delta_gdp"], 0.0, atol=1e-10)
        np.testing.assert_allclose(v["delta_pop"], 0.0, atol=1e-10)
        assert abs(v["delta_gdp_aggregate"]) < 1e-3  # aggregate of exact paths: exact up to mix


def test_ssp_driver_vectors_cover_all_ssps_and_are_finite(panel):
    from carbon_world_model.eval.propagate import driver_error_vectors
    vecs = driver_error_vectors(panel, "ssp_v9")
    assert len(vecs) == 5 and all(v["horizon"] == int(panel.years[-1]) - 2014 for v in vecs)
    assert all(np.isfinite(v["delta_gdp"]).all() and np.isfinite(v["delta_pop"]).all() for v in vecs)


def test_perturb_drivers_keeps_anchor_and_applies_trend_error(panel, cfg):
    from carbon_world_model.eval.propagate import perturb_drivers
    from carbon_world_model.plan.scenario import build_scenario
    scen = build_scenario(panel, cfg, end_year=2050)
    same = perturb_drivers(scen, np.zeros(scen.C), np.zeros(scen.C))
    np.testing.assert_array_equal(same.gdp, scen.gdp)
    d = np.full(scen.C, 0.01)
    p = perturb_drivers(scen, d, -d)
    np.testing.assert_allclose(p.gdp[:, 0], scen.gdp[:, 0])
    np.testing.assert_allclose(p.gdp[:, -1] / scen.gdp[:, -1], np.exp(-0.01 * (scen.Tf - 1)))
    np.testing.assert_allclose(p.pop[:, -1] / scen.pop[:, -1], np.exp(0.01 * (scen.Tf - 1)))
