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
