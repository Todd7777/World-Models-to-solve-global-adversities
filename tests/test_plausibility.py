"""Energy-ladder plausibility check: properties that hold by construction."""
import numpy as np

from carbon_world_model.eval.plausibility import (
    fit_ladder, historical_drifts, ladder_check, residual,
)

SPAN = 28


def _projection_with_drift(panel, drift, growth=0.02):
    """2050 values with every country's ladder residual moved by `drift` from 2022."""
    coef = fit_ladder(panel)
    pop = panel.population[:, -1] * 1.1
    gdp = panel.gdp[:, -1] * np.exp(growth * SPAN) * 1.1
    r0 = residual(coef, panel.gdp[:, -1], panel.population[:, -1], panel.primary_energy[:, -1])
    x = np.log(gdp / pop)
    energy = pop * np.exp(coef[0] + coef[1] * x + coef[2] * x ** 2 + r0 + drift)
    return gdp, pop, energy


def test_ladder_is_increasing_in_income_over_observed_range(panel):
    a, b, c = fit_ladder(panel)
    x = np.log(panel.gdp / panel.population)[panel.mask.astype(bool)]
    assert np.all(b + 2 * c * x > 0)


def test_pooled_band_holds_the_stated_share_of_history(panel):
    coef = fit_ladder(panel)
    hist = historical_drifts(panel, coef, SPAN)
    res = ladder_check(panel, *_projection_with_drift(panel, 0.0), span=SPAN)
    lo, hi = res["historical_band"]
    outside = np.mean((hist < lo) | (hist > hi))
    assert abs(outside - res["base_rate"]) < 0.02


def test_repeating_own_past_drift_gives_zero_excess(panel):
    res0 = ladder_check(panel, *_projection_with_drift(panel, 0.0), span=SPAN)
    own = np.array(res0["own_past_drift"])
    res = ladder_check(panel, *_projection_with_drift(panel, np.nan_to_num(own)), span=SPAN)
    ex = np.array(res["excess_drift_per_year"])
    np.testing.assert_allclose(ex[np.isfinite(own)], 0.0, atol=1e-9)


def test_unprecedented_drifts_are_flagged(panel):
    up = ladder_check(panel, *_projection_with_drift(panel, 2.0), span=SPAN)
    down = ladder_check(panel, *_projection_with_drift(panel, -3.0), span=SPAN)
    assert up["share_flagged"] == 1.0 and up["energy_share_above"] == 1.0
    assert down["share_flagged"] == 1.0 and down["energy_share_below"] == 1.0
    assert up["energy_weighted_excess_per_year"] > 0 > down["energy_weighted_excess_per_year"]
