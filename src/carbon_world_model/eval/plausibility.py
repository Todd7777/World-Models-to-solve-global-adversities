"""Long-run LEVEL plausibility of projections: the energy ladder.

Across countries and years, energy use per person rises with income in a stable,
concave way. A 28-year projection can be smooth yet still move a country to an energy
level no country has ever reached for its income, or drift off the cross-country
relationship faster than any country has. This check uses observed data only
(1990-2022); it needs no future data and asks only whether a projection is historically
unprecedented, not whether it is right.

  ladder:   log(E/pop) = a + b*x + c*x^2,  x = log(GDP/pop), fit by OLS on all observed
            country-years
  residual: a country-year's distance above (+) or below (-) the ladder
  drift:    change in a country's residual over `span` years

Two references, both observed:
  pooled:   every observed `span`-year drift of any country. A projected 2022->2050 drift
            outside its central `coverage` interval is flagged as unprecedented. By
            construction about (1 - coverage) of historical windows fall outside, so that
            is the base rate. This band is wide (it includes post-Soviet transitions and
            petrostates), so it only catches the absurd.
  own past: the country's own most recent `span`-year drift (e.g. 1994->2022).
            excess = projected drift - own past drift, per year. Negative means the
            projection decouples energy from income faster than that country did,
            positive slower. Because both are measured against the same income ladder,
            this accounts for faster or slower income growth in the projection.
The static ladder has no time trend, so most countries drift downward over time
(median 28-year drift about -0.47): that is ordinary historical decoupling, and it is
why the own-past comparison is the informative one.
"""
from __future__ import annotations

import numpy as np


def _xy(gdp, pop, energy):
    return np.log(gdp / pop), np.log(energy / pop)


def fit_ladder(panel) -> np.ndarray:
    """OLS coefficients (a, b, c) of the quadratic energy ladder on observed data."""
    m = panel.mask.astype(bool)
    x, y = _xy(panel.gdp, panel.population, panel.primary_energy)
    X = np.stack([np.ones(m.sum()), x[m], x[m] ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    return coef


def residual(coef, gdp, pop, energy) -> np.ndarray:
    x, y = _xy(gdp, pop, energy)
    return y - (coef[0] + coef[1] * x + coef[2] * x ** 2)


def historical_drifts(panel, coef, span: int) -> np.ndarray:
    """Every observed `span`-year change in a country's ladder residual."""
    r = residual(coef, panel.gdp, panel.population, panel.primary_energy)
    m = panel.mask.astype(bool)
    out = [r[c, t + span] - r[c, t] for c in range(r.shape[0])
           for t in range(r.shape[1] - span) if m[c, t] and m[c, t + span]]
    return np.asarray(out)


def ladder_check(panel, gdp_end, pop_end, energy_end, span: int | None = None,
                 coverage: float = 0.95) -> dict:
    """Flag countries whose projected drift from the last observed year is unprecedented.

    gdp_end, pop_end, energy_end: [C] projected values `span` years after the panel's last
    year (default span: 2050 - last year).
    """
    span = int(2050 - panel.years[-1]) if span is None else span
    coef = fit_ladder(panel)
    hist = historical_drifts(panel, coef, span)
    lo, hi = np.quantile(hist, [(1 - coverage) / 2, 1 - (1 - coverage) / 2])
    r0 = residual(coef, panel.gdp[:, -1], panel.population[:, -1], panel.primary_energy[:, -1])
    r1 = residual(coef, np.asarray(gdp_end), np.asarray(pop_end), np.asarray(energy_end))
    drift = r1 - r0
    flag = (drift < lo) | (drift > hi)
    w = np.asarray(energy_end) / np.sum(energy_end)
    r_all = residual(coef, panel.gdp, panel.population, panel.primary_energy)
    m = panel.mask.astype(bool)
    own = np.where(m[:, -1] & m[:, -1 - span], r_all[:, -1] - r_all[:, -1 - span], np.nan)
    excess = (drift - own) / span                    # log units per year
    ok = np.isfinite(excess)
    return {
        "span": span, "coverage": coverage, "n_historical_windows": int(hist.size),
        "historical_band": [float(lo), float(hi)],
        "historical_drift_range": [float(hist.min()), float(hist.max())],
        "drift": drift.tolist(),
        "flagged_countries": [panel.countries[i] for i in np.where(flag)[0]],
        "share_flagged": float(flag.mean()),
        "energy_share_flagged": float(np.sum(w * flag)),
        "energy_share_above": float(np.sum(w * (drift > hi))),
        "energy_share_below": float(np.sum(w * (drift < lo))),
        "base_rate": float(1 - coverage),
        "own_past_years": [int(panel.years[-1 - span]), int(panel.years[-1])],
        "own_past_drift": own.tolist(),
        "excess_drift_per_year": excess.tolist(),
        "energy_weighted_excess_per_year": float(np.sum(w[ok] * excess[ok]) / np.sum(w[ok])),
        "energy_share_faster_than_own_past": float(np.sum(w[ok] * (excess[ok] < 0)) / np.sum(w[ok])),
    }
