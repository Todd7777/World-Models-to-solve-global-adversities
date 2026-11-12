"""Carry long-horizon backtest error into planning ranges.

The long-horizon backtests (eval/rolling.PHASES 'longval' and 'longtest') show the world
model's aggregate CO2 error growing roughly linearly in log level with horizon h:

    log(predicted / observed)_h  ~=  delta * h

i.e. the model gets the trend rate wrong by delta per year. Each backtest (cutoff x seed)
gives one delta, fit by least squares through the origin over its horizon. To carry that
error into a projection, a projected path X_h (h years after the base year) is corrected
as X_h * exp(-delta * min(h, h_cap)): a model that ran high by e^(delta h) in a backtest
implies the truth was below its path by the same factor.

Assumptions, stated: the same trend-rate error applies to business-as-usual and to
controlled paths (an error in the natural dynamics, not in the effect of the controls);
by default the error keeps growing linearly beyond the 15-19-year backtest horizons
(h_cap=None), and h_cap caps it as a sensitivity case. The deltas are a handful of errors
the model actually made, not a confidence interval.
"""
from __future__ import annotations

import os

import numpy as np

from .rolling import PHASES, job_path

LONG_PHASES = ("longval", "longtest")


def drift_rate(pred_agg: np.ndarray, true_agg: np.ndarray) -> float:
    """delta: least-squares slope through the origin of log(pred/true) vs horizon 1..H."""
    le = np.log(pred_agg / true_agg)
    h = np.arange(1, len(le) + 1)
    return float(np.sum(h * le) / np.sum(h * h))


def backtest_drifts(panel, config: str, seeds=(0, 1), phases=LONG_PHASES) -> list[dict]:
    """One record per (phase, cutoff, seed) with a cached long-horizon forecast."""
    out = []
    for ph in phases:
        cutoffs, _ = PHASES[ph]
        for co in cutoffs:
            k = int(np.where(panel.years == co)[0][0])
            for s in seeds:
                path = job_path(f"results/sweep_{ph}", config, co, s)
                if not os.path.exists(path):
                    continue
                r = np.load(path)
                H = int(r["H"])
                m = panel.mask[:, k + 1:k + H + 1]
                true = (panel.co2[:, k + 1:k + H + 1] * m).sum(0)
                pred = (r["co2_realized"][:, 1:] * m).sum(0)
                out.append({"phase": ph, "cutoff": int(co), "seed": int(s), "horizon": H,
                            "delta": drift_rate(pred, true),
                            "cum_err": float(pred.sum() / true.sum() - 1)})
    return out


def correct_path(path: np.ndarray, delta: float, h_cap: int | None = None) -> np.ndarray:
    """Apply a trend-rate error to a path whose index 0 is the (observed) base year."""
    h = np.arange(len(path), dtype=float)
    if h_cap is not None:
        h = np.minimum(h, h_cap)
    return np.asarray(path) * np.exp(-delta * h)


# ---------------------------------------------------------------------------
# Driver (GDP / population) error
# ---------------------------------------------------------------------------
# Planning drivers are SSP paths. Their error is measured the same way as the model's:
# as a per-country trend-rate error delta (slope through the origin of
# log(projected / realised) vs horizon). Each historical projection gives one JOINT
# vector over all countries, which keeps the cross-country correlation that matters for
# the global total. Two sources:
#   'naive_long': each country's trailing 10-year growth continued from every long
#                 backtest cutoff (eval/rolling.PHASES longval/longtest), 14-20 years.
#   'ssp_v9':     the 2014-vintage SSP projections (SSP1-5), harmonised at 2014, against
#                 realised 2015-2022 (8 years, including COVID).
# Over 2015-2022 the naive trend's aggregate GDP error (+1.50%/yr) was close to, and
# slightly larger than, the 2014 SSP2's (+1.31%/yr), so naive long-horizon errors stand
# in for SSP errors at horizons no SSP vintage can yet be checked on.

NAIVE_LOOKBACK = 10


def _naive_path(series: np.ndarray, k: int, H: int) -> np.ndarray:
    j = max(0, k - NAIVE_LOOKBACK)
    r = (np.log(series[:, k]) - np.log(series[:, j])) / (k - j)
    return series[:, k:k + 1] * np.exp(r[:, None] * np.arange(1, H + 1))


def driver_error_vectors(panel, source: str) -> list[dict]:
    """Joint per-country GDP and population trend-rate errors, one record per projection."""
    out = []
    if source == "naive_long":
        for ph in LONG_PHASES:
            cutoffs, horizon_of = PHASES[ph]
            for co in cutoffs:
                k = int(np.where(panel.years == co)[0][0])
                H = horizon_of(co)
                rg, rp = panel.gdp[:, k + 1:k + H + 1], panel.population[:, k + 1:k + H + 1]
                ng, npop = _naive_path(panel.gdp, k, H), _naive_path(panel.population, k, H)
                out.append({"source": source, "label": f"naive from {co}", "cutoff": int(co),
                            "horizon": H, **_deltas(ng, npop, rg, rp)})
    elif source == "ssp_v9":
        from ..data.ssp import SSPS, load_processed, ssp_driver_paths
        base, end = 2014, int(panel.years[-1])
        k = int(np.where(panel.years == base)[0][0])
        series = load_processed("v9")
        for s in SSPS:
            g, p = ssp_driver_paths(panel, series, s, base, np.arange(base, end + 1))
            out.append({"source": source, "label": f"{s} (2014 vintage)", "cutoff": base,
                        "horizon": end - base,
                        **_deltas(g[:, 1:], p[:, 1:], panel.gdp[:, k + 1:], panel.population[:, k + 1:])})
    else:
        raise ValueError(source)
    return out


def _deltas(proj_gdp, proj_pop, real_gdp, real_pop) -> dict:
    h = np.arange(1, proj_gdp.shape[1] + 1)

    def slope(le):
        return (le * h).sum(-1) / (h * h).sum()
    return {"delta_gdp": slope(np.log(proj_gdp / real_gdp)).tolist(),
            "delta_pop": slope(np.log(proj_pop / real_pop)).tolist(),
            "delta_gdp_aggregate": float(slope(np.log(proj_gdp.sum(0) / real_gdp.sum(0))))}


def perturb_drivers(scen, delta_gdp, delta_pop, h_cap: int | None = None):
    """Scenario whose GDP / population paths are corrected by exp(-delta * min(h, h_cap))
    per country (index 0, the observed anchor year, is unchanged). h_cap freezes the
    correction, e.g. at 2050 for runs to 2100."""
    import dataclasses
    h = np.arange(scen.gdp.shape[1], dtype=float)[None, :]
    if h_cap is not None:
        h = np.minimum(h, h_cap)
    return dataclasses.replace(
        scen, gdp=scen.gdp * np.exp(-np.asarray(delta_gdp)[:, None] * h),
        pop=scen.pop * np.exp(-np.asarray(delta_pop)[:, None] * h))
