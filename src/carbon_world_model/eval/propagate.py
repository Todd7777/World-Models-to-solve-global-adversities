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
