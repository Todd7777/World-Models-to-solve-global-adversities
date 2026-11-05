"""Forecast-evaluation metrics (point and probabilistic)."""
from __future__ import annotations

import numpy as np


def _flat(pred, true, mask=None):
    pred = np.asarray(pred, float).ravel()
    true = np.asarray(true, float).ravel()
    if mask is not None:
        m = np.asarray(mask, bool).ravel()
        pred, true = pred[m], true[m]
    ok = np.isfinite(pred) & np.isfinite(true)
    return pred[ok], true[ok]


def mape(pred, true, mask=None) -> float:
    p, t = _flat(pred, true, mask)
    return float(np.mean(np.abs(p - t) / np.clip(np.abs(t), 1e-6, None)))


def smape(pred, true, mask=None) -> float:
    p, t = _flat(pred, true, mask)
    return float(np.mean(2 * np.abs(p - t) / np.clip(np.abs(p) + np.abs(t), 1e-6, None)))


def rmse(pred, true, mask=None) -> float:
    p, t = _flat(pred, true, mask)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def r2(pred, true, mask=None) -> float:
    p, t = _flat(pred, true, mask)
    ss = np.sum((p - t) ** 2)
    tot = np.sum((t - t.mean()) ** 2)
    return float(1 - ss / tot) if tot > 0 else float("nan")


def bias(pred, true, mask=None) -> float:
    p, t = _flat(pred, true, mask)
    return float(np.mean(p - t))


def skill(pred, ref, true, mask=None) -> float:
    """Forecast skill vs a reference forecast: 1 - RMSE_pred / RMSE_ref (>0 is better)."""
    return float(1.0 - rmse(pred, true, mask) / (rmse(ref, true, mask) + 1e-12))


def coverage(lower, upper, true, mask=None) -> float:
    """Empirical coverage of a prediction interval."""
    lo, t = _flat(lower, true, mask)
    up, _ = _flat(upper, true, mask)
    return float(np.mean((t >= lo) & (t <= up)))


def interval_score(lower, upper, true, alpha: float, mask=None) -> float:
    """Interval score for a central (1-alpha) interval (lower is better)."""
    lo, t = _flat(lower, true, mask)
    up, _ = _flat(upper, true, mask)
    width = up - lo
    penalty = (2 / alpha) * ((lo - t) * (t < lo) + (t - up) * (t > up))
    return float(np.mean(width + penalty))
