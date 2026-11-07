"""In-sample fidelity and split-conformal calibration of the world model's forecasts.

Out-of-sample point accuracy lives in eval/rolling.py. Here:
  * insample_report  -- how faithfully the fitted simulator reproduces history when run
                        free from the first year;
  * conformal_split  -- prediction intervals calibrated on VALIDATION-fold residuals
                        (cutoffs <= 2009, holdouts <= 2015) and checked for coverage on
                        the TEST folds (holdouts after 2015). No test data informs the
                        interval widths.
"""
from __future__ import annotations

import os

import numpy as np

from ..config import Config
from . import metrics as M
from .rolling import job_path


def insample_report(panel, cfg: Config, params, wm) -> dict:
    tr = wm.rollout(params, panel)
    m = panel.mask
    co2p, Dp, shp = np.asarray(tr["co2"]), np.asarray(tr["D"]), np.asarray(tr["shares"])
    return {
        "co2_r2": M.r2(co2p, panel.co2, m), "co2_mape": M.mape(co2p, panel.co2, m),
        "demand_r2": M.r2(Dp, panel.primary_energy, m),
        "demand_mape": M.mape(Dp, panel.primary_energy, m),
        "mix_mae": float(np.abs(shp - panel.shares())[m.astype(bool)].mean()),
    }


def _rel_residuals(panel, config, root, cutoff, seeds):
    """Relative residuals (pred-true)/true of the seed-mean forecast, [C, H]."""
    runs = [np.load(job_path(root, config, cutoff, s)) for s in seeds
            if os.path.exists(job_path(root, config, cutoff, s))]
    if not runs:
        return None, None
    fc = np.mean([r["co2_realized"] for r in runs], axis=0)[:, 1:]
    k = int(np.where(panel.years == cutoff)[0][0])
    H = fc.shape[1]
    true = panel.co2[:, k + 1:k + H + 1]
    return fc, (fc - true) / np.clip(true, 1e-6, None)


def conformal_split(panel, config: str, val_root: str, test_root: str, val_cutoffs,
                    test_cutoffs, seeds, alpha: float = 0.2) -> dict:
    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    by_h: dict[int, list] = {}
    for co in val_cutoffs:
        _, r = _rel_residuals(panel, config, val_root, co, seeds)
        if r is None:
            continue
        for h in range(r.shape[1]):
            by_h.setdefault(h + 1, []).append(r[:, h])
    q = {h: (float(np.quantile(np.concatenate(v), lo_q)), float(np.quantile(np.concatenate(v), hi_q)))
         for h, v in by_h.items()}
    hmax = max(q)
    hits, n = 0, 0
    per_cut = {}
    for co in test_cutoffs:
        fc, _ = _rel_residuals(panel, config, test_root, co, seeds)
        if fc is None:
            continue
        k = int(np.where(panel.years == co)[0][0])
        true = panel.co2[:, k + 1:k + fc.shape[1] + 1]
        cov = []
        for h in range(fc.shape[1]):
            ql, qh = q[min(h + 1, hmax)]           # horizons beyond the val range reuse the last
            lo, hi = fc[:, h] / (1 + qh), fc[:, h] / (1 + ql)
            inside = (true[:, h] >= lo) & (true[:, h] <= hi)
            hits += int(inside.sum()); n += inside.size
            cov.append(float(inside.mean()))
        per_cut[int(co)] = cov
    return {"nominal": 1 - alpha, "rel_resid_quantiles_by_horizon": q,
            "test_coverage": hits / max(n, 1), "test_coverage_by_cutoff_horizon": per_cut}
