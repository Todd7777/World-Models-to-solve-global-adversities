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

import numpy as np

from ..config import Config
from . import metrics as M
from .configs import ProvenanceError, config_diff
from .rolling import load_jobs


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


def _rel_residuals(panel, runs, cutoff):
    """Seed-mean forecast and its relative residuals (pred-true)/true, both [C, H]."""
    fc = np.mean([r["co2_realized"] for r in runs], axis=0)[:, 1:]
    k = int(np.where(panel.years == cutoff)[0][0])
    H = fc.shape[1]
    true = panel.co2[:, k + 1:k + H + 1]
    return fc, (fc - true) / np.clip(true, 1e-6, None)


def conformal_split(panel, config: str, val_root: str, test_root: str, val_cutoffs,
                    test_cutoffs, seeds, alpha: float = 0.2, steps: int | None = None) -> dict:
    """Intervals calibrated on the validation outputs, coverage checked on the test outputs.
    Both must have been trained the same way apart from the seed (for `steps` steps, if
    given): intervals calibrated on one training run say nothing about another."""
    val_runs, val_fit = load_jobs(val_root, config, val_cutoffs, seeds, steps=steps)
    test_runs, test_fit = load_jobs(test_root, config, test_cutoffs, seeds, steps=steps)
    if val_fit and test_fit and config_diff(val_fit, test_fit):
        raise ProvenanceError(
            f"refusing to calibrate {config!r} on validation outputs ({val_root}) trained "
            f"differently from the test outputs ({test_root}): "
            + ", ".join(f"{k} {x!r} vs {y!r}" for k, x, y in config_diff(val_fit, test_fit)))
    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    by_h: dict[int, list] = {}
    for co in val_cutoffs:
        if int(co) not in val_runs:
            continue
        _, r = _rel_residuals(panel, val_runs[int(co)], co)
        for h in range(r.shape[1]):
            by_h.setdefault(h + 1, []).append(r[:, h])
    q = {h: (float(np.quantile(np.concatenate(v), lo_q)), float(np.quantile(np.concatenate(v), hi_q)))
         for h, v in by_h.items()}
    hmax = max(q)
    hits, n = 0, 0
    per_cut = {}
    for co in test_cutoffs:
        if int(co) not in test_runs:
            continue
        fc, _ = _rel_residuals(panel, test_runs[int(co)], co)
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
