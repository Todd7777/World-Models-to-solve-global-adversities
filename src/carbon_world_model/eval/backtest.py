"""Temporal-holdout backtest: train on <= cutoff, forecast the rest, vs baselines."""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..model.world_model import WorldModel
from ..train.fit import fit
from . import metrics as M
from .baselines import BASELINES, persistence


def fit_ensemble(panel, cfg: Config, cutoff_year: int | None, seeds: list[int], verbose=False):
    """Fit an ensemble of world models with different seeds (temporal holdout)."""
    wm = WorldModel.build(panel, cfg)
    params_list = []
    for s in seeds:
        res = fit(panel, cfg, wm=wm, train_year_max=cutoff_year, seed=s, verbose=verbose)
        params_list.append(res.params)
    return wm, params_list


def wm_forecast(wm, panel, params_list):
    """Return (mean_co2 [C,T], stack [S,C,T]) of ensemble CO2 forecasts."""
    stack = np.stack([np.asarray(wm.rollout(p, panel)["co2"]) for p in params_list], axis=0)
    return stack.mean(axis=0), stack


def insample_report(panel, cfg: Config, params, wm) -> dict:
    """Full-data reconstruction quality (the simulator's in-sample fidelity)."""
    tr = wm.rollout(params, panel)
    m = panel.mask
    co2p, Dp, shp = np.asarray(tr["co2"]), np.asarray(tr["D"]), np.asarray(tr["shares"])
    return {
        "co2_r2": M.r2(co2p, panel.co2, m),
        "co2_mape": M.mape(co2p, panel.co2, m),
        "demand_r2": M.r2(Dp, panel.primary_energy, m),
        "demand_mape": M.mape(Dp, panel.primary_energy, m),
        "mix_mae": float(np.abs(shp - panel.shares())[m.astype(bool)].mean()),
    }


def rolling_conformal(panel, cfg: Config, cutoffs, seeds=(0, 1), horizon=7, verbose=True):
    """Rolling-origin calibration: collect WM relative residuals by horizon, then
    report calibrated multiplicative 80% intervals and their leave-one-cutoff-out
    coverage. Seed-ensemble intervals badly understate uncertainty; these do not.
    """
    resid_by_h = {h: [] for h in range(1, horizon + 1)}
    per_cutoff = {}
    for co in cutoffs:
        wm, plist = fit_ensemble(panel, cfg, co, list(seeds))
        mean, _ = wm_forecast(wm, panel, plist)
        k = int(np.where(panel.years == co)[0][0])
        rel = []
        for h in range(1, horizon + 1):
            t = k + h
            if t >= panel.n_years:
                break
            r = (mean[:, t] - panel.co2[:, t]) / np.clip(panel.co2[:, t], 1e-6, None)
            resid_by_h[h].append(r)
            rel.append(r)
        per_cutoff[co] = np.concatenate(rel) if rel else np.array([])
        if verbose:
            print(f"[conformal] cutoff {co} done")

    # Calibrated interval half-widths per horizon (10th/90th pct of residuals).
    q = {h: (float(np.quantile(np.concatenate(v), 0.1)),
             float(np.quantile(np.concatenate(v), 0.9)))
         for h, v in resid_by_h.items() if v}
    # Leave-one-cutoff-out coverage check of the 80% band.
    covs = []
    for co in cutoffs:
        held = per_cutoff[co]
        if held.size == 0:
            continue
        others = np.concatenate([per_cutoff[c] for c in cutoffs if c != co and per_cutoff[c].size])
        lo, hi = np.quantile(others, 0.1), np.quantile(others, 0.9)
        covs.append(float(np.mean((held >= lo) & (held <= hi))))
    return {"rel_resid_q10_q90_by_horizon": q,
            "loco_coverage_80": float(np.mean(covs)) if covs else float("nan")}


def run_temporal_backtest(panel, cfg: Config, cutoff_year: int, seeds=(0, 1, 2, 3, 4),
                          verbose=True) -> dict:
    years = panel.years
    k = int(np.where(years == cutoff_year)[0][0])
    hmask = np.zeros_like(panel.mask)
    hmask[:, k + 1:] = panel.mask[:, k + 1:]  # holdout years only, observed
    H = panel.n_years - (k + 1)

    if verbose:
        print(f"[backtest] cutoff={cutoff_year}  holdout={years[k+1]}-{years[-1]} "
              f"({H} yrs)  seeds={list(seeds)}")

    # World model (ensemble).
    wm, params_list = fit_ensemble(panel, cfg, cutoff_year, list(seeds))
    wm_mean, wm_stack = wm_forecast(wm, panel, params_list)

    # Baselines.
    preds = {name: fn(panel, cutoff_year) for name, fn in BASELINES.items()}
    preds["world_model"] = wm_mean
    ref = preds["persistence"]

    true = panel.co2
    results = {"cutoff": cutoff_year, "holdout_years": [int(y) for y in years[k + 1:]],
               "methods": {}}
    for name, pred in preds.items():
        results["methods"][name] = {
            "mape": M.mape(pred, true, hmask),
            "smape": M.smape(pred, true, hmask),
            "rmse_mt": M.rmse(pred, true, hmask),
            "r2": M.r2(pred, true, hmask),
            "bias_mt": M.bias(pred, true, hmask),
            "skill_vs_persistence": M.skill(pred, ref, true, hmask),
            # error by forecast horizon (1..H years ahead), aggregate across countries
            "mape_by_horizon": [
                M.mape(pred[:, k + h], true[:, k + h], panel.mask[:, k + h])
                for h in range(1, H + 1)
            ],
        }

    # Aggregate (global sum across the 60 countries) forecast of total emissions.
    agg_true = true.sum(axis=0)
    results["aggregate"] = {"years": [int(y) for y in years], "true_mt": agg_true.tolist()}
    for name, pred in preds.items():
        agg = pred.sum(axis=0)
        err = (agg[k + 1:] - agg_true[k + 1:]) / agg_true[k + 1:]
        results["aggregate"].setdefault("pred_mt", {})[name] = agg.tolist()
        results["aggregate"].setdefault("final_year_pct_err", {})[name] = float(err[-1])

    # Ensemble calibration on holdout (central 80% interval from the seed spread).
    lo = np.quantile(wm_stack, 0.1, axis=0)
    hi = np.quantile(wm_stack, 0.9, axis=0)
    results["wm_calibration"] = {
        "interval": "p10-p90 (seed ensemble)",
        "coverage_80": M.coverage(lo, hi, true, hmask),
    }
    return results, preds, (wm_stack, wm)
