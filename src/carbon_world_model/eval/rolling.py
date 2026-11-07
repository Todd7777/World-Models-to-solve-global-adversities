"""Rolling-origin evaluation with a strict validation / test split.

Protocol
--------
* Every fit uses only data up to its cutoff year (the world model is built with
  ``upto_year=cutoff``: emission factors, normalisation and static features included).
* Every forecast starts from the OBSERVED state in the cutoff year -- exactly what the
  baselines do -- and rolls forward. Process CO2 is held at its cutoff value.
* VALIDATION cutoffs (2005, 2007, 2009; 6-year horizons ending <= 2015) are the only
  ones used to choose a configuration. TEST cutoffs (2015, 2016, 2017; holdouts after
  2015) are scored once, for reporting.
* Drivers: 'realized' gives every method the realised GDP/population path (a
  conditional forecast); 'ssp' gives every method an SSP path harmonised at the cutoff
  (an ex-ante forecast, using the 2014-vintage SSP database for 2015-2017 cutoffs).
"""
from __future__ import annotations

import dataclasses
import json
import os

import numpy as np

from ..config import Config
from ..data.ssp import load_processed, ssp_driver_paths
from .baselines import BASELINES
from . import metrics as M

VAL_CUTOFFS = (2005, 2007, 2009)
VAL_HORIZON = 6
TEST_CUTOFFS = (2015, 2016, 2017)
TEST_HORIZON = 7


def horizon_for(panel, cutoff: int, horizon: int) -> int:
    return int(min(horizon, int(panel.years[-1]) - cutoff))


def job_path(root: str, config: str, cutoff: int, seed: int) -> str:
    return os.path.join(root, config, f"c{cutoff}_s{seed}.npz")


def run_job(panel, cfg: Config, cutoff: int, seed: int, horizon: int,
            ssp_vintage: str | None = None, ssps=("SSP2",)) -> dict:
    """Fit on data <= cutoff and forecast `horizon` years from the observed state."""
    from ..train.fit import fit  # local import keeps module import light
    H = horizon_for(panel, cutoff, horizon)
    r = fit(panel, cfg, train_year_max=cutoff, seed=seed, verbose=False)
    fc = r.wm.forecast(r.params, panel, cutoff, end_year=cutoff + H)
    out = {"co2_realized": np.asarray(fc["co2"]), "D_realized": np.asarray(fc["D"]),
           "shares_realized": np.asarray(fc["shares"]), "H": np.array(H),
           "final_loss": np.array(r.history[-1]["loss"])}
    if ssp_vintage:
        series = load_processed(ssp_vintage)
        years = np.arange(cutoff, cutoff + H + 1)
        for s in ssps:
            g, p = ssp_driver_paths(panel, series, s, cutoff, years)
            fcs = r.wm.forecast(r.params, panel, cutoff, gdp=g, pop=p)
            out[f"co2_{s}"] = np.asarray(fcs["co2"])
    return out


def panel_with_drivers(panel, cutoff: int, gdp_path: np.ndarray, pop_path: np.ndarray):
    """Copy of the panel whose GDP/population after `cutoff` follow the given paths."""
    k = int(np.where(panel.years == cutoff)[0][0])
    H = gdp_path.shape[1] - 1
    g, p = panel.gdp.copy(), panel.population.copy()
    g[:, k:k + H + 1] = gdp_path
    p[:, k:k + H + 1] = pop_path
    return dataclasses.replace(panel, gdp=g, population=p)


def _metrics(pred: np.ndarray, true: np.ndarray, mask: np.ndarray, china: int) -> dict:
    """pred/true/mask: [C, H] over holdout years (index 0 = first year after cutoff)."""
    agg_p, agg_t = (pred * mask).sum(0), (true * mask).sum(0)
    agg_err = (agg_p - agg_t) / agg_t
    return {
        "country_mape": M.mape(pred, true, mask),
        "rmse_mt": M.rmse(pred, true, mask),
        "agg_err_final": float(agg_err[-1]),
        "agg_abs_err_mean": float(np.mean(np.abs(agg_err))),
        "china_err_final": float((pred[china, -1] - true[china, -1]) / true[china, -1]),
        "mape_by_horizon": [M.mape(pred[:, h], true[:, h], mask[:, h])
                            for h in range(pred.shape[1])],
    }


def evaluate(panel, config: str, cutoffs, seeds, root: str, driver: str = "realized",
             ssp: str = "SSP2", ssp_vintage: str = "v9") -> dict:
    """Score the seed-ensemble world model and all baselines at each cutoff."""
    china = panel.countries.index("China")
    res = {"config": config, "driver": driver, "cutoffs": {}}
    for co in cutoffs:
        runs = [np.load(job_path(root, config, co, s)) for s in seeds
                if os.path.exists(job_path(root, config, co, s))]
        if not runs:
            continue
        key = "co2_realized" if driver == "realized" else f"co2_{ssp}"
        H = int(runs[0]["H"])
        wm_fc = np.mean([r[key] for r in runs], axis=0)       # [C, H+1]
        k = int(np.where(panel.years == co)[0][0])
        true = panel.co2[:, k + 1:k + H + 1]
        mask = panel.mask[:, k + 1:k + H + 1]
        bpanel = panel
        if driver != "realized":
            series = load_processed(ssp_vintage)
            g, p = ssp_driver_paths(panel, series, ssp, co, np.arange(co, co + H + 1))
            bpanel = panel_with_drivers(panel, co, g, p)
        row = {"world_model": _metrics(wm_fc[:, 1:], true, mask, china), "n_seeds": len(runs)}
        for name, fn in BASELINES.items():
            row[name] = _metrics(fn(bpanel, co)[:, k + 1:k + H + 1], true, mask, china)
        res["cutoffs"][int(co)] = row
    return res


def summarise(res: dict, methods=None) -> dict:
    """Average metrics over cutoffs, per method; adds a selection score."""
    cuts = list(res["cutoffs"].values())
    if not cuts:
        return {}
    methods = methods or [m for m in cuts[0] if m != "n_seeds"]
    out = {}
    for m in methods:
        keys = ("country_mape", "rmse_mt", "agg_err_final", "agg_abs_err_mean", "china_err_final")
        agg = {k: float(np.mean([c[m][k] for c in cuts])) for k in keys}
        agg["abs_china_err_final"] = float(np.mean([abs(c[m]["china_err_final"]) for c in cuts]))
        agg["score"] = 100 * (agg["country_mape"] + agg["agg_abs_err_mean"])
        out[m] = agg
    return out


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
