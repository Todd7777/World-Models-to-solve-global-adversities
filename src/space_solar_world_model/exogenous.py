"""Exogenous variables U of the structural causal model: central values and priors."""
from __future__ import annotations

import numpy as np

from .scenario import MULTIPLIERS, SIGMA, Scenario

MODEL_KEYS = (
    "launch_usd_kg", "isp_ep", "ep_tug_overhead", "ep_accel", "hw_usd_kg", "pv_w_kg", "pv_eff",
    "pmad_kg_kw", "tx_kg_kw", "ant_kg_m2", "laser_kg_kw", "struct_frac", "battery_wh_kg",
    "eta_pmad", "eta_dcrf", "eta_rfdc", "eta_laser", "eta_laser_pv", "eta_grid", "eta_batt_rt",
    "rect_usd_m2", "grid_usd_kw", "laser_rx_usd_kw", "site_fixed_usd", "om_frac", "life_yr",
    "discount", "availability", "base_degr", "module_m2", "pv_ground_usd_w",
    "batt_ground_usd_kwh", "ground_wacc",
)


def central(scen: Scenario) -> dict:
    """Central (median) world: scenario values, physics multipliers at 1."""
    d = {k: float(getattr(scen, k)) for k in MODEL_KEYS}
    d.update({m: 1.0 for m in MULTIPLIERS})
    return d


def _logit(p):
    return np.log(p / (1 - p))


def sample(scen: Scenario, n: int, seed: int = 0, fixed: dict | None = None) -> dict:
    """Draw ``n`` worlds from the prior. ``fixed`` pins variables (e.g. a do() on U)."""
    rng = np.random.default_rng(seed)
    c = central(scen)
    out = {}
    for k, v in c.items():
        if k in SIGMA:
            kind, s = SIGMA[k]
            z = rng.standard_normal(n)
            if kind == "ln":
                out[k] = v * np.exp(s * z)
            elif kind == "logit":
                out[k] = 1.0 / (1.0 + np.exp(-(_logit(v) + s * z)))
            else:
                out[k] = np.maximum(v + s * z, 3.0)
        else:
            out[k] = np.full(n, v)
    for k, v in (fixed or {}).items():
        out[k] = np.full(n, float(v))
    return out


def index(u: dict, idx) -> dict:
    return {k: np.asarray(v)[idx] for k, v in u.items()}
