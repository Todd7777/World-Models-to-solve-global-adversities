"""The terrestrial counterfactual: firm power from ground solar + batteries at the same site.

Space solar's selling point is firm (round-the-clock) power, so the fair comparison is
not raw ground-PV LCOE but the cheapest *firm* supply at the receiving site. We simulate
one year hour by hour:

* clear-sky irradiance (Haurwitz) x a daily clear-sky index drawn from a Beta
  distribution with AR(1) day-to-day persistence, calibrated so the annual total matches
  the site's GHI;
* beam/diffuse split by the Erbs correlation; single-axis (N-S) tracker, +/-60 deg;
* DC/AC ratio 1.25, 14% DC losses; 4-hour batteries, 90% round trip;
* a constant load served from PV, optional space-solar delivery, and the battery.

The cheapest mix meeting a reliability target (fraction of load energy served, 97% by
default, matching Ember's "97% of the way to 24/365" benchmark) is found by vectorised
grid search over (PV, battery, space capacity).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as _beta, norm as _norm

from ..constants import DEG
from ..sites import Site

HOURS = 8760
TSI = 1361.0


def crf(r: float, n: float) -> float:
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def _solar_geometry(lat_deg: float):
    """cos(zenith), sun unit vector (E, N, Up) and extraterrestrial normal irradiance per hour."""
    h = np.arange(HOURS) + 0.5
    doy = h / 24.0
    decl = 23.44 * DEG * np.sin(2 * np.pi * (doy - 80.0) / 365.0)
    ha = (np.mod(h, 24.0) - 12.0) * 15.0 * DEG          # local apparent solar time
    lat = lat_deg * DEG
    up = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    east = -np.cos(decl) * np.sin(ha)
    north = np.cos(lat) * np.sin(decl) - np.sin(lat) * np.cos(decl) * np.cos(ha)
    e0 = TSI * (1 + 0.033 * np.cos(2 * np.pi * doy / 365.0))
    return up, east, north, e0


def clear_sky_ghi(cos_z):
    cz = np.maximum(cos_z, 0.0)
    return np.where(cz > 0.01, 1098.0 * cz * np.exp(-0.057 / np.maximum(cz, 0.01)), 0.0)


def daily_clear_index(site: Site, seed: int = 0, rho: float = 0.6, kappa: float = 6.0):
    """Daily clear-sky index with the site's mean, Beta marginal, AR(1) persistence."""
    up, _, _, _ = _solar_geometry(site.lat)
    cs = clear_sky_ghi(up)
    annual_cs = cs.sum() / 1000.0                          # kWh/m^2
    m = float(np.clip(site.ghi / annual_cs, 0.15, 0.97))
    rng = np.random.default_rng(seed)
    z = np.zeros(365)
    eps = rng.standard_normal(365)
    for d in range(1, 365):
        z[d] = rho * z[d - 1] + np.sqrt(1 - rho ** 2) * eps[d]
    k = _beta.ppf(_norm.cdf(z), m * kappa, (1 - m) * kappa)
    k = k * m / k.mean()                                   # exact annual calibration
    return np.clip(k, 0.03, 1.05), cs


def pv_profile(site: Site, seed: int = 0, dc_ac: float = 1.25, dc_loss: float = 0.14,
               max_rot_deg: float = 60.0):
    """Hourly AC output per kW_dc of a single-axis tracking plant [kW/kW_dc]."""
    k_day, cs = daily_clear_index(site, seed)
    up, east, north, e0 = _solar_geometry(site.lat)
    ghi = cs * np.repeat(k_day, 24)
    cz = np.maximum(up, 1e-3)
    kt = np.clip(ghi / (e0 * cz), 0, 1)
    kd = np.where(kt <= 0.22, 1 - 0.09 * kt,
                  np.where(kt <= 0.8, 0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3
                           + 12.336 * kt ** 4, 0.165))
    dhi = kd * ghi
    dni = np.where(up > 0.01, (ghi - dhi) / cz, 0.0)
    rot = np.clip(np.arctan2(east, np.maximum(up, 1e-6)), -max_rot_deg * DEG, max_rot_deg * DEG)
    n_e, n_u = np.sin(rot), np.cos(rot)                    # panel normal in the E-Up plane
    cos_i = np.maximum(east * n_e + up * n_u, 0.0)
    poa = dni * cos_i + dhi * (1 + n_u) / 2 + ghi * 0.2 * (1 - n_u) / 2
    poa = np.where(up > 0, poa, 0.0)
    dc = poa / 1000.0 * (1 - dc_loss)
    return np.minimum(dc * 0.98, 1.0 / dc_ac), k_day


def dispatch(pv: np.ndarray, space: np.ndarray | None, pv_kw, batt_kwh, space_kw,
             load: float = 1.0, rt: float = 0.90, duration_h: float = 4.0):
    """Hourly dispatch for many configurations at once; returns served fraction of load."""
    pv_kw, batt_kwh, space_kw = (np.atleast_1d(np.asarray(x, float)) for x in (pv_kw, batt_kwh, space_kw))
    n = np.broadcast(pv_kw, batt_kwh, space_kw).shape
    pv_kw, batt_kwh, space_kw = (np.broadcast_to(x, n) for x in (pv_kw, batt_kwh, space_kw))
    eta = np.sqrt(rt)
    p_b = batt_kwh / duration_h
    soc = 0.5 * batt_kwh
    served = np.zeros(n)
    for yr in range(2):                                    # second pass removes start-up bias
        served[:] = 0.0
        for h in range(HOURS):
            gen = pv_kw * pv[h] + (space_kw * space[h] if space is not None else 0.0)
            net = gen - load
            ch = np.minimum(np.minimum(np.maximum(net, 0.0), p_b), (batt_kwh - soc) / eta)
            dis = np.minimum(np.minimum(np.maximum(-net, 0.0), p_b), soc * eta)
            soc = soc + ch * eta - dis / eta
            served += np.minimum(gen, load) + dis
    return served / (load * HOURS)


@dataclass
class FirmResult:
    lcoe: float                 # $/MWh of load served
    pv_kw: float                # per kW of load
    batt_kwh: float
    space_kw: float
    served: float
    pv_only_lcoe: float         # unfirmed PV LCOE at the site [$ / MWh generated]
    pv_yield: float             # kWh per kW_dc per year


def firm_supply(site: Site, u: dict, reliability: float = 0.97, space_profile: np.ndarray | None = None,
                space_lcoe: float | None = None, seed: int = 0, pv_grid=None, batt_grid=None,
                space_grid=None) -> FirmResult:
    """Cheapest PV + battery (+ optional space capacity) mix serving a 1 kW constant load.

    Space capacity is bought at ``space_lcoe`` [$ / MWh delivered], i.e. the levelised
    cost of the space system's own delivered energy, curtailed or not.
    """
    pv, _ = pv_profile(site, seed)
    r = float(u.get("ground_wacc", 0.07))
    pv_cost = float(u["pv_ground_usd_w"]) * 1000.0          # $/kW_dc
    b_cost = float(u["batt_ground_usd_kwh"])                # $/kWh
    ann_pv = pv_cost * (crf(r, 30) + 0.015)                 # $/kW_dc/yr incl. O&M
    ann_b = b_cost * (crf(r, 15) + 0.025)                   # $/kWh/yr incl. O&M
    yield_kwh = pv.sum()
    if pv_grid is None:
        pv_grid = np.concatenate([np.linspace(0, 14, 29), [16, 18, 21, 25, 30, 36, 44, 55]])
    if batt_grid is None:
        batt_grid = np.concatenate([np.linspace(0, 40, 41), [45, 50, 60, 70, 85, 100, 130, 170,
                                                            220, 300, 400]])
    if space_profile is None:
        space_grid = np.array([0.0])
        ann_s = 0.0
    else:
        space_grid = np.linspace(0, 1.3, 14) if space_grid is None else space_grid
        ann_s = float(space_lcoe) * space_profile.sum() / 1000.0     # $/kW_space/yr
    P, B, S = np.meshgrid(pv_grid, batt_grid, space_grid, indexing="ij")
    P, B, S = P.ravel(), B.ravel(), S.ravel()
    served = dispatch(pv, space_profile, P, B, S)
    cost = P * ann_pv + B * ann_b + S * ann_s
    ok = served >= reliability
    if not ok.any():                                       # infeasible within the grid
        return FirmResult(float("inf"), float("nan"), float("nan"), float("nan"),
                          float(served.max()), float(ann_pv / yield_kwh * 1000.0), float(yield_kwh))
    i = int(np.argmin(np.where(ok, cost, np.inf)))
    lcoe = cost[i] / (served[i] * HOURS / 1000.0)
    return FirmResult(float(lcoe), float(P[i]), float(B[i]), float(S[i]), float(served[i]),
                      float(ann_pv / yield_kwh * 1000.0), float(yield_kwh))


def space_hourly_profile(month_hour: np.ndarray) -> np.ndarray:
    """Tile a [12, 24] mean profile (by month and local solar hour) into 8760 hours."""
    h = np.arange(HOURS)
    month = np.minimum((h / 24.0 / 30.4375).astype(int), 11)
    return month_hour[month, h % 24]
