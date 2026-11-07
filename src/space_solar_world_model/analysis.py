"""Reusable experiments: rankings, the terrestrial counterfactual, sweeps and decompositions."""
from __future__ import annotations

import math
import time

import numpy as np

from . import planner as PLN
from . import placements as PL
from .economics.ground import FirmResult, crf, firm_supply, space_hourly_profile
from .economics.system import shared_availability_series
from .sites import SITES
from .world import Action, SpaceSolarWorld

BEAM_PLACEMENTS = ["geo", "meo_20200_55", "meo_12000_55", "meo_8000_0", "leo_1500_53",
                   "leo_1000_53", "sso18_1000", "tundra", "molniya", "l1"]
TECHS = ["mw2.45", "mw5.8", "laser"]
SITE_OPTIONS = (4, 12, 24)


def _f(x):
    x = np.asarray(x)
    return float(x) if x.size == 1 else [float(v) for v in x.ravel()]


def plan_row(world: SpaceSolarWorld, r: PLN.PlanResult) -> dict:
    o, a = r.outcome, r.action
    p = world.placement(a.placement)
    row = dict(placement=a.placement, label=p.label, family=p.family, use=a.use, tech=a.tech,
               transfer=a.transfer, site=a.site, n_sites=a.n_sites if a.mode == "shared" else 1,
               mode=a.mode, lcoe=r.lcoe, design={k: float(v) for k, v in a.design})
    for k in ("capex", "capex_per_kw", "kg_per_kw", "M_dry", "M_batt", "P_delivered_bol",
              "link_efficiency", "R_eol", "transfer_days", "cost_per_kg_orbit", "fluence_rate",
              "debris_loss", "sk_dv_yr", "catastrophic_impacts_yr", "n_sats", "C_ground"):
        if k in o:
            row[k] = _f(o[k])
    if "A_r_ground" in o:
        row["rectenna_km2_max"] = float(np.max(o["A_r_ground"])) / 1e6
    return row


def rank_beam(world: SpaceSolarWorld, site: str, placements=BEAM_PLACEMENTS, techs=TECHS,
              site_options=SITE_OPTIONS, verbose: bool = False) -> list[dict]:
    rows = []
    for key in placements:
        for tech in techs:
            t0 = time.time()
            try:
                r = PLN.plan_beam(world, key, site, tech, site_options=site_options)
            except ValueError:
                continue
            row = plan_row(world, r)
            rows.append(row)
            if verbose:
                print(f"  {world.scenario.name:9s} {site:10s} {key:14s} {tech:7s} "
                      f"${row['lcoe']:9.0f}/MWh  [{time.time() - t0:.0f}s]", flush=True)
    rows.sort(key=lambda r: r["lcoe"])
    return rows


def best_per_placement(rows: list[dict]) -> list[dict]:
    best = {}
    for r in rows:
        if r["placement"] not in best or r["lcoe"] < best[r["placement"]]["lcoe"]:
            best[r["placement"]] = r
    return sorted(best.values(), key=lambda r: r["lcoe"])


def ground_firm(world: SpaceSolarWorld, site: str, reliability: float = 0.97) -> FirmResult:
    return firm_supply(SITES[site], world.central_u(), reliability)


def space_shape(world: SpaceSolarWorld, action: Action) -> np.ndarray:
    """Hourly delivery shape (mean 1) of a beaming action at its target site."""
    sl = world.site_link(action.placement, action.site)
    if action.mode == "shared":
        q = world.link_arrays(action)["q"]                       # target site is row 0
        a = shared_availability_series(np.float64(action.design_dict()["T"]), q, np)[0]
        mh = np.zeros((12, 24)); cnt = np.zeros((12, 24))
        np.add.at(mh, (sl.month, sl.hour), a)
        np.add.at(cnt, (sl.month, sl.hour), 1.0)
        mh = mh / np.maximum(cnt, 1.0)
    else:
        mh = sl.hourly_profile()
    prof = space_hourly_profile(mh)
    return prof / max(prof.mean(), 1e-12)


def hybrid_portfolio(world: SpaceSolarWorld, plan: PLN.PlanResult, reliability: float = 0.97) -> FirmResult:
    """Cheapest PV + battery + space mix for firm power at the plan's target site."""
    return firm_supply(SITES[plan.action.site], world.central_u(), reliability,
                       space_profile=space_shape(world, plan.action), space_lcoe=plan.lcoe)


def ground_firm_cost_samples(world: SpaceSolarWorld, firm: FirmResult, u: dict) -> np.ndarray:
    """Firm-ground LCOE in sampled worlds, holding the central-cost optimal mix fixed."""
    r = float(np.mean(u["ground_wacc"]))
    ann_pv = np.asarray(u["pv_ground_usd_w"]) * 1000.0 * (crf(r, 30) + 0.015)
    ann_b = np.asarray(u["batt_ground_usd_kwh"]) * (crf(r, 15) + 0.025)
    return (firm.pv_kw * ann_pv + firm.batt_kwh * ann_b) / (firm.served * 8.76)


def in_space_table(world: SpaceSolarWorld, keys=None) -> list[dict]:
    keys = list(world.catalog) if keys is None else keys
    rows = []
    for k in keys:
        f = PLN.in_space(world, k, firm=True)
        a = PLN.in_space(world, k, firm=False)
        row = plan_row(world, f)
        row.update(lcoe_firm=f.lcoe, lcoe_asavail=a.lcoe, latency_ms=world.placement(k).latency_ms,
                   sunlit_fraction=world.placement(k).illum.sunlit_fraction,
                   max_eclipse_min=world.placement(k).illum.max_eclipse_min)
        rows.append(row)
    rows.sort(key=lambda r: r["lcoe_firm"])
    return rows


def in_space_grid(world_by_scen: dict, alts, incs, dawn_dusk_alts) -> dict:
    """LCOE of in-orbit power over an altitude x inclination grid (+ dawn-dusk line)."""
    out = {name: dict(firm=np.full((len(incs), len(alts)), np.nan),
                      asavail=np.full((len(incs), len(alts)), np.nan),
                      dd_firm=np.full(len(dawn_dusk_alts), np.nan),
                      dd_asavail=np.full(len(dawn_dusk_alts), np.nan)) for name in world_by_scen}
    for j, alt in enumerate(alts):
        for i, inc in enumerate(incs):
            p = PL.circular(float(alt), float(inc))
            for name, w in world_by_scen.items():
                w.catalog[p.key] = p
                out[name]["firm"][i, j] = PLN.in_space(w, p.key, True).lcoe
                out[name]["asavail"][i, j] = PLN.in_space(w, p.key, False).lcoe
    for j, alt in enumerate(dawn_dusk_alts):
        p = PL.circular(float(alt), ltan_hours=18.0)
        for name, w in world_by_scen.items():
            w.catalog[p.key] = p
            out[name]["dd_firm"][j] = PLN.in_space(w, p.key, True).lcoe
            out[name]["dd_asavail"][j] = PLN.in_space(w, p.key, False).lcoe
    return out


MECHANISMS = {
    "no radiation damage": dict(do={"fluence_rate": 0.0, "transfer_fluence": 0.0}),
    "no eclipses": dict(do={"site_availability": 1.0, "worst_ecl": 0.0, "max_ecl_h": 0.0}),
    "no debris": dict(do={"debris_loss": 0.0}),
    "free launch & transfer": dict(do={"cost_per_kg_orbit": 0.0}),
    "perfect power conversion": dict(u={"eta_dcrf": 0.999, "eta_rfdc": 0.999, "eta_pmad": 0.999,
                                        "eta_grid": 0.999, "eta_laser": 0.999, "eta_laser_pv": 0.999}),
    "free space hardware": dict(u={"hw_usd_kg": 0.0}),
    "infinite hardware life (40 yr)": dict(u={"life_yr": 40.0}),
}


def mechanism_decomposition(world: SpaceSolarWorld, action: Action) -> list[dict]:
    """LCOE under single-mechanism interventions (structural do-operations)."""
    base = float(world.outcome(action)["lcoe"])
    rows = [dict(intervention="baseline", lcoe=base, change=0.0)]
    for name, spec in MECHANISMS.items():
        u = dict(world.central_u(), **spec.get("u", {}))
        v = float(world.outcome(action, u, do=spec.get("do"))["lcoe"])
        rows.append(dict(intervention=name, lcoe=v, change=v / base - 1.0))
    return rows


def sweep(world: SpaceSolarWorld, key: str, site: str, tech: str, param: str, values,
          site_options=(24,)) -> list[dict]:
    """Re-plan the design at each value of an exogenous parameter (do(param = value))."""
    rows = []
    for v in values:
        w = world.with_scenario(world.scenario.with_(**{param: float(v)}))
        r = PLN.plan_beam(w, key, site, tech, site_options=site_options)
        rows.append(dict(param=param, value=float(v), lcoe=r.lcoe, placement=key, tech=tech,
                         n_sites=r.action.n_sites))
    return rows


def breakeven(values, lcoes, target):
    """Parameter value at which LCOE crosses ``target`` (log-log interpolation)."""
    v, l = np.log(np.asarray(values, float)), np.log(np.asarray(lcoes, float))
    o = np.argsort(v)
    v, l = v[o], l[o]
    for i in range(len(v) - 1):
        if (l[i] - math.log(target)) * (l[i + 1] - math.log(target)) <= 0:
            t = (math.log(target) - l[i]) / (l[i + 1] - l[i])
            return float(math.exp(v[i] + t * (v[i + 1] - v[i])))
    return None
