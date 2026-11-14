"""How far maximum effort gets when the frontier's assumptions are relaxed.

At the frontier (plan/optimize.py) the planner assumes that planned change persists in
full and that every country can run at the feasibility limits for the whole horizon.
This evaluates maximum effort under every member of the planning ensemble, with the
heuristic and SSP2 drivers, when

  - a share of the planned efficiency gain is taken back by higher energy use (rebound),
  - a share of the planned low-carbon gain displaces business as usual's own growth
    (crowding out),
  - the limits are scaled down, since each is a rate that individual countries
    sustained for a decade, not all countries at once for 28 years,

one at a time and together. Effort is charged on the plan in every case. Writes
results/tables/plan_sensitivity.json.

With --costs it also re-optimises, for a few of the settings, what 04_plan.py optimises
under the stated assumptions: the SSP2 cost-of-abatement frontier at the budgets in
results/tables/plan.json (a budget even maximum effort misses on average is infeasible),
and robust planning with 04_plan.py's own rule for a demanding but feasible budget. It
takes about 45 minutes and writes results/tables/plan_sensitivity_costs.json.

Refuses to run, before writing anything, unless the checkpoint record matches
default_config() and the ensemble was trained with it and contains the checkpoint.

Usage:  python scripts/plan_sensitivity.py [--costs]
"""
import argparse
import dataclasses
import json
import os
import sys

import numpy as np

from carbon_world_model.climate import tcre
from carbon_world_model.data.panel import Panel
from carbon_world_model.data.ssp import SSPS
from carbon_world_model.eval.configs import ProvenanceError, check_default_checkpoint
from carbon_world_model.model.world_model import WorldModel
from carbon_world_model.plan import optimize as O
from carbon_world_model.plan.ensemble import ENSEMBLE_DIR, load_ensemble, stack_members
from carbon_world_model.plan.scenario import build_scenario, ssp_ensemble

OUT = "results/tables/plan_sensitivity.json"
COSTS_OUT = "results/tables/plan_sensitivity_costs.json"
PLAN = "results/tables/plan.json"  # 04_plan.py's frontier budgets
CHECKPOINT = "results/model/params_full.pkl"
CHECKPOINT_CONFIG = "results/tables/config.json"  # written by 02_train.py with CHECKPOINT

# (rebound, crowding out, scale of both feasibility limits)
SETTINGS = {
    "as stated": (0.0, 0.0, 1.0),
    "rebound 25%": (0.25, 0.0, 1.0),
    "rebound 50%": (0.5, 0.0, 1.0),
    "crowding out 25%": (0.0, 0.25, 1.0),
    "crowding out 50%": (0.0, 0.5, 1.0),
    "limits x0.75": (0.0, 0.0, 0.75),
    "limits x0.5": (0.0, 0.0, 0.5),
    "all three moderate": (0.25, 0.25, 0.75),
}
COST_SETTINGS = ("as stated", "rebound 25%", "crowding out 25%", "limits x0.75",
                 "all three moderate")


def scenarios(panel, cfg):
    """The headline heuristic drivers and SSP2, the planning default."""
    return [build_scenario(panel, cfg, ssp=None), build_scenario(panel, cfg)]


def relaxed(scen, scale: float):
    """The scenario with both feasibility limits scaled by `scale`."""
    return dataclasses.replace(scen, g_max=scale * scen.g_max, h_max=scale * scen.h_max)


def max_effort(wm, stacked, scen, rebound: float, crowd_out: float, scale: float) -> dict:
    """Maximum effort under every member, [K] arrays: 60-country emissions in 2050 (Mt)
    and cumulative over the horizon (Gt), with business as usual for reference."""
    W = O.make_worlds(wm, stacked, [relaxed(scen, scale)], rebound=rebound, crowd_out=crowd_out)
    bau, mx = O.evaluate_worlds(W), O.evaluate_worlds(W, *O.max_plan(W))
    return {"bau_2050_mt": bau["final_mt"], "max_2050_mt": mx["final_mt"],
            "bau_cum_gt": bau["cum_gt"], "max_cum_gt": mx["cum_gt"]}


def summarise(r: dict, clim) -> dict:
    """Central values as in 04_plan.py (the cut of the mean pathways, cumulative totals
    of the mean), and every member's values."""
    b, m = r["bau_2050_mt"].astype(float), r["max_2050_mt"].astype(float)
    bc, mc = r["bau_cum_gt"].astype(float), r["max_cum_gt"].astype(float)
    return {
        "cut_2050_pct": float(100 * (1 - m.mean() / b.mean())),
        "max_2050_mt": float(m.mean()),
        "global_cum_max_gt": tcre.scale_to_global(float(mc.mean())),
        "avoided_warming_degC": tcre.avoided_warming(float(bc.mean()), float(mc.mean()), clim),
        "by_seed": {"cut_2050_pct": (100 * (1 - m / b)).tolist(), "max_2050_mt": m.tolist(),
                    "global_cum_max_gt": [tcre.scale_to_global(float(x)) for x in mc]},
    }


def _cum(ev) -> np.ndarray:
    """[W] cumulative 60-country emissions over the horizon (Gt), accumulated as
    04_plan.py accumulates them."""
    return ev["global_mt"].astype(float)[:, 1:].sum(axis=1) / 1000.0


def frontier_costs(wm, stacked, ssp2, budgets, rebound: float, crowd_out: float,
                   scale: float, steps: int = O.STEPS) -> dict:
    """04_plan.py's SSP2 frontier under one setting: the cheapest plan whose
    ensemble-mean cumulative emissions meet each budget. A budget at or below maximum
    effort's ensemble mean is infeasible and is not optimised."""
    W = O.make_worlds(wm, stacked, [relaxed(ssp2, scale)], rebound=rebound, crowd_out=crowd_out)
    floor = float(_cum(O.evaluate_worlds(W, *O.max_plan(W))).mean())
    points = []
    for b in budgets:
        if b <= floor:
            points.append({"budget_gt": float(b), "feasible": False, "cost": None, "cum_gt": None})
            continue
        g, h = O.plan_against(W, ssp2, budget_gt=float(b), mode="mean", steps=steps)
        points.append({"budget_gt": float(b), "feasible": True, "cost": float(O._cost(g, h, ssp2)),
                       "cum_gt": float(_cum(O.evaluate_worlds(W, g, h)).mean())})
    return {"max_effort_cum_gt": floor, "points": points}


def robust_price(wm, stacked, ssp_scens, rebound: float, crowd_out: float, scale: float,
                 steps: int = O.STEPS) -> dict:
    """04_plan.py's robust planning under one setting. The budget lies a quarter of the
    way from the worst SSP and member under maximum effort to the best business as usual
    (or 2% above the former if they do not overlap). The tailored plan meets it on
    average under SSP2, the robust plan under every SSP and member."""
    scens = [relaxed(s, scale) for s in ssp_scens]
    ssp2 = scens[SSPS.index("SSP2")]
    W5 = O.make_worlds(wm, stacked, scens, rebound=rebound, crowd_out=crowd_out)
    W2 = O.make_worlds(wm, stacked, [ssp2], rebound=rebound, crowd_out=crowd_out)
    worst_max = float(_cum(O.evaluate_worlds(W5, *O.max_plan(W5))).max())
    best_bau = float(_cum(O.evaluate_worlds(W5)).min())
    budget = (worst_max + 0.25 * (best_bau - worst_max) if best_bau > worst_max
              else 1.02 * worst_max)
    tg, th = O.plan_against(W2, ssp2, budget_gt=budget, mode="mean", steps=steps)
    rg, rh = O.plan_against(W5, scens[0], budget_gt=budget, mode="all", steps=steps)
    t_cost, r_cost = float(O._cost(tg, th, ssp2)), float(O._cost(rg, rh, scens[0]))
    t_cum, r_cum = _cum(O.evaluate_worlds(W5, tg, th)), _cum(O.evaluate_worlds(W5, rg, rh))
    return {"budget_gt": budget, "worst_max_effort_cum_gt": worst_max,
            "tailored_cost": t_cost, "robust_cost": r_cost,
            "price_of_robustness_pct": 100 * (r_cost / t_cost - 1),
            "tailored_worst_overshoot_gt": max(0.0, float(t_cum.max()) - budget),
            "robust_worst_cum_gt": float(r_cum.max())}


def costs(wm, stacked, panel, cfg, seeds) -> dict:
    with open(PLAN) as fh:
        budgets = [p["budget_gt"] for p in json.load(fh)["frontier_ssp2"]]
    ssp2 = build_scenario(panel, cfg)
    ssp_scens = ssp_ensemble(panel, cfg)
    res = {"seeds": seeds, "steps": O.STEPS, "frontier_budgets_gt": budgets, "settings": {}}
    for name in COST_SETTINGS:
        rebound, crowd, scale = SETTINGS[name]
        fr = frontier_costs(wm, stacked, ssp2, budgets, rebound, crowd, scale)
        rb = robust_price(wm, stacked, ssp_scens, rebound, crowd, scale)
        res["settings"][name] = {"rebound": rebound, "crowd_out": crowd, "limit_scale": scale,
                                 "frontier_ssp2": fr, "robust": rb}
        print(f"  {name:20s} frontier costs: " + ", ".join(
            f"{p['budget_gt']:.0f}:{p['cost']:.3f}" if p["feasible"] else f"{p['budget_gt']:.0f}:-"
            for p in fr["points"])
            + f" | robust budget {rb['budget_gt']:.0f} Gt: +{rb['price_of_robustness_pct']:.0f}% "
            f"(tailored {rb['tailored_cost']:.3f}, robust {rb['robust_cost']:.3f}, tailored "
            f"overshoot {rb['tailored_worst_overshoot_gt']:.0f} Gt)", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--costs", action="store_true",
                    help="also re-optimise the frontier and robust plans (about 45 minutes)")
    a = ap.parse_args()
    cfg = check_default_checkpoint(CHECKPOINT_CONFIG)  # before anything is written
    if a.costs and not os.path.exists(PLAN):
        sys.exit(f"plan_sensitivity.py: {PLAN} not found; run scripts/04_plan.py first")
    seeds, members = load_ensemble(ENSEMBLE_DIR, cfg, CHECKPOINT)
    stacked = stack_members(members)
    panel = Panel.load(cfg)
    wm = WorldModel.build(panel, cfg)
    clim = cfg.climate
    budget = tcre.remaining_budget_gtco2(1.5, clim)
    res = {"seeds": seeds, "budget_1p5_gt": budget, "settings": {}}
    scens = scenarios(panel, cfg)
    for name, (rebound, crowd, scale) in SETTINGS.items():
        rec = {"rebound": rebound, "crowd_out": crowd, "limit_scale": scale, "scenarios": {}}
        for scen in scens:
            rec["scenarios"][scen.name] = summarise(
                max_effort(wm, stacked, scen, rebound, crowd, scale), clim)
        res["settings"][name] = rec
    print(f"maximum effort in 2050 under {len(seeds)} seeds: cut versus business as usual "
          f"(seed range); global cumulative 2023-2050 (1.5 C budget {budget:.0f} Gt)")
    for name, rec in res["settings"].items():
        row = []
        for sname, d in rec["scenarios"].items():
            cuts, cums = d["by_seed"]["cut_2050_pct"], d["by_seed"]["global_cum_max_gt"]
            row.append(f"{sname.split(' ')[0]:9s} {d['cut_2050_pct']:3.0f}% ({min(cuts):2.0f}-"
                       f"{max(cuts):2.0f}), {d['global_cum_max_gt']:4.0f} Gt "
                       f"({min(cums):.0f}-{max(cums):.0f})")
        print(f"  {name:20s} " + " | ".join(row))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[ok] -> {OUT}")
    if a.costs:
        print("frontier (SSP2, ensemble mean) and robust planning under relaxed assumptions:")
        out = costs(wm, stacked, panel, cfg, seeds)
        with open(COSTS_OUT, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"[ok] -> {COSTS_OUT}")


if __name__ == "__main__":
    try:
        main()
    except ProvenanceError as e:
        sys.exit(f"plan_sensitivity.py: {e}")
