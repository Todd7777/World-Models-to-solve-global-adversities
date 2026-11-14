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

one at a time and together. Effort is charged on the plan in every case, so these only
change what maximum effort achieves. Writes results/tables/plan_sensitivity.json.

Refuses to run, before writing anything, unless the checkpoint record matches
default_config() and the ensemble was trained with it and contains the checkpoint.

Usage:  python scripts/plan_sensitivity.py
"""
import dataclasses
import json
import os
import sys

import numpy as np

from carbon_world_model.climate import tcre
from carbon_world_model.data.panel import Panel
from carbon_world_model.eval.configs import ProvenanceError, check_default_checkpoint
from carbon_world_model.model.world_model import WorldModel
from carbon_world_model.plan import optimize as O
from carbon_world_model.plan.ensemble import ENSEMBLE_DIR, load_ensemble, stack_members
from carbon_world_model.plan.scenario import build_scenario

OUT = "results/tables/plan_sensitivity.json"
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


def scenarios(panel, cfg):
    """The headline heuristic drivers and SSP2, the planning default."""
    return [build_scenario(panel, cfg, ssp=None), build_scenario(panel, cfg)]


def max_effort(wm, stacked, scen, rebound: float, crowd_out: float, scale: float) -> dict:
    """Maximum effort under every member, [K] arrays: 60-country emissions in 2050 (Mt)
    and cumulative over the horizon (Gt), with business as usual for reference."""
    scaled = dataclasses.replace(scen, g_max=scale * scen.g_max, h_max=scale * scen.h_max)
    W = O.make_worlds(wm, stacked, [scaled], rebound=rebound, crowd_out=crowd_out)
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


def main() -> None:
    cfg = check_default_checkpoint(CHECKPOINT_CONFIG)  # before anything is written
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


if __name__ == "__main__":
    try:
        main()
    except ProvenanceError as e:
        sys.exit(f"plan_sensitivity.py: {e}")
