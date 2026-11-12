"""Why the ensemble members disagree on what maximum effort achieves.

The ten training seeds of the selected configuration agree closely on business as
usual but cut 2050 emissions under maximum effort by anywhere from about a quarter to
almost three fifths (results/tables/plan.json). For each member, under the heuristic
and SSP2 drivers, this compares maximum effort in the planner's closed-loop rollouts, in
which the learned dynamics respond to the controlled state, with two counterfactuals
that keep the network on the business-as-usual path (plan/response.py): the open loop,
which adds the controls on top of the natural changes, and the frontier, which lifts
each country's total rates of change to the feasibility bounds. Both controls together
and each alone. Writes results/tables/control_response.json.

Refuses to run, before writing anything, unless the checkpoint record matches
default_config() and the ensemble was trained with it and contains the checkpoint.

Usage:  python scripts/control_response.py
"""
import json
import os
import sys

import numpy as np

from carbon_world_model.data.panel import Panel
from carbon_world_model.eval.configs import ProvenanceError, check_default_checkpoint
from carbon_world_model.model.world_model import WorldModel
from carbon_world_model.plan.ensemble import ENSEMBLE_DIR, load_ensemble, stack_members
from carbon_world_model.plan.response import LOOPS, PLANS, control_response
from carbon_world_model.plan.scenario import build_scenario

OUT = "results/tables/control_response.json"
CHECKPOINT = "results/model/params_full.pkl"
CHECKPOINT_CONFIG = "results/tables/config.json"  # written by 02_train.py with CHECKPOINT


def scenarios(panel, cfg):
    """The headline heuristic drivers and SSP2, the planning default."""
    return [build_scenario(panel, cfg, ssp=None), build_scenario(panel, cfg)]


def _range(xs, fmt="{:.0f}"):
    return f"{fmt.format(min(xs))}-{fmt.format(max(xs))}"


def main() -> None:
    cfg = check_default_checkpoint(CHECKPOINT_CONFIG)  # before anything is written
    seeds, members = load_ensemble(ENSEMBLE_DIR, cfg, CHECKPOINT)
    stacked = stack_members(members)
    panel = Panel.load(cfg)
    wm = WorldModel.build(panel, cfg)
    res = {"seeds": seeds,
           "phi": {"intensity": (1 / (1 + np.exp(-np.asarray(stacked["phi"][:, 0])))).tolist(),
                   "mix": (1 / (1 + np.exp(-np.asarray(stacked["phi"][:, 1])))).tolist()},
           "scenarios": {}}
    for scen in scenarios(panel, cfg):
        r = control_response(wm, stacked, scen)
        res["scenarios"][scen.name] = r
        print(f"\n{scen.name}: ranges over seeds {seeds}; bounds: intensity fall "
              f"{100 * scen.h_max:.1f}%/yr, low-carbon gain {100 * scen.g_max:.2f} points/yr")
        print(f"  BAU 2050: {_range([x / 1000 for x in r['bau_2050_mt']], '{:.1f}')} Gt; "
              f"intensity fall {_range(r['intensity_fall_pct_yr_bau'], '{:.1f}')}%/yr, "
              f"low-carbon gain {_range(r['lowc_gain_pp_yr_bau'], '{:.2f}')} points/yr")
        for plan in PLANS:
            print(f"  cut in 2050, {plan:10s}: " + ", ".join(
                f"{loop} {_range(r['cut_2050_pct'][loop][plan]):>6s}%" for loop in LOOPS))
        for loop in LOOPS:
            print(f"  maximum effort, {loop:8s}: intensity fall "
                  f"{_range(r['intensity_fall_pct_yr'][loop]['both'], '{:.1f}')}%/yr, "
                  f"low-carbon gain {_range(r['lowc_gain_pp_yr'][loop]['both'], '{:.2f}')} points/yr")
        print(f"  retention in the closed loop: efficiency "
              f"{_range(r['retention']['efficiency'], '{:.2f}')}, clean "
              f"{_range(r['retention']['clean'], '{:.2f}')}")
        print(f"  energy change from the clean control alone (closed loop): "
              f"{_range(r['energy_change_2050_pct']['closed']['clean'], '{:+.0f}')}%")
        for c, d in list(r["by_country"].items())[:4]:
            print(f"  {c:14s} max effort 2050, Mt: " + ", ".join(
                f"{loop} {_range(d[f'max_2050_{loop}_mt'])}" for loop in LOOPS)
                + f" (BAU {np.mean(d['bau_2050_mt']):,.0f})")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\n[ok] -> {OUT}")


if __name__ == "__main__":
    try:
        main()
    except ProvenanceError as e:
        sys.exit(f"control_response.py: {e}")
