"""Let the planner choose removals: abatement effort vs removal spending.

For each stabilised validation-tied model (results/tables/structural.json), SSP2 drivers to
2100, fitted dynamics and open loop, three budgets (2C at 50% and 67%, 1.7C at 50%;
Forster et al. 2025, from 2023) and a grid of removal spending allowances (USD trillion,
2023-2100, undiscounted), plan/cdr_planning.plan_with_removals finds the least abatement
effort that keeps PEAK cumulative net CO2 within the budget with net CO2 <= 0 in 2100.
Rules (supply, energy limit, accounting, what counts as meeting the budget) are fixed in
that module's docstring.

Each plan is made with the model as fitted. It is then checked against the model's own
long-backtest emission errors (44 per configuration, frozen at 2050 as in
scripts/12_to_2100.py; removals held fixed in GtCO2): the share of error cases in which it
still meets the budget, and the likelihood of staying below the target. Driver errors are
not included here.

Usage:  python scripts/13_cdr_planning.py
Writes results/tables/cdr_planning.json.
"""
import json
import os
import time

import numpy as np

from carbon_world_model.climate import budgets as B
from carbon_world_model.config import default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.eval.configs import SELECTED
from carbon_world_model.eval.propagate import backtest_drifts, correct_path
from carbon_world_model.plan import cdr_planning as P
from carbon_world_model.plan.scenario import build_scenario
from carbon_world_model.train.fit import load_trained

END, H_CAP, SEEDS = 2100, 28, (0, 1, 2, 3)
BUDGETS = {"2.0C_50": (2.0, 0.50), "2.0C_67": (2.0, 0.67), "1.7C_50": (1.7, 0.50)}
SPEND_GRID_TN = (0.0, 5.0, 10.0, 20.0, 40.0, float("inf"))
DYNAMICS = ("fitted", "open_loop")
OUT = "results/tables/cdr_planning.json"
CACHE = "results/cache_cdr_planning"      # per-model results, so a stopped run resumes (git-ignored)


def under_error(plan: P.CDRPlan, drifts, target: float) -> dict:
    """The plan's removals fixed; its gross path corrected by each backtest drift."""
    yrs = 2023 + np.arange(len(plan.removals))
    land = P.land_use_path(yrs)
    peaks, met = [], []
    for d in drifts:
        gm = correct_path(plan.global_mt, d["delta"], h_cap=H_CAP)
        net = gm[1:] / 1000.0 / P.COVERAGE + land - plan.removals
        peak = float(np.cumsum(net).max())
        peaks.append(peak)
        met.append(peak <= plan.budget_gt * (1 + P.MEET_TOL) and net[-1] <= P.NET_END_TOL)
    lik = B.likelihood_below(target, np.array(peaks))
    return {"n": len(drifts), "share_met": float(np.mean(met)),
            "likelihood_median": float(np.median(lik)),
            "likelihood_q10_q90": [float(np.quantile(lik, .1)), float(np.quantile(lik, .9))],
            "peak_q10_q90": [float(np.quantile(peaks, .1)), float(np.quantile(peaks, .9))]}


def main() -> None:
    panel = Panel.load(default_config())
    st = json.load(open("results/tables/structural.json"))
    plans, check = [], []
    t0 = time.time()
    for m in st["models"]:
        name = f"{m['config']}_s{m['seed']}"
        cache = os.path.join(CACHE, f"{name}.json")
        if os.path.exists(cache):
            got = json.load(open(cache))
            plans += got["plans"]
            check += got["check"]
            print(f"  {name}: {len(got['plans'])} plans from cache", flush=True)
            continue
        n0, c0 = len(plans), len(check)
        cfg, wm, params = load_trained(panel, f"results/model/structural/{name}.pkl")
        scen = build_scenario(panel, cfg, end_year=END, ssp="SSP2")
        drifts = backtest_drifts(panel, m["config"], seeds=SEEDS)
        for dyn in DYNAMICS:
            for bkey, (target, lik) in BUDGETS.items():
                bud = B.budget_gt(target, lik)
                for S in SPEND_GRID_TN:
                    r = P.plan_with_removals(wm, params, scen, bud, S, open_loop=(dyn == "open_loop"))
                    rec = {"model": name, "config": m["config"], "seed": m["seed"], "dynamics": dyn,
                           "budget": bkey, "target": target, "likelihood": lik,
                           **r.summary(), "spend_cap_tn": None if not np.isfinite(S) else S,
                           "central_likelihood": float(B.likelihood_below(target, r.peak_gt)),
                           "under_backtest_error": under_error(r, drifts, target),
                           "removals_gt": r.removals.round(4).tolist(),
                           "capacity_gt": r.capacity.round(4).tolist(),
                           "global_mt": r.global_mt.round(2).tolist()}
                    plans.append(rec)
                    print(f"  {name:30s} {dyn:9s} {bkey} S={S:>4g}: met={rec['met']!s:5s} "
                          f"effort {rec['effort']:.2f} spend {rec['spend_tn']:6.1f} tn "
                          f"peak {rec['peak_gt']:7.0f} net2100 {rec['net_2100_gt']:5.2f} "
                          f"CDR2050 {rec['removal_2050_gt']:4.1f} | under error met "
                          f"{100*rec['under_backtest_error']['share_met']:3.0f}% "
                          f"[{time.time()-t0:5.0f}s]", flush=True)
        if m["config"] == SELECTED and m["seed"] == 0:
            # optimiser check: a second initialisation on two cells
            for S in (10.0, float("inf")):
                bud = B.budget_gt(2.0, 0.5)
                a = P.plan_with_removals(wm, params, scen, bud, S, seed=0)
                b = P.plan_with_removals(wm, params, scen, bud, S, seed=1)
                check.append({"spend_cap_tn": None if not np.isfinite(S) else S,
                              "effort_seed0": a.effort, "effort_seed1": b.effort,
                              "spend_seed0": a.spend_tn, "spend_seed1": b.spend_tn,
                              "peak_seed0": a.peak_gt, "peak_seed1": b.peak_gt})
        os.makedirs(CACHE, exist_ok=True)
        with open(cache, "w") as fh:
            json.dump({"plans": plans[n0:], "check": check[c0:]}, fh)

    summary = {}
    for dyn in DYNAMICS:
        for bkey in BUDGETS:
            for S in SPEND_GRID_TN:
                cap = None if not np.isfinite(S) else S
                rows = [p for p in plans if p["dynamics"] == dyn and p["budget"] == bkey
                        and p["spend_cap_tn"] == cap]
                med = lambda k: float(np.median([p[k] for p in rows]))  # noqa: E731
                rng = lambda k: [float(min(p[k] for p in rows)), float(max(p[k] for p in rows))]  # noqa: E731
                summary[f"{dyn}|{bkey}|{'inf' if cap is None else f'{cap:g}'}"] = {
                    "n": len(rows), "share_met": float(np.mean([p["met"] for p in rows])),
                    "effort_median": med("effort"), "effort_range": rng("effort"),
                    "spend_median": med("spend_tn"), "spend_range": rng("spend_tn"),
                    "removal_2050_median": med("removal_2050_gt"),
                    "removal_2100_median": med("removal_2100_gt"),
                    "removal_total_median": med("removal_total_gt"),
                    "fossil_cum_median": med("fossil_cum_gt"),
                    "peak_median": med("peak_gt"),
                    "share_met_under_error_median": float(np.median(
                        [p["under_backtest_error"]["share_met"] for p in rows])),
                    "likelihood_under_error_median": float(np.median(
                        [p["under_backtest_error"]["likelihood_median"] for p in rows])),
                }
    out = {"end_year": END, "budgets": {k: {"target": t, "likelihood": lk, "gt": B.budget_gt(t, lk)}
                                        for k, (t, lk) in BUDGETS.items()},
           "spend_grid_tn": [None if not np.isfinite(s) else s for s in SPEND_GRID_TN],
           "selected": SELECTED, "plans": plans, "summary": summary, "optimiser_check": check}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)

    print("\n[summary] median over the 4 tied models: met share | effort (1 = max) | spend tn | "
          "CDR 2050 Gt | fossil 2023-2100 Gt | met under backtest error")
    for k, v in summary.items():
        print(f"  {k:28s} met {100*v['share_met']:3.0f}% | effort {v['effort_median']:.2f} "
              f"[{v['effort_range'][0]:.2f}-{v['effort_range'][1]:.2f}] | spend {v['spend_median']:5.1f} | "
              f"CDR2050 {v['removal_2050_median']:4.1f} | fossil {v['fossil_cum_median']:6.0f} | "
              f"under error {100*v['share_met_under_error_median']:3.0f}% "
              f"(likelihood {100*v['likelihood_under_error_median']:.0f}%)")
    print("optimiser check:", check)
    print(f"[ok] wrote {OUT}")


if __name__ == "__main__":
    main()
