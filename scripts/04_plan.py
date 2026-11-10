"""Decarbonisation planning inside the world model under SSP1-5 socioeconomic futures.

1. Business-as-usual and maximum-feasible-effort pathways for each SSP (v3.2 drivers),
   plus the previous heuristic scenario for comparison.
2. Climate translation (TCRE) and comparison with the remaining 1.5/2.0 C budgets.
3. Cost-of-abatement frontier under SSP2.
4. Robust planning: one plan that meets a cumulative budget in EVERY SSP, versus a plan
   tuned to SSP2 alone -- the price of robustness and the regret of not hedging.

Usage:  python scripts/04_plan.py
"""
import json
import os

import numpy as np

from carbon_world_model.config import default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.data.ssp import SSPS
from carbon_world_model.train.fit import load_trained
from carbon_world_model.plan.scenario import build_scenario, ssp_ensemble
from carbon_world_model.plan import optimize as O
from carbon_world_model.climate import tcre

OUT = "results/tables"


def main() -> None:
    panel = Panel.load(default_config())
    cfg, wm, params = load_trained(panel)  # model built with the checkpoint's own config
    clim = cfg.climate
    base_2022 = float(panel.co2[:, -1].sum())

    scens = ssp_ensemble(panel, cfg, end_year=2050)
    heur = build_scenario(panel, cfg, end_year=2050, ssp=None)
    print(f"feasibility (sustained frontier): clean-share +{scens[0].g_max*100:.1f}%/yr, "
          f"efficiency {scens[0].h_max*100:.1f}%/yr | 2022 emissions {base_2022:,.0f} Mt")

    per_ssp = {}
    print(f"\n{'scenario':16s} {'BAU 2050':>10s} {'BAU cum':>9s} {'MAX 2050':>10s} {'MAX cum':>9s} "
          f"{'cut':>5s} {'avoided C':>10s}")
    for s in scens + [heur]:
        bau, mx = O.business_as_usual(wm, params, s), O.max_ambition(wm, params, s)
        av = tcre.avoided_warming(bau.cum_gt, mx.cum_gt, clim)
        per_ssp[s.name] = {
            "bau_global_mt": bau.global_co2.tolist(), "max_global_mt": mx.global_co2.tolist(),
            "bau_cum_gt": bau.cum_gt, "max_cum_gt": mx.cum_gt,
            "bau_2050_mt": float(bau.global_co2[-1]), "max_2050_mt": float(mx.global_co2[-1]),
            "avoided_warming_degC": av,
            "global_cum_bau_gt": tcre.scale_to_global(bau.cum_gt),
            "global_cum_max_gt": tcre.scale_to_global(mx.cum_gt),
        }
        print(f"{s.name:16s} {bau.global_co2[-1]:10,.0f} {bau.cum_gt:9,.0f} {mx.global_co2[-1]:10,.0f} "
              f"{mx.cum_gt:9,.0f} {100*(1-mx.global_co2[-1]/bau.global_co2[-1]):4.0f}% "
              f"{av['central']:9.3f}")

    bud15 = tcre.remaining_budget_gtco2(1.5, clim)
    bud20 = tcre.remaining_budget_gtco2(2.0, clim)
    print(f"\nremaining global budget from 2022: 1.5C ~{bud15:,.0f} Gt, 2.0C ~{bud20:,.0f} Gt")

    # Cost-of-abatement frontier under SSP2.
    ssp2 = scens[SSPS.index("SSP2")]
    lo, hi = per_ssp[ssp2.name]["max_cum_gt"], per_ssp[ssp2.name]["bau_cum_gt"]
    frontier = O.pareto_frontier(wm, params, ssp2, list(np.linspace(lo + 5, hi - 5, 8)), steps=1000)

    # Robust planning. Budget: the tightest cumulative level feasible in every SSP plus a
    # margin, i.e. a demanding target that all futures can still meet.
    worst_max = max(per_ssp[s.name]["max_cum_gt"] for s in scens)
    best_bau = min(per_ssp[s.name]["bau_cum_gt"] for s in scens)
    # A quarter of the way from the worst-case feasibility floor towards the best-case
    # BAU; if the SSPs do not overlap that way, sit just above the worst-case floor.
    budget = (worst_max + 0.25 * (best_bau - worst_max) if best_bau > worst_max
              else 1.02 * worst_max)
    tailored = O.optimize_pathway(wm, params, ssp2, budget_gt=budget, penalty=2.0, steps=1500)
    ev_tail = O.evaluate_controls(wm, params, scens, tailored.g, tailored.h)
    robust = O.robust_pathway(wm, params, scens, budget_gt=budget, penalty=2.0, mode="all",
                              steps=1500)
    print(f"\nrobust planning, budget {budget:,.0f} Gt (2023-2050, 60 countries):")
    print(f"  SSP2-tailored plan: cost {tailored.cost:.3f}; cum by SSP = "
          + ", ".join(f"{s}:{c:,.0f}" for s, c in zip(SSPS, ev_tail['cum_gt'])))
    print(f"  robust plan:        cost {robust['cost']:.3f}; cum by SSP = "
          + ", ".join(f"{s}:{c:,.0f}" for s, c in zip(SSPS, robust['cum_gt'])))
    over = {s: max(0.0, c - budget) for s, c in zip(SSPS, ev_tail["cum_gt"])}
    print(f"  price of robustness: +{100*(robust['cost']/tailored.cost-1):.0f}% effort; "
          f"regret of SSP2-only plan: worst overshoot {max(over.values()):,.0f} Gt "
          f"({max(over, key=over.get)})")

    # Who must do what under maximum effort (SSP2).
    bau2, mx2 = O.business_as_usual(wm, params, ssp2), O.max_ambition(wm, params, ssp2)
    order = np.argsort(-panel.co2[:, -1])[:12]
    percountry = [{"country": panel.countries[i], "emis_2022_mt": float(panel.co2[i, -1]),
                   "bau_2050_mt": float(bau2.co2[i, -1]), "max_2050_mt": float(mx2.co2[i, -1]),
                   "cut_vs_bau_pct": float(100 * (1 - mx2.co2[i, -1] / max(bau2.co2[i, -1], 1e-6)))}
                  for i in order]

    result = {
        "years": [int(y) for y in ssp2.years], "base_2022_mt": base_2022,
        "feasibility": {"g_max": ssp2.g_max, "h_max": ssp2.h_max},
        "scenarios": per_ssp, "frontier_ssp2": frontier,
        "budgets": {"global_1p5_gt": bud15, "global_2p0_gt": bud20,
                    "tcre_degC_per_ttco2": clim.tcre_degC_per_ttco2},
        "robust": {"budget_gt": budget, "tailored_cost": tailored.cost,
                   "tailored_cum_by_ssp": dict(zip(SSPS, ev_tail["cum_gt"])),
                   "robust_cost": robust["cost"], "robust_cum_by_ssp": dict(zip(SSPS, robust["cum_gt"])),
                   "price_of_robustness_pct": 100 * (robust["cost"] / tailored.cost - 1),
                   "tailored_overshoot_gt": over},
        "per_country_top12_ssp2": percountry,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/plan.json", "w") as fh:
        json.dump(result, fh, indent=2)
    np.savez(f"{OUT}/plan_trajectories.npz", years=ssp2.years,
             bau_co2=bau2.co2, max_co2=mx2.co2, bau_shares=bau2.shares, max_shares=mx2.shares,
             bau_D=bau2.D, max_D=mx2.D, countries=np.array(panel.countries),
             robust_g=robust["g"], robust_h=robust["h"])
    print(f"\n[ok] wrote {OUT}/plan.json and plan_trajectories.npz")


if __name__ == "__main__":
    main()
