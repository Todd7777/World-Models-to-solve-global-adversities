"""Decarbonisation planning inside the world model: business-as-usual, maximum
feasible ambition, the cost-emissions frontier, and the climate (TCRE) translation.

Usage:  python scripts/04_plan.py
"""
import json
import os

import numpy as np

from carbon_world_model.config import default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.model.world_model import WorldModel
from carbon_world_model.train.fit import fit, load_params
from carbon_world_model.plan.scenario import build_scenario
from carbon_world_model.plan import optimize as O
from carbon_world_model.climate import tcre

OUT = "results/tables"


def main() -> None:
    cfg = default_config()
    panel = Panel.load(cfg)
    wm = WorldModel.build(panel, cfg)
    try:
        params = load_params("results/model/params_full.pkl")
    except FileNotFoundError:
        params = fit(panel, cfg, wm=wm, verbose=False).params

    scen = build_scenario(panel, cfg, end_year=2050)
    base_2022 = float(panel.co2[:, -1].sum())

    bau = O.business_as_usual(wm, params, scen)
    mx = O.max_ambition(wm, params, scen)

    print(f"feasibility (sustained frontier): clean-share +{scen.g_max*100:.1f}%/yr, "
          f"efficiency {scen.h_max*100:.1f}%/yr")
    print(f"2022 emissions (60 countries): {base_2022:,.0f} Mt")
    print(f"BAU 2050:          {bau.global_co2[-1]:,.0f} Mt  (cum 2023-50 = {bau.cum_gt:,.0f} Gt)")
    print(f"MAX feasible 2050: {mx.global_co2[-1]:,.0f} Mt  (cum 2023-50 = {mx.cum_gt:,.0f} Gt)  "
          f"= -{100*(1-mx.global_co2[-1]/bau.global_co2[-1]):.0f}% vs BAU")

    # Cost-emissions Pareto frontier.
    lams = [0.0, 0.01, 0.03, 0.08, 0.2, 0.5, 1.5, 5.0]
    frontier = O.pareto_frontier(wm, params, scen, lams, steps=700)

    # Climate translation.
    clim = cfg.climate
    avoided = tcre.avoided_warming(bau.cum_gt, mx.cum_gt, clim)
    bud15 = tcre.remaining_budget_gtco2(1.5, clim)
    bud20 = tcre.remaining_budget_gtco2(2.0, clim)
    glob_bau = tcre.scale_to_global(bau.cum_gt)
    glob_max = tcre.scale_to_global(mx.cum_gt)
    print(f"\nAvoided warming (MAX vs BAU, this group): "
          f"{avoided['central']:.3f} degC [{avoided['low']:.3f}-{avoided['high']:.3f}]")
    print(f"Scaled-to-global cumulative 2023-50: BAU {glob_bau:,.0f} Gt, MAX {glob_max:,.0f} Gt")
    print(f"Remaining global budget from 2022: 1.5C ~ {bud15:,.0f} Gt, 2.0C ~ {bud20:,.0f} Gt")

    # Per-country effort and emission cuts under max ambition (who must do what).
    cut = 1 - mx.co2[:, -1] / np.clip(bau.co2[:, -1], 1e-6, None)
    order = np.argsort(-panel.co2[:, -1])[:12]
    percountry = [
        {"country": panel.countries[i], "emis_2022_mt": float(panel.co2[i, -1]),
         "bau_2050_mt": float(bau.co2[i, -1]), "max_2050_mt": float(mx.co2[i, -1]),
         "cut_vs_bau_pct": float(100 * cut[i])}
        for i in order
    ]

    result = {
        "scenario": {"years": [int(y) for y in scen.years], "g_max": scen.g_max,
                      "h_max": scen.h_max, "end_year": 2050},
        "base_2022_mt": base_2022,
        "bau": {"global_mt": bau.global_co2.tolist(), "cum_gt": bau.cum_gt},
        "max_ambition": {"global_mt": mx.global_co2.tolist(), "cum_gt": mx.cum_gt,
                          "cost": mx.cost},
        "frontier": frontier,
        "climate": {
            "avoided_warming_degC": avoided,
            "global_cum_bau_gt": glob_bau, "global_cum_max_gt": glob_max,
            "budget_1p5_gt": bud15, "budget_2p0_gt": bud20,
            "tcre_degC_per_ttco2": clim.tcre_degC_per_ttco2,
        },
        "per_country_top12": percountry,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/plan.json", "w") as fh:
        json.dump(result, fh, indent=2)
    # Save trajectories for figures.
    np.savez(f"{OUT}/plan_trajectories.npz",
             years=scen.years, bau_co2=bau.co2, max_co2=mx.co2,
             bau_shares=bau.shares, max_shares=mx.shares,
             countries=np.array(panel.countries))
    print(f"\n[ok] wrote {OUT}/plan.json and plan_trajectories.npz")


if __name__ == "__main__":
    main()
