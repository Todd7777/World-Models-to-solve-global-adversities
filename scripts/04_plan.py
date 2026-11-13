"""Decarbonisation planning inside the world model under SSP1-5 socioeconomic futures,
with a seed ensemble of the selected configuration (plan/ensemble.py).

1. Business-as-usual and maximum-feasible-effort pathways for each SSP (v3.2 drivers),
   plus the previous heuristic scenario for comparison, under every ensemble member.
2. Climate translation (TCRE) and comparison with the remaining 1.5/2.0 C budgets.
3. Cost-of-abatement frontier under SSP2: for each budget, the cheapest single plan whose
   ensemble-mean cumulative emissions meet it.
4. Robust planning: one plan that meets a cumulative budget in EVERY SSP under EVERY
   ensemble member, versus a plan tuned to SSP2 and the ensemble mean -- the price of
   robustness and the regret of not hedging.

Plans are applied at the feasibility frontier of each member's business as usual
(plan/optimize.py): the network projects business as usual only, and a plan adds extra
annual rates of intensity fall and low-carbon gain on top, capped at the limits.

Central values are ensemble means: trajectories are the mean of the members' global
totals, and the 2050 cut is that of the mean pathways. `by_seed` lists every member's
value. The ensemble is results/model/ensemble (scripts/02b_train_ensemble.py); its
seed-0 member is the committed checkpoint.

Refuses to run, before writing anything, unless results/tables/config.json, which
scripts/02_train.py writes with the checkpoint, matches default_config(), and the
ensemble was trained with that configuration and contains the checkpoint.

Usage:  python scripts/04_plan.py
"""
import json
import os
import sys

import numpy as np

from carbon_world_model.config import SELECTED_CONFIG
from carbon_world_model.data.panel import Panel
from carbon_world_model.data.ssp import SSPS
from carbon_world_model.model.world_model import WorldModel
from carbon_world_model.plan.ensemble import (
    ENSEMBLE_DIR, ensemble_pathway, evaluate_members, load_ensemble, max_controls,
    rollout_members, stack_members,
)
from carbon_world_model.plan.scenario import build_scenario, ssp_ensemble
from carbon_world_model.climate import tcre
from carbon_world_model.eval.configs import ProvenanceError, check_default_checkpoint

OUT = "results/tables"
CHECKPOINT = "results/model/params_full.pkl"
CHECKPOINT_CONFIG = os.path.join(OUT, "config.json")  # written by 02_train.py with CHECKPOINT


def global_totals(run) -> np.ndarray:
    """[K, Tf] 60-country totals (Mt), summed in float32 as optimize._summaries does."""
    return run["co2"].sum(axis=1).astype(float)


def mix(run) -> np.ndarray:
    """[Tf, F] energy-weighted aggregate fuel shares, averaged over the members."""
    E = run["shares"] * run["D"][..., None]                         # [K, C, Tf, F]
    agg = E.sum(axis=1) / np.clip(E.sum(axis=(1, 3))[..., None], 1e-9, None)
    return agg.mean(axis=0)


def spread(x) -> str:
    return f"{min(x):,.0f}-{max(x):,.0f}"


def main() -> None:
    cfg = check_default_checkpoint(CHECKPOINT_CONFIG)  # before anything is written
    seeds, members = load_ensemble(ENSEMBLE_DIR, cfg, CHECKPOINT)
    stacked = stack_members(members)
    panel = Panel.load(cfg)
    wm = WorldModel.build(panel, cfg)
    clim = cfg.climate
    base_2022 = float(panel.co2[:, -1].sum())

    scens = ssp_ensemble(panel, cfg, end_year=2050)
    heur = build_scenario(panel, cfg, end_year=2050, ssp=None)
    print(f"ensemble: {SELECTED_CONFIG}, training seeds {seeds}. Central values are ensemble "
          "means; ranges are the lowest and highest seed.")
    print(f"feasibility (sustained frontier): clean-share +{scens[0].g_max*100:.1f}%/yr, "
          f"efficiency {scens[0].h_max*100:.1f}%/yr | 2022 emissions {base_2022:,.0f} Mt")

    per_ssp, runs = {}, {}
    print(f"\n{'scenario':16s} {'BAU 2050 (range)':>24s} {'MAX 2050 (range)':>24s} "
          f"{'cut (range)':>14s} {'avoided C':>9s}")
    for s in scens + [heur]:
        g, h = max_controls(s)
        bau, mx = rollout_members(wm, stacked, s), rollout_members(wm, stacked, s, g, h)
        runs[s.name] = (bau, mx)
        bg, mg = global_totals(bau), global_totals(mx)
        bcum, mcum = bg[:, 1:].sum(axis=1) / 1000.0, mg[:, 1:].sum(axis=1) / 1000.0
        cut = 100 * (1 - mg[:, -1] / bg[:, -1])
        av = tcre.avoided_warming(float(bcum.mean()), float(mcum.mean()), clim)
        per_ssp[s.name] = {
            "bau_global_mt": bg.mean(axis=0).tolist(), "max_global_mt": mg.mean(axis=0).tolist(),
            "bau_cum_gt": float(bcum.mean()), "max_cum_gt": float(mcum.mean()),
            "bau_2050_mt": float(bg[:, -1].mean()), "max_2050_mt": float(mg[:, -1].mean()),
            "cut_2050_pct": float(100 * (1 - mg[:, -1].mean() / bg[:, -1].mean())),
            "avoided_warming_degC": av,
            "global_cum_bau_gt": tcre.scale_to_global(float(bcum.mean())),
            "global_cum_max_gt": tcre.scale_to_global(float(mcum.mean())),
            "by_seed": {
                "bau_2050_mt": bg[:, -1].tolist(), "max_2050_mt": mg[:, -1].tolist(),
                "bau_cum_gt": bcum.tolist(), "max_cum_gt": mcum.tolist(),
                "cut_2050_pct": cut.tolist(),
                "avoided_warming_central_degC": [
                    tcre.avoided_warming(float(b), float(m), clim)["central"]
                    for b, m in zip(bcum, mcum)],
            },
        }
        d = per_ssp[s.name]
        print(f"{s.name:16s} {d['bau_2050_mt']:9,.0f} ({spread(bg[:, -1]):>13s}) "
              f"{d['max_2050_mt']:9,.0f} ({spread(mg[:, -1]):>13s}) "
              f"{d['cut_2050_pct']:4.0f}% ({cut.min():2.0f}-{cut.max():2.0f}%) "
              f"{av['central']:9.3f}")

    bud15 = tcre.remaining_budget_gtco2(1.5, clim)
    bud20 = tcre.remaining_budget_gtco2(2.0, clim)
    print(f"\nremaining global budget from 2022: 1.5C ~{bud15:,.0f} Gt, 2.0C ~{bud20:,.0f} Gt")

    # Cost-of-abatement frontier under SSP2: one plan per budget, met by the ensemble mean.
    ssp2 = scens[SSPS.index("SSP2")]
    lo, hi = per_ssp[ssp2.name]["max_cum_gt"], per_ssp[ssp2.name]["bau_cum_gt"]
    frontier = []
    for bud in np.linspace(lo + 5, hi - 5, 8):
        r = ensemble_pathway(wm, stacked, [ssp2], budget_gt=float(bud), mode="mean")
        cum = r["cum_gt"][:, 0]
        frontier.append({"budget_gt": float(bud), "cum_gt": float(cum.mean()), "cost": r["cost"],
                         "final_global_mt": float(r["final_mt"][:, 0].mean()),
                         "cum_gt_by_seed": cum.tolist()})

    # Robust planning. Budget: the tightest cumulative level that maximum effort reaches in
    # every SSP under every member, plus a margin: demanding, but feasible in all cases.
    max_cum = np.array([per_ssp[s.name]["by_seed"]["max_cum_gt"] for s in scens])  # [S, K]
    bau_cum = np.array([per_ssp[s.name]["by_seed"]["bau_cum_gt"] for s in scens])
    worst_max, best_bau = float(max_cum.max()), float(bau_cum.min())
    # A quarter of the way from the worst-case feasibility floor towards the best-case
    # BAU; if they do not overlap that way, sit just above the worst-case floor.
    budget = (worst_max + 0.25 * (best_bau - worst_max) if best_bau > worst_max
              else 1.02 * worst_max)
    tailored = ensemble_pathway(wm, stacked, [ssp2], budget_gt=budget, mode="mean")
    tail_cum = evaluate_members(wm, stacked, scens, tailored["g"], tailored["h"])["cum_gt"]
    robust = ensemble_pathway(wm, stacked, scens, budget_gt=budget, mode="all")
    rob_cum = robust["cum_gt"]                                                      # [K, S]
    over = {s: max(0.0, float(tail_cum[:, i].max()) - budget) for i, s in enumerate(SSPS)}
    price = 100 * (robust["cost"] / tailored["cost"] - 1)
    print(f"\nrobust planning, budget {budget:,.0f} Gt (2023-2050, 60 countries), "
          f"{len(seeds)} seeds:")
    print("  SSP2-tailored plan (ensemble mean): cost "
          f"{tailored['cost']:.3f}; worst seed by SSP = "
          + ", ".join(f"{s}:{c:,.0f}" for s, c in zip(SSPS, tail_cum.max(axis=0))))
    print(f"  robust plan (every SSP and seed):   cost {robust['cost']:.3f}; worst seed by SSP = "
          + ", ".join(f"{s}:{c:,.0f}" for s, c in zip(SSPS, rob_cum.max(axis=0))))
    print(f"  price of robustness: +{price:.0f}% effort; regret of the SSP2-tailored plan: "
          f"worst overshoot {max(over.values()):,.0f} Gt ({max(over, key=over.get)})")

    # Who must do what under maximum effort (SSP2), ensemble means.
    bau2, mx2 = runs[ssp2.name]
    bc, mc = bau2["co2"][:, :, -1].astype(float), mx2["co2"][:, :, -1].astype(float)  # [K, C]
    order = np.argsort(-panel.co2[:, -1])[:12]
    percountry = []
    for i in order:
        b, m = bc[:, i], mc[:, i]
        percountry.append({
            "country": panel.countries[i], "emis_2022_mt": float(panel.co2[i, -1]),
            "bau_2050_mt": float(b.mean()), "max_2050_mt": float(m.mean()),
            "cut_vs_bau_pct": float(100 * (1 - m.mean() / max(b.mean(), 1e-6))),
            "cut_vs_bau_pct_by_seed": (100 * (1 - m / np.clip(b, 1e-6, None))).tolist()})

    result = {
        "years": [int(y) for y in ssp2.years], "base_2022_mt": base_2022,
        "ensemble": {"config": SELECTED_CONFIG, "seeds": seeds,
                     "central": "mean over seeds (trajectories: mean of the seeds' totals)"},
        "feasibility": {"g_max": ssp2.g_max, "h_max": ssp2.h_max},
        "scenarios": per_ssp, "frontier_ssp2": frontier,
        "budgets": {"global_1p5_gt": bud15, "global_2p0_gt": bud20,
                    "tcre_degC_per_ttco2": clim.tcre_degC_per_ttco2},
        "robust": {"budget_gt": budget, "tailored_cost": tailored["cost"],
                   "tailored_cum_by_ssp": dict(zip(SSPS, tail_cum.mean(axis=0).tolist())),
                   "tailored_cum_by_ssp_worst_seed": dict(zip(SSPS, tail_cum.max(axis=0).tolist())),
                   "robust_cost": robust["cost"],
                   "robust_cum_by_ssp": dict(zip(SSPS, rob_cum.mean(axis=0).tolist())),
                   "robust_cum_by_ssp_worst_seed": dict(zip(SSPS, rob_cum.max(axis=0).tolist())),
                   "price_of_robustness_pct": price, "tailored_overshoot_gt": over},
        "per_country_top12_ssp2": percountry,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/plan.json", "w") as fh:
        json.dump(result, fh, indent=2)
    names = list(per_ssp)
    np.savez(f"{OUT}/plan_trajectories.npz", years=ssp2.years, seeds=np.array(seeds),
             countries=np.array(panel.countries),
             bau_co2=bau2["co2"].mean(axis=0), max_co2=mx2["co2"].mean(axis=0),
             bau_shares=bau2["shares"].mean(axis=0), max_shares=mx2["shares"].mean(axis=0),
             bau_D=bau2["D"].mean(axis=0), max_D=mx2["D"].mean(axis=0),
             bau_mix=mix(bau2), max_mix=mix(mx2), scenario_names=np.array(names),
             bau_global_by_seed=np.stack([global_totals(runs[n][0]) for n in names]),
             max_global_by_seed=np.stack([global_totals(runs[n][1]) for n in names]),
             robust_g=robust["g"], robust_h=robust["h"])
    print(f"\n[ok] wrote {OUT}/plan.json and plan_trajectories.npz")


if __name__ == "__main__":
    try:
        main()
    except ProvenanceError as e:
        sys.exit(f"04_plan.py: {e}")
