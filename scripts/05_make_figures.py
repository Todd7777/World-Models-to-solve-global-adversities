"""Generate all figures into results/figures/ from the result tables.

Usage:  python scripts/05_make_figures.py
Requires scripts 01-04 (and the sweeps) to have produced the tables and checkpoint.
"""
import json
import os

import numpy as np

from carbon_world_model.config import default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.data.ssp import SSPS, load_processed, ssp_driver_paths
from carbon_world_model.model.world_model import WorldModel
from carbon_world_model.model.emissions import estimate_emission_factors, reconstruction_report
from carbon_world_model.train.fit import load_params
from carbon_world_model.eval.configs import make_config
from carbon_world_model.eval.rolling import job_path
from carbon_world_model.eval.baselines import trend_kaya
from carbon_world_model.plan.scenario import build_scenario
from carbon_world_model.plan import optimize as O
from carbon_world_model.viz import figures as F

FIG, TAB = "results/figures", "results/tables"


def load(name):
    with open(os.path.join(TAB, name)) as fh:
        return json.load(fh)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")


def main() -> None:
    cfg = default_config()
    panel = Panel.load(cfg)
    os.makedirs(FIG, exist_ok=True)
    selected = load("selected_config.json")["config"]
    wm = WorldModel.build(panel, cfg)
    params = load_params("results/model/params_full.pkl")

    ef = estimate_emission_factors(panel, "nnls")
    save(F.fig_reconstruction(panel, ef, reconstruction_report(panel, ef)), "fig1_reconstruction.png")
    save(F.fig_insample(panel, wm, params), "fig2_insample.png")
    save(F.fig_ablation(load("ablation_val.json"), selected), "fig3_ablation_validation.png")
    save(F.fig_test(load("test.json"), load("china_decomposition.json")), "fig4_test_and_china.png")

    # China forecasts from 2015 (seed-mean), selected vs original config vs trend-Kaya.
    k = int(np.where(panel.years == 2015)[0][0])
    yrs = panel.years[k:]
    series = {}
    for name, label in ((selected, f"world model ({selected})"), ("orig", "world model (original)")):
        runs = [np.load(job_path("results/sweep_test", name, 2015, s)) for s in range(5)
                if os.path.exists(job_path("results/sweep_test", name, 2015, s))]
        if runs:
            series[label] = (yrs, np.mean([r["co2_realized"] for r in runs], 0)[panel.country_index("China")])
    series["trend-Kaya"] = (yrs, trend_kaya(panel, 2015)[panel.country_index("China"), k:])
    save(F.fig_china_forecasts(panel, 2015, series), "fig5_china_forecasts.png")

    save(F.fig_ssp_drivers(panel, load_processed("v9"), load_processed("v3.2"), ssp_driver_paths, SSPS),
         "fig6_ssp_drivers.png")
    plan = load("plan.json")
    save(F.fig_ssp_pathways(panel, plan), "fig7_ssp_pathways_robust.png")

    # Fuel mix BAU vs max-feasible under SSP2 (energy-weighted aggregate shares).
    scen = build_scenario(panel, cfg, end_year=2050, ssp="SSP2")
    bau, mx = O.business_as_usual(wm, params, scen), O.max_ambition(wm, params, scen)

    def agg(res):
        E = res.shares * res.D[:, :, None]
        return E.sum(axis=0) / np.clip(E.sum(axis=(0, 2))[:, None], 1e-9, None)

    save(F.fig_mix(panel, scen.years, agg(bau), agg(mx)), "fig8_mix_ssp2.png")

    # Cost-of-abatement frontier (SSP2), reusing the planning panel layout.
    legacy = {"scenario": {"years": plan["years"]},
              "bau": {"global_mt": plan["scenarios"][scen.name]["bau_global_mt"]},
              "max_ambition": {"global_mt": plan["scenarios"][scen.name]["max_global_mt"]},
              "frontier": plan["frontier_ssp2"]}
    save(F.fig_planning(panel, legacy), "fig9_frontier_ssp2.png")
    print("[ok] figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
