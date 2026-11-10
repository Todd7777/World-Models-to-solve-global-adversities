"""Generate all figures into results/figures/ from the result tables.

Usage:  python scripts/05_make_figures.py
Requires scripts 01-04 (and the sweeps) to have produced the tables and checkpoint.
Refuses to run, before writing anything, unless results/tables/config.json, which
scripts/02_train.py writes with the checkpoint, matches default_config(): the
configuration the checkpoint is run under here.
"""
import json
import os
import sys

import numpy as np

from carbon_world_model.data.panel import Panel
from carbon_world_model.data.ssp import SSPS, load_processed, ssp_driver_paths
from carbon_world_model.model.world_model import WorldModel
from carbon_world_model.model.emissions import estimate_emission_factors, reconstruction_report
from carbon_world_model.train.fit import load_params
from carbon_world_model.eval.configs import ProvenanceError, check_default_checkpoint
from carbon_world_model.eval.rolling import load_jobs
from carbon_world_model.eval.baselines import trend_kaya
from carbon_world_model.plan.scenario import build_scenario
from carbon_world_model.viz import figures as F

FIG, TAB = "results/figures", "results/tables"
CHECKPOINT = "results/model/params_full.pkl"
CHECKPOINT_CONFIG = os.path.join(TAB, "config.json")  # written by 02_train.py with CHECKPOINT


def load(name):
    with open(os.path.join(TAB, name)) as fh:
        return json.load(fh)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")


def main() -> None:
    cfg = check_default_checkpoint(CHECKPOINT_CONFIG)  # before anything is written
    panel = Panel.load(cfg)
    os.makedirs(FIG, exist_ok=True)
    selected = load("selected_config.json")["config"]
    wm = WorldModel.build(panel, cfg)
    params = load_params(CHECKPOINT)

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
        runs = load_jobs("results/sweep_test", name, [2015], range(5))[0].get(2015, [])
        if runs:
            series[label] = (yrs, np.mean([r["co2_realized"] for r in runs], 0)[panel.country_index("China")])
    series["trend-Kaya"] = (yrs, trend_kaya(panel, 2015)[panel.country_index("China"), k:])
    save(F.fig_china_forecasts(panel, 2015, series), "fig5_china_forecasts.png")

    save(F.fig_ssp_drivers(panel, load_processed("v9"), load_processed("v3.2"), ssp_driver_paths, SSPS),
         "fig6_ssp_drivers.png")
    plan = load("plan.json")
    traj = dict(np.load(os.path.join(TAB, "plan_trajectories.npz")))
    save(F.fig_ssp_pathways(panel, plan, traj), "fig7_ssp_pathways_robust.png")

    # Fuel mix BAU vs max-feasible under SSP2 (energy-weighted aggregate shares, averaged
    # over the seed ensemble by 04_plan.py).
    scen = build_scenario(panel, cfg, end_year=2050, ssp="SSP2")
    save(F.fig_mix(panel, traj["years"], traj["bau_mix"], traj["max_mix"]), "fig8_mix_ssp2.png")

    # Cost-of-abatement frontier (SSP2), reusing the planning panel layout.
    legacy = {"scenario": {"years": plan["years"]},
              "bau": {"global_mt": plan["scenarios"][scen.name]["bau_global_mt"]},
              "max_ambition": {"global_mt": plan["scenarios"][scen.name]["max_global_mt"]},
              "frontier": plan["frontier_ssp2"]}
    save(F.fig_planning(panel, legacy), "fig9_frontier_ssp2.png")
    print("[ok] figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    try:
        main()
    except ProvenanceError as e:
        sys.exit(f"05_make_figures.py: {e}")
