"""Generate all figures into results/figures/.

Usage:  python scripts/05_make_figures.py
Requires: scripts 01-04 to have produced the panel, params, and result tables.
"""
import json
import os

import numpy as np

from carbon_world_model.config import default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.model.world_model import WorldModel
from carbon_world_model.model.emissions import estimate_emission_factors, reconstruction_report
from carbon_world_model.train.fit import load_params
from carbon_world_model.eval.transfer import spatial_transfer
from carbon_world_model.plan.scenario import build_scenario
from carbon_world_model.plan import optimize as O
from carbon_world_model.viz import figures as F

FIG = "results/figures"
TAB = "results/tables"


def main() -> None:
    cfg = default_config()
    panel = Panel.load(cfg)
    wm = WorldModel.build(panel, cfg)
    params = load_params("results/model/params_full.pkl")
    os.makedirs(FIG, exist_ok=True)

    ef = estimate_emission_factors(panel, "nnls")
    rec = reconstruction_report(panel, ef)
    F.fig_reconstruction(panel, ef, rec).savefig(f"{FIG}/fig1_reconstruction.png", bbox_inches="tight")
    F.fig_insample(panel, wm, params).savefig(f"{FIG}/fig2_insample.png", bbox_inches="tight")

    with open(f"{TAB}/backtest_2015.json") as fh:
        bt = json.load(fh)
    F.fig_backtest(panel, bt).savefig(f"{FIG}/fig3_backtest.png", bbox_inches="tight")

    # Transfer (recompute; fast enough).
    tr_res = spatial_transfer(panel, cfg, test_every=5, cutoff=2015, seed=0, verbose=False)
    F.fig_transfer(panel, tr_res).savefig(f"{FIG}/fig4_transfer.png", bbox_inches="tight")

    with open(f"{TAB}/plan.json") as fh:
        plan = json.load(fh)
    F.fig_planning(panel, plan).savefig(f"{FIG}/fig5_planning.png", bbox_inches="tight")

    # Energy-weighted aggregate mix for BAU vs max-ambition.
    scen = build_scenario(panel, cfg, end_year=2050)
    bau = O.business_as_usual(wm, params, scen)
    mx = O.max_ambition(wm, params, scen)

    def agg_shares(res):
        E = res.shares * res.D[:, :, None]        # [C,Tf,F] energy by fuel
        tot = E.sum(axis=(0, 2), keepdims=True)[..., 0]  # [1,Tf]
        return E.sum(axis=0) / np.clip(tot.T, 1e-9, None)  # [Tf,F]

    F.fig_mix(panel, scen.years, agg_shares(bau), agg_shares(mx)).savefig(
        f"{FIG}/fig6_mix.png", bbox_inches="tight")

    print("[ok] figures written to", FIG)
    for f in sorted(os.listdir(FIG)):
        print("  ", f)


if __name__ == "__main__":
    main()
