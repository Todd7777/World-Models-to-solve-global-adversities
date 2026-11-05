"""Evaluate the Carbon World Model: in-sample fidelity, temporal backtest vs
baselines, and rolling-origin calibration. Writes result tables to results/tables/.

Usage:  python scripts/03_backtest.py
"""
import json
import os

import numpy as np

from carbon_world_model.config import default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.model.world_model import WorldModel
from carbon_world_model.model.emissions import estimate_emission_factors, reconstruction_report
from carbon_world_model.train.fit import fit, load_params
from carbon_world_model.eval.backtest import (
    run_temporal_backtest, insample_report, rolling_conformal,
)

OUT = "results/tables"


def main() -> None:
    cfg = default_config()
    panel = Panel.load(cfg)
    os.makedirs(OUT, exist_ok=True)

    # Mechanistic layer validation.
    ef = estimate_emission_factors(panel, "nnls")
    rec = reconstruction_report(panel, ef)
    with open(f"{OUT}/emissions_reconstruction.json", "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"[A] emissions reconstruction R2={rec['r2']:.4f} MAPE={100*rec['mape']:.1f}%")

    # In-sample reconstruction with the trained full model.
    wm = WorldModel.build(panel, cfg)
    try:
        params = load_params("results/model/params_full.pkl")
    except FileNotFoundError:
        params = fit(panel, cfg, wm=wm, verbose=False).params
    ins = insample_report(panel, cfg, params, wm)
    with open(f"{OUT}/insample.json", "w") as fh:
        json.dump(ins, fh, indent=2)
    print(f"[B] in-sample co2 R2={ins['co2_r2']:.4f} MAPE={100*ins['co2_mape']:.1f}% "
          f"| demand R2={ins['demand_r2']:.4f} | mix MAE={ins['mix_mae']:.4f}")

    # Temporal backtest at 2015.
    res, preds, _ = run_temporal_backtest(panel, cfg, cutoff_year=2015, seeds=(0, 1, 2, 3, 4))
    with open(f"{OUT}/backtest_2015.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("\n[C] Holdout 2016-2022 forecast (per country-year):")
    print(f"    {'method':20s} {'MAPE%':>7s} {'RMSE_Mt':>9s} {'R2':>7s} {'skill':>7s} {'agg22%':>8s}")
    for name, m in sorted(res["methods"].items(), key=lambda kv: kv[1]["mape"]):
        agg = 100 * res["aggregate"]["final_year_pct_err"][name]
        print(f"    {name:20s} {100*m['mape']:7.2f} {m['rmse_mt']:9.1f} "
              f"{m['r2']:7.3f} {m['skill_vs_persistence']:7.3f} {agg:+8.2f}")

    # Robustness: a second cutoff.
    res12, _, _ = run_temporal_backtest(panel, cfg, cutoff_year=2012, seeds=(0, 1, 2), verbose=False)
    with open(f"{OUT}/backtest_2012.json", "w") as fh:
        json.dump(res12, fh, indent=2)

    # Calibration.
    cal = rolling_conformal(panel, cfg, cutoffs=[2011, 2013, 2015, 2017], seeds=(0, 1), verbose=True)
    with open(f"{OUT}/calibration.json", "w") as fh:
        json.dump(cal, fh, indent=2)
    print(f"\n[D] calibrated 80% interval leave-one-cutoff-out coverage = "
          f"{cal['loco_coverage_80']:.3f}")
    print("[ok] tables written to", OUT)


if __name__ == "__main__":
    main()
