"""Final evaluation tables for a selected configuration (chosen on VALIDATION folds).

Reads the cached sweep jobs (results/sweep_val, results/sweep_test) and the trained
full-data model; writes JSON tables to results/tables/:
  emissions_reconstruction.json  physics layer vs reported CO2
  insample.json                  free-running fidelity of the final model
  ablation_val.json              every configuration on the validation folds
  test.json                      selected config vs baselines on the test folds,
                                 realised drivers and SSP-v9 ex-ante drivers
  china_decomposition.json       China's forecast error split into demand vs carbon
                                 intensity, per test cutoff
  driver_decomposition.json      how much of the ex-ante error comes from the SSP GDP path
  calibration.json               split-conformal intervals (val-calibrated, test-checked)
  transfer.json                  spatial transfer to held-out countries

Refuses to run unless results/tables/config.json, which scripts/02_train.py writes with
the checkpoint, matches make_config(<selected>): the in-sample table must describe the
selected configuration, and a checkpoint of another configuration may not even run under
it. Every sweep output read must have been trained for the same number of steps as the
checkpoint.

Usage:  python scripts/03_evaluate.py --config <selected>
"""
import argparse
import json
import os
import sys

import numpy as np

from carbon_world_model.config import default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.data.ssp import SSPS
from carbon_world_model.model.world_model import WorldModel
from carbon_world_model.model.emissions import estimate_emission_factors, reconstruction_report
from carbon_world_model.train.fit import load_params
from carbon_world_model.eval.configs import REGISTRY, ProvenanceError, check_checkpoint_config
from carbon_world_model.eval.rolling import (
    VAL_CUTOFFS, TEST_CUTOFFS, job_path, evaluate, summarise, load_jobs, describe_steps,
)
from carbon_world_model.eval.backtest import insample_report, conformal_split
from carbon_world_model.eval.baselines import trend_kaya
from carbon_world_model.eval.transfer import spatial_transfer

OUT = "results/tables"
CHECKPOINT = "results/model/params_full.pkl"
CHECKPOINT_CONFIG = os.path.join(OUT, "config.json")  # written by 02_train.py with CHECKPOINT
VAL_ROOT, TEST_ROOT = "results/sweep_val", "results/sweep_test"


def dump(obj, name):
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(obj, fh, indent=2)


def seeds_in(root, config, cutoff):
    return [s for s in range(10) if os.path.exists(job_path(root, config, cutoff, s))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(REGISTRY))
    ap.add_argument("--skip-transfer", action="store_true")
    a = ap.parse_args()
    # Check provenance before anything is written: the checkpoint must have been trained with
    # this config, and every sweep output read below for as many steps as the checkpoint.
    cfg = check_checkpoint_config(a.config, CHECKPOINT_CONFIG)
    steps = cfg.train.steps
    seeds = seeds_in(TEST_ROOT, a.config, TEST_CUTOFFS[0])
    cal_seeds = sorted(set(seeds_in(VAL_ROOT, a.config, VAL_CUTOFFS[0])) & set(seeds))
    try:
        test_runs, _ = load_jobs(TEST_ROOT, a.config, TEST_CUTOFFS, range(10), steps=steps)
        load_jobs(VAL_ROOT, a.config, VAL_CUTOFFS, cal_seeds, steps=steps)
    except ProvenanceError as e:
        raise ProvenanceError(f"{e} The checkpoint was trained for {steps} steps and these "
                              "tables must describe that training, so rerun the sweep for "
                              f"{a.config!r} at that length.") from None
    os.makedirs(OUT, exist_ok=True)
    panel = Panel.load(default_config())
    china = panel.countries.index("China")

    # 1. physics layer
    rec = reconstruction_report(panel, estimate_emission_factors(panel, "nnls"))
    dump(rec, "emissions_reconstruction.json")
    print(f"[A] physics reconstruction R2={rec['r2']:.4f} median err={100*rec['median_abs_pct_err']:.1f}%")

    # 2. in-sample fidelity of the final (full-data) model
    wm = WorldModel.build(panel, cfg)
    ins = insample_report(panel, cfg, load_params(CHECKPOINT), wm)
    dump(ins, "insample.json")
    print(f"[B] in-sample co2 R2={ins['co2_r2']:.4f} MAPE={100*ins['co2_mape']:.1f}% "
          f"demand R2={ins['demand_r2']:.4f} mix MAE={ins['mix_mae']:.4f}")

    # 3. ablation on validation folds (all configs that were swept)
    val_report = json.load(open(os.path.join(VAL_ROOT, "report.json")))
    abl = {k.split("|")[0]: v["summary"] for k, v in val_report.items() if k.endswith("|realized")}
    dump(abl, "ablation_val.json")
    print(f"[ablation] {len(abl)} configs on the validation folds, trained for "
          f"{describe_steps(v for k, v in val_report.items() if k.endswith('|realized'))}; "
          f"the checkpoint was trained for {steps}")

    # 4. test folds: selected config vs baselines, realised and ex-ante drivers
    test = {}
    for drv in ("realized", "ssp"):
        res = evaluate(panel, a.config, TEST_CUTOFFS, seeds, TEST_ROOT, driver=drv, steps=steps)
        test[drv] = {"summary": summarise(res), "detail": res}
    dump(test, "test.json")
    print(f"[C] test folds {TEST_CUTOFFS} (seeds {seeds}):")
    for drv in ("realized", "ssp"):
        for m in ("world_model", "trend_kaya", "growth_persistence", "persistence", "frozen_kaya"):
            v = test[drv]["summary"][m]
            print(f"    {drv:8s} {m:20s} MAPE {100*v['country_mape']:5.2f}%  |agg| "
                  f"{100*v['agg_abs_err_mean']:5.2f}%  aggFinal {100*v['agg_err_final']:+6.2f}%  "
                  f"China {100*v['china_err_final']:+6.2f}%  score {v['score']:.2f}")

    # 5. China decomposition (log error = demand part + carbon-intensity part)
    chd = {}
    for co in TEST_CUTOFFS:
        runs = test_runs[co]
        k = int(np.where(panel.years == co)[0][0])
        D = np.mean([r["D_realized"] for r in runs], 0)[china, -1]
        co2 = np.mean([r["co2_realized"] for r in runs], 0)[china, -1]
        H = int(runs[0]["H"])
        Dt, co2t = panel.primary_energy[china, k + H], panel.co2[china, k + H]
        tk = trend_kaya(panel, co)[china, k + H]
        chd[int(co)] = {"year": int(panel.years[k + H]),
                        "co2_err": float(co2 / co2t - 1), "demand_err": float(D / Dt - 1),
                        "carbon_intensity_err": float((co2 / D) / (co2t / Dt) - 1),
                        "trend_kaya_co2_err": float(tk / co2t - 1)}
    dump(chd, "china_decomposition.json")
    print("[D] China final-year error decomposition:",
          {c: {kk: round(100 * vv, 1) for kk, vv in v.items() if kk != 'year'} for c, v in chd.items()})

    # 6. ex-ante driver decomposition: SSP-v9 forecast vs realised-driver forecast
    drv = {}
    for co in TEST_CUTOFFS:
        runs = test_runs[co]
        if "co2_SSP2" not in runs[0]:
            continue
        real = np.mean([r["co2_realized"] for r in runs], 0)[:, -1].sum()
        per = {s: float(np.mean([r[f"co2_{s}"] for r in runs], 0)[:, -1].sum() / real - 1) for s in SSPS}
        drv[int(co)] = per
    dump(drv, "driver_decomposition.json")
    print("[E] aggregate final-year shift from using SSP-v9 instead of realised drivers:",
          {c: {s: round(100 * v, 1) for s, v in d.items()} for c, d in drv.items()})

    # 7. split-conformal calibration
    cal = conformal_split(panel, a.config, VAL_ROOT, TEST_ROOT, VAL_CUTOFFS, TEST_CUTOFFS,
                          cal_seeds, steps=steps)
    dump(cal, "calibration.json")
    print(f"[F] split-conformal 80% interval: test coverage {cal['test_coverage']:.3f}")

    # 8. spatial transfer
    if not a.skip_transfer:
        tr = spatial_transfer(panel, cfg, test_every=5, cutoff=2015, seed=0, verbose=False)
        dump(tr, "transfer.json")
        print(f"[G] transfer to 12 unseen countries: WM MAPE {100*tr['world_model']['mape']:.1f}% "
              f"R2 {tr['world_model']['r2']:.3f} | trend-Kaya MAPE {100*tr['trend_kaya']['mape']:.1f}%")
    with open(os.path.join(OUT, "selected_config.json"), "w") as fh:
        json.dump({"config": a.config}, fh)


if __name__ == "__main__":
    try:
        main()
    except ProvenanceError as e:
        sys.exit(f"03_evaluate.py: {e}")
