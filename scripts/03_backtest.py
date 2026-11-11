"""Selection-independent evaluation tables: the physics layer's reconstruction of
reported CO2, and the in-sample (free-running) fidelity of the trained full-data model.
Writes emissions_reconstruction.json and insample.json to results/tables/.

This script used to also write a temporal backtest (cutoffs 2015 and 2012) and rolling
conformal calibration. Commit 3fc2559 removed that protocol because it leaked data from
after the cutoff (emission factors, normalisation statistics and static features were
computed from all years). The leak-free replacements are the rolling-origin sweep
(scripts/06_sweep.py: validation folds for selection, test folds for reporting) and
scripts/03_evaluate.py, which is run once a configuration has been selected.

Refuses to run, before writing anything, if the committed checkpoint was not trained with
the default configuration it is run under here: results/tables/config.json, which
scripts/02_train.py writes with it, must match default_config(). Without a checkpoint, a
model is fitted here with the default configuration instead.

Usage:  python scripts/03_backtest.py
"""
import json
import os
import sys

from carbon_world_model.config import default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.model.world_model import WorldModel
from carbon_world_model.model.emissions import estimate_emission_factors, reconstruction_report
from carbon_world_model.train.fit import fit, load_params
from carbon_world_model.eval.backtest import insample_report
from carbon_world_model.eval.configs import ProvenanceError, check_default_checkpoint

OUT = "results/tables"
CHECKPOINT = "results/model/params_full.pkl"
CHECKPOINT_CONFIG = os.path.join(OUT, "config.json")  # written by 02_train.py with CHECKPOINT


def main() -> None:
    cfg = default_config()
    if os.path.exists(CHECKPOINT):  # check before anything is written
        check_default_checkpoint(CHECKPOINT_CONFIG)
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
        params = load_params(CHECKPOINT)
    except FileNotFoundError:
        params = fit(panel, cfg, wm=wm, verbose=False).params
    ins = insample_report(panel, cfg, params, wm)
    with open(f"{OUT}/insample.json", "w") as fh:
        json.dump(ins, fh, indent=2)
    print(f"[B] in-sample co2 R2={ins['co2_r2']:.4f} MAPE={100*ins['co2_mape']:.1f}% "
          f"| demand R2={ins['demand_r2']:.4f} | mix MAE={ins['mix_mae']:.4f}")
    print("[ok] tables written to", OUT)
    print("Out-of-sample tables: scripts/06_sweep.py, then scripts/03_evaluate.py --config <selected>")


if __name__ == "__main__":
    try:
        main()
    except ProvenanceError as e:
        sys.exit(f"03_backtest.py: {e}")
