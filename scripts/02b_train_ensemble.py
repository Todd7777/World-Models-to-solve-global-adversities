"""Train the seed ensemble that scripts/04_plan.py plans with: the selected configuration
(default_config()) fitted to the full data with several training seeds.

Planning numbers from one checkpoint depend on its training seed
(scripts/seed_sensitivity.py), so 04_plan.py evaluates every pathway under each member
and optimises its plans against all of them (plan/ensemble.py). Writes
results/model/ensemble/params_seed<k>.pkl and manifest.json. Seed 0 is not retrained:
it is copied from results/model/params_full.pkl, after checking that the checkpoint was
trained with the default configuration, so the committed checkpoint is always a member.

Usage:  python scripts/02b_train_ensemble.py [--seeds 0-9]
"""
import argparse
import json
import sys
import time

from carbon_world_model.config import SELECTED_CONFIG
from carbon_world_model.data.panel import Panel
from carbon_world_model.eval.backtest import insample_report
from carbon_world_model.eval.configs import ProvenanceError, check_default_checkpoint
from carbon_world_model.model.world_model import WorldModel
from carbon_world_model.plan.ensemble import ENSEMBLE_DIR, save_ensemble
from carbon_world_model.train.fit import fit, load_params

CHECKPOINT = "results/model/params_full.pkl"
CHECKPOINT_CONFIG = "results/tables/config.json"  # written by 02_train.py with CHECKPOINT
TRAIN_HISTORY = "results/tables/train_history.json"


def parse_seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9", help="training seeds, e.g. 0-9 or 0,1,2")
    ap.add_argument("--out", default=ENSEMBLE_DIR)
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    if 0 not in seeds:
        ap.error("the ensemble must include seed 0, the committed checkpoint")
    cfg = check_default_checkpoint(CHECKPOINT_CONFIG)  # before anything is written
    panel = Panel.load(cfg)
    wm = WorldModel.build(panel, cfg)
    members, info = {}, {}
    for s in seeds:
        t0 = time.time()
        if s == 0:
            params = load_params(CHECKPOINT)
            with open(TRAIN_HISTORY) as fh:
                loss = json.load(fh)[-1]["loss"]
            source = CHECKPOINT
        else:
            r = fit(panel, cfg, wm=wm, seed=s, verbose=False)
            params, loss, source = r.params, r.history[-1]["loss"], "trained"
        ins = insample_report(panel, cfg, params, wm)
        members[s] = params
        info[s] = {"source": source, "final_loss": loss, "insample_co2_r2": ins["co2_r2"],
                   "insample_co2_mape": ins["co2_mape"]}
        print(f"seed {s}: {source}, final loss {loss:.4f}, in-sample CO2 R2 {ins['co2_r2']:.4f} "
              f"MAPE {100 * ins['co2_mape']:.1f}% ({time.time() - t0:.0f}s)", flush=True)
    save_ensemble(a.out, cfg, members, info)
    print(f"[ok] {SELECTED_CONFIG} ensemble of {len(seeds)} seeds -> {a.out}")


if __name__ == "__main__":
    try:
        main()
    except ProvenanceError as e:
        sys.exit(f"02b_train_ensemble.py: {e}")
