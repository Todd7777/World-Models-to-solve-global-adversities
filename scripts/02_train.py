"""Train the Carbon World Model on the full historical window and save parameters.

The configuration should be the one selected on the VALIDATION folds of the rolling-origin
sweep (scripts/06_sweep.py --phase val); its name is stored with the checkpoint so every
later script rebuilds the same model structure.

Usage:  python scripts/02_train.py --config <selected> [--seed 0]
"""
import argparse
import json
import os

from carbon_world_model.config import default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.eval.configs import REGISTRY, SELECTED, make_config
from carbon_world_model.train.fit import DEFAULT_CHECKPOINT, fit, save_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=SELECTED, choices=sorted(REGISTRY))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = make_config(a.config)
    panel = Panel.load(default_config())
    res = fit(panel, cfg, seed=a.seed, verbose=True)

    os.makedirs("results/tables", exist_ok=True)
    save_checkpoint(res.params, DEFAULT_CHECKPOINT, a.config,
                    {"seed": a.seed, "steps": cfg.train.steps,
                     "train_years": [int(panel.years[0]), int(panel.years[-1])],
                     "final_loss": float(res.history[-1]["loss"])})
    with open("results/tables/train_history.json", "w") as fh:
        json.dump(res.history, fh, indent=2)
    cfg.to_json("results/tables/config.json")
    print(f"[ok] config={a.config} seed={a.seed} -> {DEFAULT_CHECKPOINT}")


if __name__ == "__main__":
    main()
