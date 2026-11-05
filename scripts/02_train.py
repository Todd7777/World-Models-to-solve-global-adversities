"""Train the Carbon World Model on the full historical window and save parameters.

Usage:  python scripts/02_train.py
"""
import json
import os

from carbon_world_model.config import default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.train.fit import fit, save_params


def main() -> None:
    cfg = default_config()
    panel = Panel.load(cfg)
    res = fit(panel, cfg, verbose=True)

    os.makedirs("results/tables", exist_ok=True)
    save_params(res.params, "results/model/params_full.pkl")
    with open("results/tables/train_history.json", "w") as fh:
        json.dump(res.history, fh, indent=2)
    cfg.to_json("results/tables/config.json")
    print("[ok] saved params -> results/model/params_full.pkl")


if __name__ == "__main__":
    main()
