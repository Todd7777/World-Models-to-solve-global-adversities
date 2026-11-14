"""An equal-weight combination of the world model's forecast and trend-Kaya's, on the validation folds.

Forecast combination (Bates & Granger 1969): the combined CO2 forecast for each country and year is the
plain average of the configuration's seed-ensemble forecast and trend-Kaya's. Equal weights are the
standard robust choice, so no weight is estimated and nothing is tuned on these folds. Other weights are
reported only to show how the score depends on the choice.

Validation folds only (cutoffs 2005, 2007, 2009; 6-year horizons, so nothing after 2015 is read), with
realised drivers. The script reads a sweep's cached forecasts and fits nothing. It scores everything the
way the sweep scores its methods (`rolling._metrics`, `summarise`), and checks that it reproduces the
sweep's own scores for the model alone and trend-Kaya. A combination is a forecasting device: it changes
neither the model nor the planner, whose business as usual is the network's alone.

Usage:  python scripts/combination.py [--config ms_full] [--root results/sweep_val_4seeds]
                                      [--seeds 0,1,2,3] [--out results/tables/combination.json]
"""
import argparse
import json
import os

import numpy as np

from carbon_world_model.config import SELECTED_CONFIG, default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.eval.baselines import trend_kaya
from carbon_world_model.eval.rolling import (
    VAL_CUTOFFS,
    _metrics,
    evaluate,
    job_path,
    load_jobs,
    summarise,
)

MATURE = ("United Kingdom", "Hungary", "Belgium", "Denmark", "Finland", "Greece",
          "United States", "Italy", "Sweden", "France", "Germany", "Japan")  # as transfer_diagnostics.py
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)   # the world model's share; 0.5 is the combination


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=SELECTED_CONFIG)
    ap.add_argument("--root", default="results/sweep_val_4seeds")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--out", default="results/tables/combination.json")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    missing = [job_path(a.root, a.config, co, s) for co in VAL_CUTOFFS for s in seeds
               if not os.path.exists(job_path(a.root, a.config, co, s))]
    if missing:
        raise SystemExit(f"{len(missing)} job outputs missing, e.g. {missing[0]}. Run scripts/06_sweep.py "
                         f"--phase val --configs {a.config} --seeds {a.seeds} --root {a.root} first.")
    _, fit = load_jobs(a.root, a.config, VAL_CUTOFFS, seeds)     # refuses outputs trained differently
    panel = Panel.load(default_config())
    china = panel.countries.index("China")
    mature = [panel.countries.index(n) for n in MATURE]

    def scored(seed_list, w):
        res = {"cutoffs": {}}
        final = {}
        for co in VAL_CUTOFFS:
            k = int(np.where(panel.years == co)[0][0])
            wm = np.mean([np.load(job_path(a.root, a.config, co, s))["co2_realized"]
                          for s in seed_list], axis=0)[:, 1:]
            H = wm.shape[1]
            true, mask = panel.co2[:, k + 1:k + H + 1], panel.mask[:, k + 1:k + H + 1]
            f = w * wm + (1 - w) * trend_kaya(panel, co)[:, k + 1:k + H + 1]
            res["cutoffs"][co] = {"m": _metrics(f, true, mask, china), "steps": fit["train"]["steps"]}
            final[co] = float(np.mean([f[i, -1] / true[i, -1] - 1 for i in mature]))
        s = summarise(res)["m"]
        by_cutoff = {co: 100 * (c["m"]["country_mape"] + c["m"]["agg_abs_err_mean"])
                     for co, c in res["cutoffs"].items()}
        return {"score": s["score"], "country_mape": s["country_mape"],
                "agg_abs_err_mean": s["agg_abs_err_mean"], "score_by_cutoff": by_cutoff,
                "mature_final_year_error_by_cutoff": final}

    sweep = summarise(evaluate(panel, a.config, VAL_CUTOFFS, seeds, a.root))
    out = {"config": a.config, "root": a.root, "seeds": seeds, "steps": fit["train"]["steps"],
           "weights_are_the_world_models_share": list(WEIGHTS),
           "seed_ensemble": {str(w): scored(seeds, w) for w in WEIGHTS},
           "single_seed": {s: {str(w): scored([s], w)["score"] for w in (0.0, 0.5, 1.0)} for s in seeds}}
    e = out["seed_ensemble"]
    assert abs(e["1.0"]["score"] - sweep["world_model"]["score"]) < 1e-9, "model score differs from the sweep's"
    assert abs(e["0.0"]["score"] - sweep["trend_kaya"]["score"]) < 1e-9, "trend-Kaya differs from the sweep's"

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    for w, lab in ((1.0, f"{a.config} alone"), (0.0, "trend-Kaya"), (0.5, "equal-weight combination")):
        v = e[str(w)]
        print(f"{lab:26s} score {v['score']:6.2f}  MAPE {100 * v['country_mape']:.2f}%  "
              f"|agg| {100 * v['agg_abs_err_mean']:.2f}%  by cutoff "
              + " ".join(f"{x:.2f}" for x in v["score_by_cutoff"].values()))
    print("single seeds, combination (model alone):",
          ", ".join(f"{s}: {v['0.5']:.2f} ({v['1.0']:.2f})" for s, v in out["single_seed"].items()))
    print("other weights, shown only for robustness:",
          ", ".join(f"{w}: {e[str(w)]['score']:.2f}" for w in WEIGHTS))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
