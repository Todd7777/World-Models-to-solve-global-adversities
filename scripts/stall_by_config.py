"""Which configurations stall mature economies' efficiency gains, on the validation folds.

Models trained on data up to 2005-2009 forecast too little decline in the energy intensity
of 12 mature economies (README, Limitations; scripts/transfer_diagnostics.py measures it
for the selected configuration). This script measures the same quantities for every
configuration of a validation sweep from its cached job outputs, and fits nothing:

- the 12 economies' forecast trend in log energy intensity (per seed; %/yr over the
  6-year horizon), against the realised and pre-cutoff (10-year) trends;
- their final-year CO2 error, and their share of the model's per-country MAPE gap to
  trend-Kaya over all 60 countries (more than 100%: the other 48 together beat it);
- every pair of configurations in eval/configs.py that differ only in the calendar-time
  features (`time_features`), with the change in that trend and in the selection score
  (seed-ensemble scores, as in the sweep's report; each seed's own score is recorded too).

Validation folds only (cutoffs 2005, 2007, 2009; 6-year horizons, so nothing after 2015 is
read). Needs the sweep's outputs for every configuration and seed asked for
(scripts/06_sweep.py --phase val), all trained for the same number of steps.

Usage:  python scripts/stall_by_config.py [--root results/sweep_val] [--seeds 0,1]
                                          [--configs a,b] [--out ...]
"""
import argparse
import json
import os

import numpy as np

from carbon_world_model.config import default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.eval import metrics as M
from carbon_world_model.eval.baselines import trend_kaya
from carbon_world_model.eval.configs import REGISTRY, ProvenanceError
from carbon_world_model.eval.rolling import (
    VAL_CUTOFFS,
    VAL_HORIZON,
    evaluate,
    job_path,
    load_jobs,
    summarise,
)
from carbon_world_model.model.world_model import _year_index

MATURE = ("United Kingdom", "Hungary", "Belgium", "Denmark", "Finland", "Greece",
          "United States", "Italy", "Sweden", "France", "Germany", "Japan")  # as transfer_diagnostics.py


def _country_mape(pred, true, mask, i):
    return M.mape(pred[i:i + 1], true[i:i + 1], mask[i:i + 1])


def time_feature_pairs(names):
    """(with, without) pairs among `names` whose overrides differ only in time_features."""
    pairs = []
    for a in names:
        ma, ta = REGISTRY[a]
        if ma.get("time_features", True) is False:
            continue
        for b in names:
            mb, tb = REGISTRY[b]
            if mb.get("time_features", True) is False and ta == tb and \
                    {k: v for k, v in mb.items() if k != "time_features"} == ma:
                pairs.append((a, b))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/sweep_val")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--configs", default=None, help="default: every config in <root>/report.json")
    ap.add_argument("--out", default="results/tables/stall_by_config.json")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.configs:
        names = a.configs.split(",")
    else:
        with open(os.path.join(a.root, "report.json")) as fh:
            names = sorted({k.split("|")[0] for k in json.load(fh)})
    missing = [job_path(a.root, n, co, s) for n in names for co in VAL_CUTOFFS for s in seeds
               if not os.path.exists(job_path(a.root, n, co, s))]
    if missing:
        raise SystemExit(f"{len(missing)} job outputs missing, e.g. {missing[0]}. Run "
                         f"scripts/06_sweep.py --phase val --configs {','.join(names)} "
                         f"--seeds {a.seeds} --root {a.root} first.")
    steps = set()
    for n in names:
        _, fit = load_jobs(a.root, n, VAL_CUTOFFS, seeds)
        steps.add(fit["train"]["steps"])
    if len(steps) > 1:
        raise ProvenanceError(f"refusing to compare configurations trained for different step "
                              f"counts ({sorted(steps)})")

    panel = Panel.load(default_config())
    mature = [panel.countries.index(n) for n in MATURE]
    lI = np.log(panel.primary_energy) - np.log(panel.gdp)
    H = VAL_HORIZON
    out = {"root": a.root, "seeds": seeds, "steps": steps.pop(), "cutoffs": list(VAL_CUTOFFS),
           "horizon": H, "mature": list(MATURE), "observed": {}, "configs": {}}
    for co in VAL_CUTOFFS:
        k = _year_index(panel, co)
        out["observed"][co] = {
            "realised_pct_per_year": float(100 * np.mean([(lI[i, k + H] - lI[i, k]) / H for i in mature])),
            "pre_cutoff_10y_pct_per_year": float(100 * np.mean([(lI[i, k] - lI[i, k - 10]) / 10 for i in mature])),
        }
    for n in names:
        w = summarise(evaluate(panel, n, VAL_CUTOFFS, seeds, a.root))["world_model"]
        row = {"time_features": REGISTRY[n][0].get("time_features", True), "score": w["score"],
               "country_mape": w["country_mape"], "agg_abs_err_mean": w["agg_abs_err_mean"],
               "score_by_seed": {s: summarise(evaluate(panel, n, VAL_CUTOFFS, [s], a.root))
                                 ["world_model"]["score"] for s in seeds},
               "by_cutoff": {}}
        for co in VAL_CUTOFFS:
            k = _year_index(panel, co)
            true, mask = panel.co2[:, k + 1:k + H + 1], panel.mask[:, k + 1:k + H + 1]
            tk = trend_kaya(panel, co)[:, k + 1:k + H + 1]
            tr, err, share = {}, {}, {}
            for s in seeds:
                run = np.load(job_path(a.root, n, co, s))
                D, fc = run["D_realized"], run["co2_realized"][:, 1:]
                tr[s] = float(100 * np.mean([(np.log(D[i, -1] / panel.gdp[i, k + H])
                                              - np.log(D[i, 0] / panel.gdp[i, k])) / H for i in mature]))
                err[s] = float(np.mean([fc[i, -1] / true[i, -1] - 1 for i in mature]))
                gap = np.array([_country_mape(fc, true, mask, i) - _country_mape(tk, true, mask, i)
                                for i in range(panel.n_countries)])
                share[s] = float(gap[mature].sum() / gap.sum()) if gap.sum() > 0 else None
            row["by_cutoff"][co] = {"mature_trend_pct_per_year_by_seed": tr,
                                    "mature_final_year_co2_error_by_seed": err,
                                    "mature_share_of_gap_to_trend_kaya_by_seed": share}
        row["mature_trend_pct_per_year"] = float(np.mean(
            [v for c in row["by_cutoff"].values() for v in c["mature_trend_pct_per_year_by_seed"].values()]))
        out["configs"][n] = row
    out["realised_pct_per_year"] = float(np.mean([o["realised_pct_per_year"] for o in out["observed"].values()]))
    out["time_feature_pairs"] = [
        {"with": t, "without": f,
         "mature_trend_change_pct_per_year": out["configs"][f]["mature_trend_pct_per_year"]
         - out["configs"][t]["mature_trend_pct_per_year"],
         "score_change": out["configs"][f]["score"] - out["configs"][t]["score"]}
        for t, f in time_feature_pairs(names)]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"realised mature trend {out['realised_pct_per_year']:+.2f} %/yr (mean over cutoffs)")
    print(f"{'config':22s} time {'score':>6s}  mature trend %/yr (mean over cutoffs and seeds)")
    for n, r in sorted(out["configs"].items(), key=lambda kv: kv[1]["score"]):
        print(f"{n:22s} {'on ' if r['time_features'] else 'off'}  {r['score']:6.2f}  {r['mature_trend_pct_per_year']:+.2f}")
    for p in out["time_feature_pairs"]:
        print(f"time features off: {p['with']} -> {p['without']}: trend {p['mature_trend_change_pct_per_year']:+.2f} "
              f"%/yr, score {p['score_change']:+.2f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
