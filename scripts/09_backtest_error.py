"""Planning ranges with long-horizon backtest error carried in (eval/propagate.py).

Inputs (no model is rerun):
  results/tables/structural.json  trajectories of the stabilised validation-tied models
                                  (BAU, max effort fitted, max effort open loop; SSP2)
  results/tables/plan.json        the selected model under SSP1-5
  results/sweep_long*/<config>/*  cached long-horizon backtests (06_sweep --phase longval
                                  and longtest), one trend-rate error delta per cutoff x seed

Every planning path of a model is corrected with each of its own configuration's deltas
(pooled over cutoffs and seeds), with the error growing linearly to 2050 (main case) or
capped at the longest backtest horizon (sensitivity).

Summaries. Each range is given two ways:
  q10-q90   10th to 90th percentile over all cases (model x dynamics x delta). This is
            the headline: it is stable as backtests are added.
  envelope  min to max over all cases. It widens by construction as draws are added.
Neither is a confidence interval; the deltas come from overlapping windows.
Both are reported for the expanded sample (11 cutoffs x seeds 0-3) and, for comparison,
for the previous sample (cutoffs 1998/2000/2003/2005 x seeds 0-1).

Budget shares printed here compare fossil+industry CO2 2023-2050 with the 50% budgets
only (a lenient lower-bound accounting); scripts/11_budgets.py gives the full comparison
with budget uncertainty, land-use CO2 and post-2050 emissions.

Usage:  python scripts/09_backtest_error.py
Writes results/tables/planning_with_backtest_error.json.
"""
import json

import numpy as np

from carbon_world_model.climate import tcre
from carbon_world_model.config import default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.eval.configs import SELECTED
from carbon_world_model.eval.propagate import backtest_drifts, correct_path

SEEDS = (0, 1, 2, 3)
SAMPLES = {
    "expanded": lambda d: True,
    "previous": lambda d: d["cutoff"] in (1998, 2000, 2003, 2005) and d["seed"] in (0, 1),
}
PATHS = {"bau": "bau_global_mt", "max_fitted": "max_global_mt",
         "max_open_loop": "open_loop_global_mt"}
MAX_KINDS = ("max_fitted", "max_open_loop")


def global_cum(path_mt: np.ndarray) -> float:
    """60-country Mt path (index 0 = 2022) -> global cumulative 2023-2050 (Gt)."""
    return tcre.scale_to_global(float(np.sum(path_mt[1:])) / 1000.0)


def ranges(values) -> dict:
    v = np.asarray(list(values), dtype=float)
    return {"q10_q90": [float(np.quantile(v, 0.1)), float(np.quantile(v, 0.9))],
            "envelope": [float(v.min()), float(v.max())], "n": int(v.size)}


def main() -> None:
    panel = Panel.load(default_config())
    clim = default_config().climate
    b15, b20 = tcre.remaining_budget_gtco2(1.5, clim), tcre.remaining_budget_gtco2(2.0, clim)
    st = json.load(open("results/tables/structural.json"))
    configs = sorted({m["config"] for m in st["models"]} | {SELECTED})
    deltas = {c: backtest_drifts(panel, c, seeds=SEEDS) for c in configs}
    h_cap = max(d["horizon"] for ds in deltas.values() for d in ds)

    cases = []  # one row per model x path x delta x variant
    for m in st["models"]:
        for kind, key in PATHS.items():
            path = np.asarray(m[key])
            base = {"config": m["config"], "seed": m["seed"], "path": kind}
            cases.append({**base, "variant": "none", "cum_gt": global_cum(path),
                          "y2050_mt": float(path[-1])})
            for d in deltas[m["config"]]:
                for variant, cap in (("linear", None), ("capped", h_cap)):
                    c = correct_path(path, d["delta"], cap)
                    cases.append({**base, "variant": variant, "delta": d["delta"],
                                  "bt_cutoff": d["cutoff"], "bt_seed": d["seed"],
                                  "cum_gt": global_cum(c), "y2050_mt": float(c[-1])})

    def in_sample(c, sample):
        return c["variant"] == "none" or SAMPLES[sample]({"cutoff": c["bt_cutoff"],
                                                          "seed": c["bt_seed"]})

    def summary(variant: str, sample: str) -> dict:
        rows = [c for c in cases if c["variant"] == variant and in_sample(c, sample)]
        out = {kind: {"global_cum_gt": ranges(c["cum_gt"] for c in rows if c["path"] == kind),
                      "y2050_mt": ranges(c["y2050_mt"] for c in rows if c["path"] == kind)}
               for kind in PATHS}
        mx = [c for c in rows if c["path"] in MAX_KINDS]
        out["max_effort_cases"] = len(mx)
        out["max_effort_share_below_1p5"] = float(np.mean([c["cum_gt"] < b15 for c in mx]))
        out["max_effort_share_above_2p0"] = float(np.mean([c["cum_gt"] > b20 for c in mx]))
        out["bau_share_above_2p0"] = float(np.mean([c["cum_gt"] > b20 for c in rows
                                                    if c["path"] == "bau"]))
        for kind in MAX_KINDS:
            r = [c for c in mx if c["path"] == kind]
            out[f"{kind}_share_below_1p5"] = float(np.mean([c["cum_gt"] < b15 for c in r]))
        return out

    # Selected model across SSP1-5 (driver scenario) x its own deltas, expanded sample.
    plan = json.load(open("results/tables/plan.json"))
    ssp = {}
    for kind, key in (("bau", "bau_global_mt"), ("max_fitted", "max_global_mt"),
                      ("max_open_loop", "open_loop_max_global_mt")):
        base, err = [], []
        for name, s in plan["scenarios"].items():
            if name.startswith("SSP"):
                path = np.asarray(s[key])
                base.append(global_cum(path))
                err += [global_cum(correct_path(path, d["delta"])) for d in deltas[SELECTED]]
        ssp[kind] = {"without_error": ranges(base), "with_error": ranges(err),
                     "share_below_1p5": float(np.mean(np.array(err) < b15))}

    summ = {sample: {v: summary(v, sample) for v in ("none", "linear", "capped")}
            for sample in SAMPLES}
    delta_stats = {c: {"n": len(ds), "q10_q90": ranges(d["delta"] for d in ds)["q10_q90"],
                       "envelope": ranges(d["delta"] for d in ds)["envelope"],
                       "mean": float(np.mean([d["delta"] for d in ds]))}
                   for c, ds in deltas.items()}
    out = {"budget_1p5_gt": b15, "budget_2p0_gt": b20, "h_cap": h_cap, "seeds": list(SEEDS),
           "selected": SELECTED, "deltas": deltas, "delta_stats": delta_stats,
           "summary": summ, "selected_ssp1to5": ssp, "cases": cases}
    with open("results/tables/planning_with_backtest_error.json", "w") as fh:
        json.dump(out, fh, indent=2)

    for c, s in delta_stats.items():
        print(f"{c:28s} n={s['n']:2d} delta %/yr: mean {100*s['mean']:+.2f}  q10-q90 "
              f"{100*s['q10_q90'][0]:+.2f}..{100*s['q10_q90'][1]:+.2f}  envelope "
              f"{100*s['envelope'][0]:+.2f}..{100*s['envelope'][1]:+.2f}")
    print(f"\nglobal cumulative 2023-2050 CO2 (Gt), SSP2, stabilised tied models; budgets "
          f"1.5C {b15:.0f}, 2C {b20:.0f}. Ranges: q10-q90 [envelope]")
    for sample in SAMPLES:
        print(f" sample: {sample}")
        for v, lab in (("none", "model as fitted"), ("linear", "+ error, linear to 2050"),
                       ("capped", f"+ error, capped at {h_cap} y")):
            s = summ[sample][v]
            def fmt(kind):
                r = s[kind]["global_cum_gt"]
                return (f"{r['q10_q90'][0]:5.0f}-{r['q10_q90'][1]:5.0f} "
                        f"[{r['envelope'][0]:4.0f}-{r['envelope'][1]:4.0f}]")
            print(f"  {lab:26s} BAU {fmt('bau')} | max fitted {fmt('max_fitted')} | "
                  f"open loop {fmt('max_open_loop')} | below 1.5C: fitted "
                  f"{100*s['max_fitted_share_below_1p5']:4.1f}% open {100*s['max_open_loop_share_below_1p5']:4.1f}% "
                  f"| max above 2C {100*s['max_effort_share_above_2p0']:.1f}% | BAU above 2C "
                  f"{100*s['bau_share_above_2p0']:.0f}%")
    print(f"\nselected model ({SELECTED}), SSP1-5 x its {len(deltas[SELECTED])} deltas:")
    for kind, r in ssp.items():
        w = r["with_error"]
        print(f"  {kind:14s} without error {r['without_error']['envelope'][0]:5.0f}-"
              f"{r['without_error']['envelope'][1]:5.0f} | with error q10-q90 {w['q10_q90'][0]:5.0f}-"
              f"{w['q10_q90'][1]:5.0f} [{w['envelope'][0]:4.0f}-{w['envelope'][1]:4.0f}] | below 1.5C "
              f"{100*r['share_below_1p5']:.1f}%")
    print("[ok] wrote results/tables/planning_with_backtest_error.json")


if __name__ == "__main__":
    main()
