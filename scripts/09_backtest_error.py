"""Planning ranges with long-horizon backtest error carried in (eval/propagate.py).

Inputs (no model is rerun):
  results/tables/structural.json  trajectories of the stabilised validation-tied models
                                  (BAU, max effort fitted, max effort open loop; SSP2)
  results/tables/plan.json        the selected model under SSP1-5
  results/sweep_long*/<config>/*  cached long-horizon backtests (06_sweep --phase longval
                                  and longtest), one trend-rate error delta per cutoff x seed

Every planning path of a model is corrected with each of its own configuration's deltas
(pooled over seeds), with the error growing linearly to 2050 (main case) or capped at the
longest backtest horizon (sensitivity). Ranges are envelopes over models, dynamics and
deltas: errors the model has actually made, not confidence intervals.

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

PATHS = {"bau": "bau_global_mt", "max_fitted": "max_global_mt",
         "max_open_loop": "open_loop_global_mt"}


def global_cum(path_mt: np.ndarray) -> float:
    """60-country Mt path (index 0 = 2022) -> global cumulative 2023-2050 (Gt)."""
    return tcre.scale_to_global(float(np.sum(path_mt[1:])) / 1000.0)


def envelope(values) -> list[float]:
    values = list(values)
    return [float(min(values)), float(max(values))]


def main() -> None:
    panel = Panel.load(default_config())
    clim = default_config().climate
    b15, b20 = tcre.remaining_budget_gtco2(1.5, clim), tcre.remaining_budget_gtco2(2.0, clim)
    st = json.load(open("results/tables/structural.json"))
    deltas = {}
    for m in st["models"]:
        if m["config"] not in deltas:
            deltas[m["config"]] = backtest_drifts(panel, m["config"])
    h_cap = max(d["horizon"] for ds in deltas.values() for d in ds)

    cases = []  # one row per model x path x delta x variant
    for m in st["models"]:
        for kind, key in PATHS.items():
            path = np.asarray(m[key])
            cases.append({"config": m["config"], "seed": m["seed"], "path": kind,
                          "variant": "none", "delta": 0.0,
                          "cum_gt": global_cum(path), "y2050_mt": float(path[-1])})
            for d in deltas[m["config"]]:
                for variant, cap in (("linear", None), ("capped", h_cap)):
                    c = correct_path(path, d["delta"], cap)
                    cases.append({"config": m["config"], "seed": m["seed"], "path": kind,
                                  "variant": variant, "delta": d["delta"],
                                  "cum_gt": global_cum(c), "y2050_mt": float(c[-1])})

    def summary(variant: str) -> dict:
        out = {}
        for kind in PATHS:
            rows = [c for c in cases if c["path"] == kind and c["variant"] == variant]
            out[kind] = {"global_cum_gt": envelope(c["cum_gt"] for c in rows),
                         "y2050_mt": envelope(c["y2050_mt"] for c in rows), "n_cases": len(rows)}
        mx = [c for c in cases if c["path"] in ("max_fitted", "max_open_loop")
              and c["variant"] == variant]
        out["max_effort_cases_below_1p5_budget"] = sum(c["cum_gt"] < b15 for c in mx)
        out["max_effort_cases_above_2p0_budget"] = sum(c["cum_gt"] > b20 for c in mx)
        out["max_effort_cases"] = len(mx)
        below = [c for c in mx if c["cum_gt"] < b15]
        out["below_1p5_from"] = sorted({(c["config"], c["path"], round(c["delta"], 4)) for c in below})
        return out

    # Selected model across SSP1-5 (driver scenario) x its own deltas.
    plan = json.load(open("results/tables/plan.json"))
    sel = deltas.get(SELECTED) or backtest_drifts(panel, SELECTED)
    ssp_rows = []
    for name, s in plan["scenarios"].items():
        if not name.startswith("SSP"):
            continue
        for kind, key in (("bau", "bau_global_mt"), ("max_fitted", "max_global_mt"),
                          ("max_open_loop", "open_loop_max_global_mt")):
            path = np.asarray(s[key])
            ssp_rows.append({"ssp": name, "path": kind, "delta": 0.0, "cum_gt": global_cum(path)})
            for d in sel:
                ssp_rows.append({"ssp": name, "path": kind, "delta": d["delta"],
                                 "cum_gt": global_cum(correct_path(path, d["delta"]))})
    ssp_summary = {kind: {"without_error": envelope(r["cum_gt"] for r in ssp_rows
                                                    if r["path"] == kind and r["delta"] == 0.0),
                          "with_error": envelope(r["cum_gt"] for r in ssp_rows if r["path"] == kind)}
                   for kind in ("bau", "max_fitted", "max_open_loop")}

    out = {"budget_1p5_gt": b15, "budget_2p0_gt": b20, "h_cap": h_cap,
           "deltas": deltas, "selected": SELECTED,
           "summary": {v: summary(v) for v in ("none", "linear", "capped")},
           "selected_ssp1to5": ssp_summary, "cases": cases}
    with open("results/tables/planning_with_backtest_error.json", "w") as fh:
        json.dump(out, fh, indent=2)

    for cfg, ds in deltas.items():
        print(f"{cfg:28s} deltas %/yr: " + " ".join(f"{100*d['delta']:+.2f}" for d in ds))
    print(f"\nglobal cumulative 2023-2050 CO2 (Gt), SSP2, {len(st['models'])} stabilised tied models; "
          f"budgets 1.5C {b15:.0f}, 2C {b20:.0f}")
    for v, lab in (("none", "model as fitted"), ("linear", "+ backtest error, linear to 2050"),
                   ("capped", f"+ backtest error, capped at {h_cap} y")):
        s = out["summary"][v]
        print(f"  {lab:36s} BAU {s['bau']['global_cum_gt'][0]:5.0f}-{s['bau']['global_cum_gt'][1]:5.0f} | "
              f"max fitted {s['max_fitted']['global_cum_gt'][0]:5.0f}-{s['max_fitted']['global_cum_gt'][1]:5.0f} | "
              f"max open loop {s['max_open_loop']['global_cum_gt'][0]:5.0f}-{s['max_open_loop']['global_cum_gt'][1]:5.0f} | "
              f"max-effort cases below 1.5C budget: {s['max_effort_cases_below_1p5_budget']}/{s['max_effort_cases']}, "
              f"above 2C: {s['max_effort_cases_above_2p0_budget']}")
        for row in s["below_1p5_from"]:
            print(f"      below 1.5C: {row}")
    print(f"\nselected model ({SELECTED}), SSP1-5 x its {len(sel)} deltas:")
    for kind, r in ssp_summary.items():
        print(f"  {kind:14s} without error {r['without_error'][0]:5.0f}-{r['without_error'][1]:5.0f} | "
              f"with error {r['with_error'][0]:5.0f}-{r['with_error'][1]:5.0f}")
    print("[ok] wrote results/tables/planning_with_backtest_error.json")


if __name__ == "__main__":
    main()
