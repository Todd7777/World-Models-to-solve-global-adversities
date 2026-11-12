"""Planning ranges with GDP / population driver error carried in (eval/propagate.py).

The planning runs use SSP drivers. Scenario spread (SSP1-5) is not forecast error: each
SSP is itself a projection that will be wrong. This script measures how wrong such
projections have been, as joint per-country trend-rate errors, and reruns the stabilised
validation-tied models (results/tables/structural.json) under SSP2 with the drivers
perturbed accordingly:
  naive_long  11 vectors: trailing-10-year-growth projections from the long backtest
              cutoffs 1997-2007 (14-20 years)
  ssp_v9       5 vectors: the 2014-vintage SSP1-5 projections against 2015-2022

Each perturbed run is then combined with the model's own emission backtest error
(eval/propagate.backtest_drifts):
  paired       (headline) each naive driver vector from cutoff c with the same
               configuration's emission drifts from the same cutoff c (4 seeds): the errors
               the two actually had together, historically
  independent  (sensitivity) every driver vector with every emission drift

Ranges are 10th-90th percentiles over cases with envelopes alongside; not confidence
intervals. Driver errors are used unscaled: over 2015-2022 the naive trend's aggregate
GDP error was slightly larger than the 2014 SSP2's.

Usage:  python scripts/10_driver_error.py
Writes results/tables/planning_with_driver_error.json.
"""
import json

import numpy as np

from carbon_world_model.climate import tcre
from carbon_world_model.config import default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.eval.configs import SELECTED
from carbon_world_model.eval.propagate import (
    backtest_drifts, correct_path, driver_error_vectors, perturb_drivers,
)
from carbon_world_model.plan import optimize as O
from carbon_world_model.plan.scenario import build_scenario
from carbon_world_model.train.fit import load_trained

SEEDS = (0, 1, 2, 3)
KINDS = ("bau", "max_fitted", "max_open_loop")
MAX_KINDS = ("max_fitted", "max_open_loop")


def global_cum(path_mt) -> float:
    return tcre.scale_to_global(float(np.sum(np.asarray(path_mt)[1:])) / 1000.0)


def ranges(values) -> dict:
    v = np.asarray(list(values), dtype=float)
    return {"q10_q90": [float(np.quantile(v, 0.1)), float(np.quantile(v, 0.9))],
            "envelope": [float(v.min()), float(v.max())], "n": int(v.size)}


def run_paths(wm, params, scen) -> dict:
    return {"bau": O.business_as_usual(wm, params, scen).global_co2,
            "max_fitted": O.max_ambition(wm, params, scen).global_co2,
            "max_open_loop": O.max_ambition_open_loop(wm, params, scen).global_co2}


def main() -> None:
    panel = Panel.load(default_config())
    clim = default_config().climate
    b15, b20 = tcre.remaining_budget_gtco2(1.5, clim), tcre.remaining_budget_gtco2(2.0, clim)
    st = json.load(open("results/tables/structural.json"))
    vectors = {s: driver_error_vectors(panel, s) for s in ("naive_long", "ssp_v9")}
    drifts = {}

    runs = []  # one row per model x driver vector (None = unperturbed SSP2)
    for m in st["models"]:
        cfg, wm, params = load_trained(panel, f"results/model/structural/{m['config']}_s{m['seed']}.pkl")
        if m["config"] not in drifts:
            drifts[m["config"]] = backtest_drifts(panel, m["config"], seeds=SEEDS)
        scen = build_scenario(panel, cfg, end_year=2050, ssp="SSP2")
        runs.append({"config": m["config"], "seed": m["seed"], "source": "none", "cutoff": None,
                     "paths": run_paths(wm, params, scen)})
        for src, vecs in vectors.items():
            for v in vecs:
                ps = perturb_drivers(scen, v["delta_gdp"], v["delta_pop"])
                runs.append({"config": m["config"], "seed": m["seed"], "source": src,
                             "cutoff": v["cutoff"], "label": v["label"],
                             "paths": run_paths(wm, params, ps)})

    cases = []  # (kind, cum) rows per combination
    for r in runs:
        for kind in KINDS:
            base = {"config": r["config"], "seed": r["seed"], "kind": kind,
                    "driver": r["source"], "driver_cutoff": r["cutoff"]}
            path = np.asarray(r["paths"][kind])
            cases.append({**base, "combo": f"driver:{r['source']}", "cum_gt": global_cum(path)})
            if r["source"] != "naive_long":
                continue
            for d in drifts[r["config"]]:
                c = global_cum(correct_path(path, d["delta"]))
                cases.append({**base, "combo": "both:independent", "cum_gt": c,
                              "emission_cutoff": d["cutoff"], "emission_seed": d["seed"]})
                if d["cutoff"] == r["cutoff"]:
                    cases.append({**base, "combo": "both:paired", "cum_gt": c,
                                  "emission_cutoff": d["cutoff"], "emission_seed": d["seed"]})

    combos = ("driver:none", "driver:naive_long", "driver:ssp_v9", "both:paired", "both:independent")
    summary = {}
    for combo in combos:
        rows = [c for c in cases if c["combo"] == combo]
        s = {kind: ranges(c["cum_gt"] for c in rows if c["kind"] == kind) for kind in KINDS}
        for kind in MAX_KINDS:
            r = [c["cum_gt"] for c in rows if c["kind"] == kind]
            s[f"{kind}_share_below_1p5"] = float(np.mean(np.array(r) < b15))
            s[f"{kind}_share_above_2p0"] = float(np.mean(np.array(r) > b20))
        s["bau_share_above_2p0"] = float(np.mean([c["cum_gt"] > b20 for c in rows if c["kind"] == "bau"]))
        sel = [c["cum_gt"] for c in rows if c["kind"] in MAX_KINDS and c["config"] == SELECTED]
        s["selected_max_share_above_2p0"] = float(np.mean(np.array(sel) > b20)) if sel else None
        summary[combo] = s

    vec_stats = {src: [{"label": v["label"], "horizon": v["horizon"],
                        "delta_gdp_aggregate": v["delta_gdp_aggregate"],
                        "delta_gdp_q10_q90": [float(np.quantile(v["delta_gdp"], q)) for q in (0.1, 0.9)],
                        "delta_pop_q10_q90": [float(np.quantile(v["delta_pop"], q)) for q in (0.1, 0.9)]}
                       for v in vecs] for src, vecs in vectors.items()}
    out = {"budget_1p5_gt": b15, "budget_2p0_gt": b20, "selected": SELECTED, "ssp": "SSP2",
           "driver_vectors": vectors, "driver_vector_stats": vec_stats, "summary": summary,
           "cases": cases}
    with open("results/tables/planning_with_driver_error.json", "w") as fh:
        json.dump(out, fh, indent=2)

    for src, vs in vec_stats.items():
        print(f"[{src}] aggregate GDP trend-rate error by projection (%/yr): "
              + " ".join(f"{100*v['delta_gdp_aggregate']:+.2f}" for v in vs))
    print(f"\nglobal cumulative 2023-2050 CO2 (Gt), SSP2, stabilised tied models; budgets 1.5C "
          f"{b15:.0f}, 2C {b20:.0f}. q10-q90 [envelope] (n cases)")
    for combo in combos:
        s = summary[combo]

        def fmt(kind):
            r = s[kind]
            return (f"{r['q10_q90'][0]:5.0f}-{r['q10_q90'][1]:5.0f} "
                    f"[{r['envelope'][0]:4.0f}-{r['envelope'][1]:4.0f}] ({r['n']})")
        print(f"  {combo:18s} BAU {fmt('bau')} | max fitted {fmt('max_fitted')} | open loop "
              f"{fmt('max_open_loop')}")
        print(f"  {'':18s} below 1.5C: fitted {100*s['max_fitted_share_below_1p5']:4.1f}% open "
              f"{100*s['max_open_loop_share_below_1p5']:4.1f}% | above 2C: fitted "
              f"{100*s['max_fitted_share_above_2p0']:4.1f}% open {100*s['max_open_loop_share_above_2p0']:4.1f}% "
              f"selected {100*(s['selected_max_share_above_2p0'] or 0):4.1f}% | BAU above 2C "
              f"{100*s['bau_share_above_2p0']:3.0f}%")
    print("[ok] wrote results/tables/planning_with_driver_error.json")


if __name__ == "__main__":
    main()
