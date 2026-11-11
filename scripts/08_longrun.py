"""Long-run LEVEL checks (diagnostics; never used for model selection).

1. Long-horizon backtests: fit at an early cutoff and forecast 15-19 years to a fixed end
   year (eval/rolling.PHASES 'longval': 1998/2000 -> 2015, 'longtest': 2003/2005 -> 2022),
   scoring level errors, including cumulative CO2 over the horizon. Run the sweeps first:
     python scripts/06_sweep.py --phase longval  --configs <...> --seeds 0,1
     python scripts/06_sweep.py --phase longtest --configs <...> --seeds 0,1
2. Energy-ladder plausibility (eval/plausibility.py) of every structural-ensemble model's
   SSP2 business-as-usual 2050 projection (results/model/structural/*.pkl), with two
   reference extrapolations: constant energy intensity, and each country's 2012-2022
   intensity trend continued.

Usage:  python scripts/08_longrun.py
Writes results/tables/longrun.json.
"""
import glob
import json
import os

import numpy as np

from carbon_world_model.config import default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.eval.plausibility import ladder_check
from carbon_world_model.plan import optimize as O
from carbon_world_model.plan.scenario import build_scenario
from carbon_world_model.train.fit import load_trained

METHODS = ("world_model", "trend_kaya", "growth_persistence", "persistence", "frozen_kaya")
BIG = ("China", "United States", "India", "Russia", "Japan", "Germany")


def backtests() -> dict:
    out = {}
    for phase in ("longval", "longtest"):
        path = f"results/sweep_{phase}/report.json"
        if not os.path.exists(path):
            print(f"[skip] {path} missing")
            continue
        rep = json.load(open(path))
        rows = {}
        for key, r in rep.items():
            cfg = key.split("|")[0]
            s = r["summary"]
            rows[cfg] = {m: {k: s[m][k] for k in ("country_mape", "agg_err_final", "abs_agg_err_final",
                                                  "cum_err", "abs_cum_err", "china_err_final", "score")}
                         for m in METHODS if m in s}
            cuts = r["detail"]["cutoffs"]
            rows[cfg]["agg_err_by_cutoff"] = {
                co: {m: v[m]["agg_err_final"] for m in METHODS if m in v} for co, v in cuts.items()}
            rows[cfg]["cum_err_by_cutoff"] = {
                co: {m: v[m]["cum_err"] for m in METHODS if m in v} for co, v in cuts.items()}
            rows[cfg]["n_seeds_by_cutoff"] = {co: v["n_seeds"] for co, v in cuts.items()}
        out[phase] = rows
    return out


def ladder_reports(panel) -> dict:
    out = {}
    years_back = 10
    for path in sorted(glob.glob("results/model/structural/*.pkl")) + ["results/model/params_full.pkl"]:
        cfg, wm, params = load_trained(panel, path)
        scen = build_scenario(panel, cfg, end_year=2050, ssp="SSP2")
        bau = O.business_as_usual(wm, params, scen)
        gdp, pop = np.asarray(scen.gdp)[:, -1], np.asarray(scen.pop)[:, -1]
        name = "selected" if path.endswith("params_full.pkl") else os.path.basename(path)[:-4]
        out[name] = ladder_check(panel, gdp, pop, bau.D[:, -1])
        cfg_name = json.load(open(path[:-4] + ".json"))["config"]
        out[name]["config"] = cfg_name
        out[name]["family"] = "stab" if "_stab" in cfg_name else "unconstrained"
    # reference extrapolations, same SSP2 drivers
    I = np.log(panel.primary_energy / panel.gdp)
    slope = (I[:, -1] - I[:, -1 - years_back]) / years_back
    span = int(2050 - panel.years[-1])
    refs = {"ref_constant_intensity": np.exp(I[:, -1]) * gdp,
            "ref_own_trend_intensity": np.exp(I[:, -1] + slope * span) * gdp}
    for name, D in refs.items():
        out[name] = ladder_check(panel, gdp, pop, D)
        out[name]["family"] = "reference"
    return out


def main() -> None:
    panel = Panel.load(default_config())
    from carbon_world_model.eval.configs import SELECTED
    res = {"selected": SELECTED, "backtests": backtests(), "ladder": ladder_reports(panel)}
    with open("results/tables/longrun.json", "w") as fh:
        json.dump(res, fh, indent=2)

    for phase, rows in res["backtests"].items():
        base = next(iter(rows.values()))
        print(f"\n[{phase}] {len(base['cum_err_by_cutoff'])} cutoffs. Mean over cutoffs: |final agg err|%, "
              f"cumulative err% [min..max over cutoffs], China final%, MAPE%, seeds")

        def line(name, r, m):
            v = r[m]
            cc = [x[m] for x in r["cum_err_by_cutoff"].values()]
            seeds = sorted(set(r["n_seeds_by_cutoff"].values())) if m == "world_model" else "-"
            print(f"  {name:34s} {100*v['abs_agg_err_final']:6.1f} {100*v['cum_err']:+7.1f} "
                  f"[{100*min(cc):+6.1f}..{100*max(cc):+6.1f}] {100*v['china_err_final']:+7.1f} "
                  f"{100*v['country_mape']:6.1f}  {seeds}")
        for m in METHODS[1:]:
            line("baseline " + m, base, m)
        for cfg, r in rows.items():
            line(cfg, r, "world_model")

    idx = {c: panel.countries.index(c) for c in BIG}
    print("\n[ladder] SSP2 BAU 2050 vs history: flagged = outside the pooled 95% band of 28-year "
          "drifts; excess = drift beyond the country's own 1994-2022 drift, %/yr (energy-weighted)")
    for name, r in res["ladder"].items():
        ex = np.array(r["excess_drift_per_year"])
        big = " ".join(f"{c[:5]} {100*ex[i]:+.1f}" for c, i in idx.items())
        print(f"  {name:32s} flagged {100*r['energy_share_flagged']:4.1f}% of energy "
              f"({len(r['flagged_countries'])} ctry) | excess {100*r['energy_weighted_excess_per_year']:+5.2f}%/yr, "
              f"faster than own past: {100*r['energy_share_faster_than_own_past']:3.0f}% | {big}")
    print("[ok] wrote results/tables/longrun.json")


if __name__ == "__main__":
    main()
