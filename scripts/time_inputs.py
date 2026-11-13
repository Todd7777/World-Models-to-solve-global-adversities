"""What the calendar-time inputs do, in backtests and in planning.

Configurations with `time_features` give the network calendar time, index / (T_ref - 1)
(0 in 1990, 1 in 2022), and its square. Without them, models trained up to 2005-2009
stall mature economies' efficiency gains in the validation backtests; with them they do
not (scripts/stall_by_config.py). In a backtest, calendar time runs on past the cutoff,
beyond the range the model was trained on. Planning holds it at its 2022 value
(`Scenario.tfrac`). This script runs a configuration with time inputs both ways (default
`ms_percap_damped`, the selected configuration plus time inputs), for seeds 0-3:

1. Validation backtests: cutoffs 2005, 2007 and 2009, 6-year horizons, so nothing after
   2015 is read. Each model is refitted as the sweep fits it and forecast two ways:
   - with calendar time running on, as the sweep does. Where the sweep's cached output
     exists, this must reproduce it bit for bit;
   - with calendar time held at the cutoff's value, as planning holds it at 2022.
   Both are scored as the sweep scores them, with the 12 mature economies' forecast
   energy-intensity trend. The selected configuration is scored from the sweep's cached
   outputs for comparison.
2. Planning business as usual: full-data fits under the heuristic drivers, with calendar
   time held at 2022, as planning does, and running on to 2050. The comparison is the
   planning ensemble's members with the same seeds (the selected configuration, without
   time inputs). It reports, against 2012-2022 observed:
   - the intensity trends of the mature economies and of the other 48 countries, for
     2022-2032 and 2032-2050;
   - total emissions in 2030 and 2050.

No test fold is involved. About 15 minutes with 4 workers. Writes
results/tables/time_inputs.json.

Usage:  python scripts/time_inputs.py [--config ms_percap_damped] [--seeds 0,1,2,3]
                                      [--root results/sweep_val_4seeds] [--jobs 4]
"""
import argparse
import dataclasses
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

# One XLA thread per worker, as in scripts/run_job.py, so the refits can reproduce the
# sweep's outputs bit for bit.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                                   "intra_op_parallelism_threads=1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from carbon_world_model.config import SELECTED_CONFIG, default_config  # noqa: E402
from carbon_world_model.data.panel import Panel  # noqa: E402
from carbon_world_model.eval.configs import REGISTRY, make_config  # noqa: E402
from carbon_world_model.eval.rolling import (  # noqa: E402
    VAL_CUTOFFS,
    VAL_HORIZON,
    _metrics,
    evaluate,
    horizon_for,
    job_path,
    summarise,
)
from carbon_world_model.model.world_model import WorldModel, _year_index  # noqa: E402
from carbon_world_model.plan import optimize as O  # noqa: E402
from carbon_world_model.plan.ensemble import load_ensemble  # noqa: E402
from carbon_world_model.plan.scenario import build_scenario  # noqa: E402

OUT = "results/tables/time_inputs.json"
ENSEMBLE = "results/model/ensemble"
CHECKPOINT = "results/model/params_full.pkl"
MATURE = ("United Kingdom", "Hungary", "Belgium", "Denmark", "Finland", "Greece",
          "United States", "Italy", "Sweden", "France", "Germany", "Japan")  # as transfer_diagnostics.py


def _full(job):
    """Full-data fit, as seed_sensitivity.py --config fits it."""
    from carbon_world_model.train.fit import fit
    name, seed = job
    cfg = make_config(name)
    r = fit(Panel.load(cfg), cfg, seed=seed, verbose=False)
    return "full", None, seed, jax.tree_util.tree_map(np.asarray, r.params)


def _backtest(job):
    """Fit on data <= cutoff as rolling.run_job does; forecast with calendar time running
    on (the sweep's forecast) and held at its value in the cutoff year."""
    from carbon_world_model.train.fit import fit
    name, cutoff, seed = job
    panel = Panel.load(default_config())
    H = horizon_for(panel, cutoff, VAL_HORIZON)
    r = fit(panel, make_config(name), train_year_max=cutoff, seed=seed, verbose=False)
    on = r.wm.forecast(r.params, panel, cutoff, end_year=cutoff + H)
    cap = float(r.wm.tfrac(_year_index(panel, cutoff)))
    held = r.wm.forecast(r.params, panel, cutoff, end_year=cutoff + H, tfrac_cap=cap)
    return "backtest", cutoff, seed, {"running_on": {k: np.asarray(on[k]) for k in ("co2", "D")},
                                      "held": {k: np.asarray(held[k]) for k in ("co2", "D")}}


def _mature_trend(panel, D, cutoff, rows):
    """Mean forecast trend in log energy intensity over the horizon, %/yr."""
    k = _year_index(panel, cutoff)
    H = D.shape[1] - 1
    return float(100 * np.mean([(np.log(D[i, -1] / panel.gdp[i, k + H])
                                 - np.log(D[i, 0] / panel.gdp[i, k])) / H for i in rows]))


def _bau(wm, params, scen, rows):
    bau = O.business_as_usual(wm, params, scen)
    intensity = np.log(bau.D) - np.log(scen.gdp)
    end = scen.Tf - 1

    def trend(group, a, b):
        return float(100 * np.mean([(intensity[i, b] - intensity[i, a]) / (b - a) for i in rows[group]]))

    return {"mature_pct_per_year_2022_2032": trend("mature", 0, 10),
            "mature_pct_per_year_2032_2050": trend("mature", 10, end),
            "others_pct_per_year_2022_2032": trend("others", 0, 10),
            "others_pct_per_year_2032_2050": trend("others", 10, end),
            "bau_2030_gt": float(bau.global_co2[8] / 1e3),
            "bau_2050_gt": float(bau.global_co2[-1] / 1e3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ms_percap_damped")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--root", default="results/sweep_val_4seeds",
                    help="validation sweep whose cached outputs the refits are checked against "
                         "and the selected configuration is scored from")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if REGISTRY[a.config][0].get("time_features", True) is False:
        ap.error(f"{a.config} has no calendar-time inputs")
    seeds = [int(s) for s in a.seeds.split(",")]
    cfg_sel = default_config()
    panel = Panel.load(cfg_sel)
    ens_seeds, members = load_ensemble(ENSEMBLE, cfg_sel, checkpoint=CHECKPOINT)
    missing = [s for s in seeds if s not in ens_seeds]
    if missing:
        raise SystemExit(f"the planning ensemble has no member for seeds {missing}")

    jobs = [(_full, (a.config, s)) for s in seeds]        # the longest first
    jobs += [(_backtest, (a.config, co, s)) for co in VAL_CUTOFFS for s in seeds]
    with ProcessPoolExecutor(max_workers=a.jobs, mp_context=mp.get_context("spawn")) as pool:
        results = [f.result() for f in [pool.submit(fn, job) for fn, job in jobs]]
    full = {s: p for kind, _, s, p in results if kind == "full"}
    bt = {(co, s): r for kind, co, s, r in results if kind == "backtest"}

    T = panel.n_years
    mature = [panel.countries.index(n) for n in MATURE]
    rows = {"mature": mature, "others": [i for i in range(panel.n_countries) if i not in mature]}
    china = panel.countries.index("China")
    lI = np.log(panel.primary_energy) - np.log(panel.gdp)
    wm = WorldModel.build(panel, make_config(a.config))
    steps = make_config(a.config).train.steps

    # 1. Validation backtests.
    def scored(variant, seed_list):
        res = {"config": a.config, "cutoffs": {}}
        for co in VAL_CUTOFFS:
            k = _year_index(panel, co)
            fc = np.mean([bt[(co, s)][variant]["co2"] for s in seed_list], axis=0)
            H = fc.shape[1] - 1
            res["cutoffs"][co] = {"world_model": _metrics(fc[:, 1:], panel.co2[:, k + 1:k + H + 1],
                                                          panel.mask[:, k + 1:k + H + 1], china),
                                  "steps": steps}
        return summarise(res)["world_model"]

    def variant_summary(variant):
        w = scored(variant, seeds)
        trend = {co: {s: _mature_trend(panel, bt[(co, s)][variant]["D"], co, mature) for s in seeds}
                 for co in VAL_CUTOFFS}
        return {"score": w["score"], "country_mape": w["country_mape"],
                "agg_abs_err_mean": w["agg_abs_err_mean"],
                "score_by_seed": {s: scored(variant, [s])["score"] for s in seeds},
                "mature_pct_per_year": trend,
                "mature_pct_per_year_mean": float(np.mean([v for c in trend.values() for v in c.values()]))}

    backtest = {"root": a.root, "running_on_reproduces_sweep_output": {}, "calendar_time": {},
                "realised_mature_pct_per_year": {}}
    for co in VAL_CUTOFFS:
        k = _year_index(panel, co)
        H = horizon_for(panel, co, VAL_HORIZON)
        backtest["calendar_time"][co] = {"trained_up_to": float(wm.tfrac(k)),
                                         "running_on_up_to": float(wm.tfrac(k + H))}
        backtest["realised_mature_pct_per_year"][co] = float(
            100 * np.mean([(lI[i, k + H] - lI[i, k]) / H for i in mature]))
        for s in seeds:
            path = job_path(a.root, a.config, co, s)
            backtest["running_on_reproduces_sweep_output"][f"{co}_s{s}"] = (
                bool(np.array_equal(bt[(co, s)]["running_on"]["co2"], np.load(path)["co2_realized"]))
                if os.path.exists(path) else None)
    backtest["calendar_time_running_on"] = variant_summary("running_on")
    backtest["calendar_time_held_at_cutoff"] = variant_summary("held")
    sel_paths = [job_path(a.root, SELECTED_CONFIG, co, s) for co in VAL_CUTOFFS for s in seeds]
    if all(os.path.exists(p) for p in sel_paths):
        w = summarise(evaluate(panel, SELECTED_CONFIG, VAL_CUTOFFS, seeds, a.root,
                               steps=cfg_sel.train.steps))["world_model"]
        trend = [_mature_trend(panel, np.load(job_path(a.root, SELECTED_CONFIG, co, s))["D_realized"],
                               co, mature) for co in VAL_CUTOFFS for s in seeds]
        backtest["selected"] = {"config": SELECTED_CONFIG, "score": w["score"],
                                "country_mape": w["country_mape"],
                                "agg_abs_err_mean": w["agg_abs_err_mean"],
                                "mature_pct_per_year_mean": float(np.mean(trend))}

    # 2. Planning business as usual.
    scen_sel = build_scenario(panel, cfg_sel, end_year=2050, ssp=None)
    wm_sel = WorldModel.build(panel, cfg_sel)
    held = build_scenario(panel, make_config(a.config), end_year=2050, ssp=None)
    assert np.all(held.tfrac == 1.0) and abs(wm.tfrac(T - 1) - 1.0) < 1e-12
    running_on = dataclasses.replace(held, tfrac=np.asarray(wm.tfrac(T - 1 + np.arange(held.Tf))))
    obs = (panel.co2 * panel.mask).sum(axis=0)
    planning = {"drivers": "heuristic",
                "calendar_time": {"held_at": 1.0, "running_on_up_to": float(running_on.tfrac[-1])},
                "observed_2012_2022": {f"{g}_pct_per_year": float(100 * np.mean(
                    [(lI[i, T - 1] - lI[i, T - 11]) / 10 for i in r])) for g, r in rows.items()},
                "observed_2022_gt": float(obs[-1] / 1e3), "by_seed": {}}
    for s in seeds:
        p = jax.tree_util.tree_map(jnp.asarray, full[s])
        planning["by_seed"][s] = {
            SELECTED_CONFIG: _bau(wm_sel, members[ens_seeds.index(s)], scen_sel, rows),
            f"{a.config}, calendar time held at 2022": _bau(wm, p, held, rows),
            f"{a.config}, calendar time running on": _bau(wm, p, running_on, rows)}
    planning["mean"] = {v: {k: float(np.mean([planning["by_seed"][s][v][k] for s in seeds]))
                            for k in planning["by_seed"][seeds[0]][v]}
                        for v in planning["by_seed"][seeds[0]]}

    out = {"config": a.config, "selected": SELECTED_CONFIG, "seeds": seeds, "mature": list(MATURE),
           "backtest": backtest, "planning_bau": planning}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)

    checks = list(backtest["running_on_reproduces_sweep_output"].values())
    print(f"refits reproduce the sweep's outputs: {sum(c is True for c in checks)}/{len(checks)} "
          f"({sum(c is None for c in checks)} not cached)")
    real = np.mean(list(backtest["realised_mature_pct_per_year"].values()))
    print(f"validation backtests, seeds {seeds} (realised mature trend {real:+.2f} %/yr):")
    lines = [(f"{a.config}, calendar time running on", backtest["calendar_time_running_on"]),
             (f"{a.config}, calendar time held at cutoff", backtest["calendar_time_held_at_cutoff"])]
    if "selected" in backtest:
        lines.append((f"{SELECTED_CONFIG} (no time inputs)", backtest["selected"]))
    for lab, v in lines:
        print(f"  {lab:46s} score {v['score']:6.2f}  MAPE {100 * v['country_mape']:.2f}%  "
              f"|agg| {100 * v['agg_abs_err_mean']:.2f}%  mature trend {v['mature_pct_per_year_mean']:+.2f} %/yr")
    o = planning["observed_2012_2022"]
    print(f"planning BAU, heuristic drivers (observed 2012-2022: mature {o['mature_pct_per_year']:+.2f}, "
          f"other 48 {o['others_pct_per_year']:+.2f} %/yr; {planning['observed_2022_gt']:.1f} Gt in 2022):")
    for v, m in planning["mean"].items():
        print(f"  {v:46s} mature {m['mature_pct_per_year_2022_2032']:+.2f}, others "
              f"{m['others_pct_per_year_2022_2032']:+.2f} %/yr (2022-2032); "
              f"BAU {m['bau_2030_gt']:.1f} Gt in 2030, {m['bau_2050_gt']:.1f} in 2050")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
