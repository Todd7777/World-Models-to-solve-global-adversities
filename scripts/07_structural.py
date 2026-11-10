"""Structural (model-form) uncertainty of the planning results.

Validation folds use 6-year horizons; planning runs 28 years and asks counterfactual
questions. Several configurations are statistically tied on validation (their scores
differ by less than the spread between single seeds of one configuration), yet they
disagree strongly beyond the validation horizon and in how they respond to the
planner's controls. No data in this project can choose between them. So instead of
reporting one model, this script reports planning results across all of them, each
under the fitted dynamics and under the open-loop counterfactual (no learned reaction
to the controls, see plan/optimize.max_ambition_open_loop).

Tied set: every configuration whose validation score (results/sweep_val/report.json)
is within TIE_TOL points of the best. The rule uses validation folds only.

Usage:  python scripts/07_structural.py [--seeds 0,1] [--jobs 4] [--ssp SSP2]
Writes results/model/structural/<config>_s<seed>.pkl (+ .json) and
results/tables/structural.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time

TIE_TOL = 0.5  # validation-score points; single seeds of one config differ by ~1.8
ROOT = "results/model/structural"
VAL_REPORT = "results/sweep_val/report.json"


FAMILIES = {
    "all": lambda c: True,
    "stab": lambda c: "_stab" in c,          # trained with the stability term
    "unconstrained": lambda c: "_stab" not in c,
}


def tied_configs(path: str = VAL_REPORT, tol: float = TIE_TOL,
                 family: str = "all") -> list[tuple[str, float]]:
    """Configs in `family` within `tol` of the best validation score over ALL configs."""
    rep = json.load(open(path))
    scores = {k.split("|")[0]: v["summary"]["world_model"]["score"]
              for k, v in rep.items() if k.endswith("|realized")}
    best = min(scores.values())
    return sorted(((c, s) for c, s in scores.items()
                   if s <= best + tol and FAMILIES[family](c)), key=lambda x: x[1])


def ckpt(config: str, seed: int) -> str:
    return os.path.join(ROOT, f"{config}_s{seed}.pkl")


def train_one(config: str, seed: int) -> None:
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                                       "intra_op_parallelism_threads=1")
    from carbon_world_model.config import default_config
    from carbon_world_model.data.panel import Panel
    from carbon_world_model.eval.configs import make_config
    from carbon_world_model.train.fit import fit, save_checkpoint
    panel = Panel.load(default_config())
    cfg = make_config(config)
    r = fit(panel, cfg, seed=seed, verbose=False)
    save_checkpoint(r.params, ckpt(config, seed), config,
                    {"seed": seed, "steps": cfg.train.steps,
                     "final_loss": float(r.history[-1]["loss"])})
    print(f"[trained] {config} s{seed} loss={r.history[-1]['loss']:.4f}", flush=True)


def train_missing(jobs: list[tuple[str, int]], n_parallel: int) -> None:
    todo = [(c, s) for c, s in jobs if not os.path.exists(ckpt(c, s))]
    running, t0 = [], time.time()
    while todo or running:
        while todo and len(running) < n_parallel:
            c, s = todo.pop(0)
            running.append(subprocess.Popen([sys.executable, __file__, "--train-one", c, str(s)]))
        time.sleep(1)
        for p in [p for p in running if p.poll() is not None]:
            if p.returncode != 0:
                raise SystemExit(f"training failed: {p.args}")
            running.remove(p)
    print(f"[structural] {len(jobs)} checkpoints ready ({time.time()-t0:.0f}s)")


def analyse(jobs, ssp: str) -> dict:
    import numpy as np
    from carbon_world_model.config import default_config
    from carbon_world_model.data.panel import Panel
    from carbon_world_model.train.fit import load_trained
    from carbon_world_model.plan.scenario import build_scenario
    from carbon_world_model.plan import optimize as O
    from carbon_world_model.plan.diagnostics import control_retention
    from carbon_world_model.climate import tcre

    panel = Panel.load(default_config())
    clim = default_config().climate
    ch, us = panel.countries.index("China"), panel.countries.index("United States")
    models, years = [], None
    for c, s in jobs:
        cfg, wm, params = load_trained(panel, ckpt(c, s))
        scen = build_scenario(panel, cfg, end_year=2050, ssp=ssp)
        years = scen.years
        bau, mx = O.business_as_usual(wm, params, scen), O.max_ambition(wm, params, scen)
        ol = O.max_ambition_open_loop(wm, params, scen)
        k35 = list(scen.years).index(2035)
        growth = np.diff(np.log(bau.global_co2))
        models.append({
            "config": c, "seed": s,
            "bau_global_mt": bau.global_co2.tolist(), "max_global_mt": mx.global_co2.tolist(),
            "open_loop_global_mt": ol.global_co2.tolist(),
            "bau_2050_mt": float(bau.global_co2[-1]), "max_2050_mt": float(mx.global_co2[-1]),
            "open_loop_2050_mt": float(ol.global_co2[-1]),
            "max_cut_2050": float(1 - mx.global_co2[-1] / bau.global_co2[-1]),
            "open_loop_cut_2050": float(1 - ol.global_co2[-1] / bau.global_co2[-1]),
            "global_cum_bau_gt": tcre.scale_to_global(bau.cum_gt),
            "global_cum_max_gt": tcre.scale_to_global(mx.cum_gt),
            "global_cum_open_loop_gt": tcre.scale_to_global(ol.cum_gt),
            "avoided_warming_max_degC": tcre.avoided_warming(bau.cum_gt, mx.cum_gt, clim)["central"],
            "avoided_warming_open_loop_degC":
                tcre.avoided_warming(bau.cum_gt, ol.cum_gt, clim)["central"],
            "retention": control_retention(bau, mx, ol, scen),
            "china_bau_change_2022_2035": float(bau.co2[ch, k35] / bau.co2[ch, 0] - 1),
            "us_bau_change_2022_2035": float(bau.co2[us, k35] / bau.co2[us, 0] - 1),
            "bau_max_annual_growth": float(growth.max()),
            "bau_min_annual_growth": float(growth.min()),
        })

    def rng(key, f=lambda m, k: m[k]):
        v = [f(m, key) for m in models]
        return [float(min(v)), float(max(v))]

    summary = {k: rng(k) for k in (
        "bau_2050_mt", "max_2050_mt", "open_loop_2050_mt", "max_cut_2050", "open_loop_cut_2050",
        "global_cum_bau_gt", "global_cum_max_gt", "global_cum_open_loop_gt",
        "avoided_warming_max_degC", "avoided_warming_open_loop_degC",
        "china_bau_change_2022_2035", "us_bau_change_2022_2035")}
    summary["clean_retention"] = rng("clean_retention", lambda m, k: m["retention"][k])
    summary["intensity_retention"] = rng("intensity_retention", lambda m, k: m["retention"][k])
    lo_cum = min(min(m["global_cum_max_gt"], m["global_cum_open_loop_gt"]) for m in models)
    summary["lowest_max_effort_global_cum_gt"] = float(lo_cum)
    summary["budget_1p5_gt"] = tcre.remaining_budget_gtco2(1.5, clim)
    summary["budget_2p0_gt"] = tcre.remaining_budget_gtco2(2.0, clim)
    return {"ssp": ssp, "years": [int(y) for y in years], "models": models, "summary": summary}


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--train-one":
        train_one(sys.argv[2], int(sys.argv[3]))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--ssp", default="SSP2")
    ap.add_argument("--family", default="all", choices=sorted(FAMILIES))
    a = ap.parse_args()
    tied = tied_configs(family=a.family)
    if not tied:
        raise SystemExit(f"no {a.family} configuration is tied on validation")
    print(f"[structural] tied on validation (within {TIE_TOL} of best): "
          + ", ".join(f"{c} ({s:.2f})" for c, s in tied))
    jobs = [(c, int(s)) for c, _ in tied for s in a.seeds.split(",")]
    os.makedirs(ROOT, exist_ok=True)
    train_missing(jobs, a.jobs)
    out = analyse(jobs, a.ssp)
    out["tied"] = [{"config": c, "val_score": s} for c, s in tied]
    out["tie_tol"] = TIE_TOL
    out["family"] = a.family
    out_path = ("results/tables/structural.json" if a.family == "all"
                else f"results/tables/structural_{a.family}.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\n{'model':24s} {'BAU2050':>8s} {'MAX2050':>8s} {'OL2050':>8s} {'cut':>5s} {'OLcut':>6s} "
          f"{'cumMAX':>7s} {'cumOL':>7s} {'retC':>5s} {'retE':>5s} {'China35':>8s} {'US35':>6s}")
    for m in out["models"]:
        print(f"{m['config']+' s'+str(m['seed']):24s} {m['bau_2050_mt']/1e3:8.1f} {m['max_2050_mt']/1e3:8.1f} "
              f"{m['open_loop_2050_mt']/1e3:8.1f} {100*m['max_cut_2050']:4.0f}% {100*m['open_loop_cut_2050']:5.0f}% "
              f"{m['global_cum_max_gt']:7.0f} {m['global_cum_open_loop_gt']:7.0f} "
              f"{m['retention']['clean_retention']:5.2f} {m['retention']['intensity_retention']:5.2f} "
              f"{100*m['china_bau_change_2022_2035']:+7.0f}% {100*m['us_bau_change_2022_2035']:+5.0f}%")
    s = out["summary"]
    print(f"\nranges across {len(out['models'])} models ({a.ssp}):")
    for k, v in s.items():
        print(f"  {k:36s} {v}")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
