"""Run the planning models to 2100 instead of assuming what happens after 2050.

scripts/11_budgets.py needs CO2 until net zero and assumed a linear decline from each
path's 2050 level to net zero in 2070. Here the stabilised validation-tied models are
run under SSP2 (drivers to 2100) for business-as-usual and maximum feasible effort
(fitted dynamics and open loop), as fitted and with the model's and the drivers'
measured errors (paired by cutoff, as in scripts/10).

Rules fixed before running:
  * error corrections (emission drifts, driver perturbations) are frozen at their 2050
    values: the errors were measured over 14-20 years, and extending their linear growth
    to 78 years would scale some countries' GDP by factors of up to ~200;
  * total CO2 = model fossil+industry CO2 2023-2100 + land-use CO2 falling linearly to
    zero in 2050. No case reaches zero by 2100, so this is a LOWER bound on CO2 before
    net zero and the likelihoods are UPPER bounds;
  * process CO2 (cement, flaring; held at its 2022 level under business-as-usual) is cut
    under maximum effort either by the history-based lever ('historical':
    plan/scenario.process_bound, -2.7%/yr) or along published technology roadmaps
    ('technology': plan/scenario.technology_process_floor; cement -20% by 2030 and near
    zero by 2050, routine flaring ended by 2030).
Negative emissions (climate/cdr.py): maximum feasible NOVEL CDR (logistic scale-up from the
measured 0.002 GtCO2/yr in 2025 at the measured 40%/yr, saturating at a ceiling) is
subtracted from annual net CO2. Ceilings: the scenario level of 3.5 GtCO2/yr (1.75 and 7 as
sensitivities), or a bottom-up supply under land and cost limits (AR6 WGIII Table 12.6
costs and potentials; price caps of USD 100/200/300 per t; biomass-based CDR capped at
1.1/2.4/5.3 GtCO2/yr). Each is subtracted
under both accountings (model to 2100; assumed linear tail to zero gross CO2 in 2070).
With removals net CO2 can turn negative, so budgets are compared with PEAK cumulative CO2;
if net CO2 is still positive in 2100 the peak is a lower bound.
Beyond 2050 the runs are long extrapolations into states (fossil shares far below any
observed) that the model has never seen; treat them as a check on the assumed tail, not
as a projection.

Usage:  python scripts/12_to_2100.py
Writes results/tables/to_2100.json.
"""
import dataclasses
import json

import numpy as np

from carbon_world_model.climate import budgets as B
from carbon_world_model.climate import cdr as CDR
from carbon_world_model.climate import tcre
from carbon_world_model.config import FUELS, default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.eval.configs import SELECTED
from carbon_world_model.eval.propagate import (
    backtest_drifts, correct_path, driver_error_vectors, perturb_drivers,
)
from carbon_world_model.plan import optimize as O
from carbon_world_model.plan.scenario import build_scenario
from carbon_world_model.train.fit import load_trained

END, H_CAP, SEEDS = 2100, 28, (0, 1, 2, 3)
KINDS = ("bau", "max_fitted", "max_open_loop")
TARGETS = (1.5, 1.7, 2.0)
FOS = [FUELS.index(f) for f in ("coal", "oil", "gas")]


def run(wm, params, scen) -> dict:
    out = {}
    for kind, fn in (("bau", O.business_as_usual), ("max_fitted", O.max_ambition),
                     ("max_open_loop", O.max_ambition_open_loop)):
        r = fn(wm, params, scen)
        E = r.shares * r.D[:, :, None]
        out[kind] = {"global_mt": r.global_co2,
                     "fossil_share": E[:, :, FOS].sum((0, 2)) / E.sum((0, 2))}
    return out


def summarise_path(path_mt, years, k2050) -> dict:
    g = np.asarray(path_mt)
    gl = lambda x: float(tcre.scale_to_global(float(x)) if np.ndim(x) == 0 else tcre.scale_to_global(x))  # noqa: E731
    glob = tcre.scale_to_global(g / 1000.0)
    below = lambda th: next((int(y) for y, v in zip(years, glob) if v < th), None)  # noqa: E731
    return {"cum_2050_gt": gl(g[1:k2050 + 1].sum() / 1000), "cum_2100_gt": gl(g[1:].sum() / 1000),
            "y2050_gt": gl(g[k2050] / 1000), "y2100_gt": gl(g[-1] / 1000),
            "first_year_below_10gt": below(10.0), "first_year_below_5gt": below(5.0)}


def cdr_accounting(path_mt, years, k2050) -> dict:
    """Peak cumulative CO2 (from 2023), whether it is reached by 2100, and the net-zero
    year, for each CDR ceiling and each post-2050 accounting."""
    yrs = np.asarray(years)[1:]
    fossil = tcre.scale_to_global(np.asarray(path_mt)[1:] / 1000.0)
    n50 = k2050                                          # 2023..2050 = first n50 entries
    land = np.zeros_like(fossil)
    land[:n50] = B.LAND_USE_CO2_2024 * np.linspace(1.0, 0.0, n50)
    tail = fossil.copy()
    after = yrs > 2050
    tail[after] = fossil[n50 - 1] * np.clip(1.0 - (yrs[after] - 2050) / 20.0, 0.0, None)
    out = {}
    for ck in ("none", *CDR.CDR_VARIANTS):
        rem = np.zeros_like(fossil) if ck == "none" else CDR.novel_cdr_path(yrs, CDR.CDR_VARIANTS[ck])
        out[ck] = {}
        for acc, gross in (("model", fossil), ("tail", tail)):
            net = gross + land - rem
            peak, reached = CDR.peak_cumulative(net)
            out[ck][acc] = {"peak_gt": peak, "peak_reached": reached,
                            "net_zero_year": CDR.net_zero_year(yrs, net)}
    return out


def main() -> None:
    panel = Panel.load(default_config())
    st = json.load(open("results/tables/structural.json"))
    naive = driver_error_vectors(panel, "naive_long")
    land = B.land_use_co2(mode="linear_to_zero")
    cases, fitted_paths = [], {}
    for m in st["models"]:
        cfg, wm, params = load_trained(panel, f"results/model/structural/{m['config']}_s{m['seed']}.pkl")
        drifts = backtest_drifts(panel, m["config"], seeds=SEEDS)
        base = build_scenario(panel, cfg, end_year=END, ssp="SSP2")
        tech_p_max = build_scenario(panel, cfg, end_year=END, ssp="SSP2",
                                    process_lever="technology").p_max
        years = base.years
        k2050 = int(np.where(years == 2050)[0][0])
        variants = [("as_fitted", None, base)] + [
            ("paired", v, perturb_drivers(base, v["delta_gdp"], v["delta_pop"], h_cap=H_CAP))
            for v in naive]
        for vname, vec, scen in variants:
            for proc in ("historical", "technology"):
                sc = scen if proc == "historical" else dataclasses.replace(scen, p_max=tech_p_max)
                res = run(wm, params, sc)
                if vname == "as_fitted" and proc == "historical":
                    fitted_paths[f"{m['config']}_s{m['seed']}"] = {
                        k: {"global_mt": np.asarray(v["global_mt"]).tolist(),
                            "fossil_share": np.asarray(v["fossil_share"]).tolist()}
                        for k, v in res.items()}
                for kind in KINDS:
                    path = np.asarray(res[kind]["global_mt"])
                    ds = [None] if vec is None else [d for d in drifts if d["cutoff"] == vec["cutoff"]]
                    for d in ds:
                        p = path if d is None else correct_path(path, d["delta"], h_cap=H_CAP)
                        s = summarise_path(p, years, k2050)
                        acc = cdr_accounting(p, years, k2050)
                        # without removals the peak is the total: must match the earlier accounting
                        assert np.isclose(acc["none"]["model"]["peak_gt"], s["cum_2100_gt"] + land, rtol=1e-6)
                        assert np.isclose(acc["none"]["tail"]["peak_gt"], B.total_until_net_zero(
                            s["cum_2050_gt"], s["y2050_gt"], "linear_to_zero", 2070), rtol=1e-6)
                        cases.append({"config": m["config"], "seed": m["seed"], "set": vname, "cdr": acc,
                                      "process": proc, "kind": kind,
                                      "fossil_share_2100": float(res[kind]["fossil_share"][-1]),
                                      **s,
                                      "total_model_to_2100": float(s["cum_2100_gt"] + land),
                                      "total_tail_2070": float(B.total_until_net_zero(
                                          s["cum_2050_gt"], s["y2050_gt"], "linear_to_zero", 2070))})

    def summary(rows) -> dict:
        out = {}
        for key in ("cum_2100_gt", "y2100_gt", "y2050_gt", "fossil_share_2100",
                    "total_model_to_2100", "total_tail_2070"):
            v = np.array([r[key] for r in rows])
            out[key] = {"q10_q90": [float(np.quantile(v, .1)), float(np.quantile(v, .9))],
                        "envelope": [float(v.min()), float(v.max())]}
        out["share_reaching_5gt_by_2100"] = float(np.mean([r["first_year_below_5gt"] is not None for r in rows]))
        for acc in ("total_model_to_2100", "total_tail_2070"):
            tot = np.array([r[acc] for r in rows])
            for t in TARGETS:
                lik = B.likelihood_below(t, tot)
                out[f"{acc}|{t}"] = {"median": float(np.median(lik)),
                                     "q10_q90": [float(np.quantile(lik, .1)), float(np.quantile(lik, .9))],
                                     "share_ge_50": float(np.mean(lik >= 0.5))}
        for ck in ("none", *CDR.CDR_VARIANTS):
            for acc in ("model", "tail"):
                peak = np.array([r["cdr"][ck][acc]["peak_gt"] for r in rows])
                nz = [r["cdr"][ck][acc]["net_zero_year"] for r in rows]
                key = f"cdr|{ck}|{acc}"
                out[key] = {"peak_q10_q90": [float(np.quantile(peak, .1)), float(np.quantile(peak, .9))],
                            "share_net_zero_by_2100": float(np.mean([y is not None for y in nz])),
                            "median_net_zero_year": (float(np.median([y for y in nz if y is not None]))
                                                     if any(y is not None for y in nz) else None)}
                for t in TARGETS:
                    lik = B.likelihood_below(t, peak)
                    out[f"{key}|{t}"] = {"median": float(np.median(lik)),
                                         "q10_q90": [float(np.quantile(lik, .1)), float(np.quantile(lik, .9))],
                                         "share_ge_50": float(np.mean(lik >= 0.5))}
        out["n"] = len(rows)
        return out

    summ = {}
    for vname in ("as_fitted", "paired"):
        for proc in ("historical", "technology"):
            for subset in ("all", "selected"):
                for kind in KINDS:
                    rows = [c for c in cases if c["set"] == vname and c["process"] == proc
                            and c["kind"] == kind and (subset == "all" or c["config"] == SELECTED)]
                    summ[f"{vname}|{proc}|{subset}|{kind}"] = summary(rows)
    cdr_info = {k: {"ceiling_gt": v} for k, v in CDR.CDR_VARIANTS.items()}
    for k in cdr_info:
        if k.startswith("cost"):
            price = float(k[4:7])
            land = CDR.LAND_CAPS[k.split("land_")[1]]
            cdr_info[k].update(price_cap=price, land_cap=land, **CDR.cdr_spending_and_energy(price, land))
    out = {"end_year": END, "h_cap": H_CAP, "years": [int(y) for y in years], "cdr_variants": cdr_info,
           "fitted_paths": fitted_paths, "summary": summ, "cases": cases}
    with open("results/tables/to_2100.json", "w") as fh:
        json.dump(out, fh, indent=2)

    for vname in ("as_fitted", "paired"):
        for proc in ("historical", "technology"):
            for subset in ("all", "selected"):
                print(f"\n[{vname} | process {proc} | {subset} models]  q10-q90 [envelope]")
                for kind in KINDS:
                    s = summ[f"{vname}|{proc}|{subset}|{kind}"]
                    q = lambda k: f"{s[k]['q10_q90'][0]:6.1f}-{s[k]['q10_q90'][1]:6.1f}"  # noqa: E731
                    lk = " | ".join(
                        f"{t}C to2100 {100*s[f'total_model_to_2100|{t}']['median']:3.0f}% "
                        f"tail2070 {100*s[f'total_tail_2070|{t}']['median']:3.0f}%" for t in (1.7, 2.0))
                    print(f"  {kind:14s} 2100 level {q('y2100_gt')} Gt/yr | fossil share {q('fossil_share_2100')} | "
                          f"total to 2100 {q('total_model_to_2100')} vs tail-2070 {q('total_tail_2070')} | "
                          f"<5Gt by 2100: {100*s['share_reaching_5gt_by_2100']:3.0f}% | {lk}")
    print("\n[negative emissions] max effort, paired errors; peak-cumulative likelihood of staying below; "
          "share of cases reaching net zero by 2100 (median year)")
    for proc in ("historical", "technology"):
        for subset in ("all", "selected"):
            for kind in ("max_fitted", "max_open_loop"):
                s = summ[f"paired|{proc}|{subset}|{kind}"]
                for acc in ("tail", "model"):
                    cells = []
                    for ck in ("none", "central", "cost100_land_central", "cost100_land_low",
                               "cost200_land_central", "cost300_land_central"):
                        k = f"cdr|{ck}|{acc}"
                        nzy = s[k]["median_net_zero_year"]
                        cells.append(f"{ck} 1.5C {100*s[k + '|1.5']['median']:2.0f}% 2C {100*s[k + '|2.0']['median']:2.0f}% "
                                     f"NZ {100*s[k]['share_net_zero_by_2100']:3.0f}% ({'-' if nzy is None else int(nzy)})")
                    print(f"  {proc:10s} {subset:8s} {kind:13s} {acc:5s} | " + " | ".join(cells))
    print("[ok] wrote results/tables/to_2100.json")


if __name__ == "__main__":
    main()
