"""Carbon-budget verdicts with budget (TCRE, non-CO2) uncertainty and full CO2 accounting.

Earlier scripts compare 2023-2050 fossil and industry CO2 with a single budget number. A
remaining carbon budget is uncertain (published as likelihoods) and covers ALL CO2 until
net zero. This script turns every planning case into a likelihood of staying below 1.5,
1.7 and 2 C (climate/budgets.py, Forster et al. 2025 Table 8, rebased to 2023), after
completing the accounting.

Accounting variants (fixed before results were computed):
  headline       model fossil+industry 2023-2050 + land-use CO2 falling linearly from
                 4.2 GtCO2/yr to zero by 2050 + a post-2050 tail falling linearly from the
                 path's 2050 level to net zero in 2070
  lenient        fossil+industry 2023-2050 only (the project's earlier practice; a lower
                 bound on CO2 before net zero)
  net_zero_2060  as headline, but the tail reaches net zero in 2060
  land_constant  as headline, but land-use CO2 stays at 4.2 GtCO2/yr to 2050
Non-CO2: the published budgets move by about +/-200 GtCO2 with non-CO2 mitigation;
reported as sensitivities (+200 = deeper non-CO2 cuts).

Cases (from scripts/09 and 10; stabilised tied models, SSP2):
  as_fitted        the models as fitted
  emission_error   + long-horizon emission backtest error (44 drifts per config)
  both_paired      + GDP/population driver error, paired by cutoff (headline)
  both_independent + both, independent (sensitivity)

Usage:  python scripts/11_budgets.py
Writes results/tables/budget_likelihoods.json.
"""
import json

import numpy as np

from carbon_world_model.climate import budgets as B
from carbon_world_model.climate import tcre
from carbon_world_model.eval.configs import SELECTED

ACCOUNTING = {
    "headline": ("linear_to_zero", 2070),
    "lenient": ("zero", None),
    "net_zero_2060": ("linear_to_zero", 2060),
    "land_constant": ("constant", 2070),
}
TARGETS = (1.5, 1.7, 2.0)
KINDS = ("bau", "max_fitted", "max_open_loop")


def load_cases() -> dict:
    pe = json.load(open("results/tables/planning_with_backtest_error.json"))
    de = json.load(open("results/tables/planning_with_driver_error.json"))
    out = {"as_fitted": [], "emission_error": [], "both_paired": [], "both_independent": []}
    for c in pe["cases"]:
        row = {"config": c["config"], "kind": c["path"], "cum_gt": c["cum_gt"],
               "y2050_global_gt": tcre.scale_to_global(c["y2050_mt"] / 1000.0)}
        if c["variant"] == "none":
            out["as_fitted"].append(row)
        elif c["variant"] == "linear":
            out["emission_error"].append(row)
    for combo, key in (("both:paired", "both_paired"), ("both:independent", "both_independent")):
        out[key] = [{"config": c["config"], "kind": c["kind"], "cum_gt": c["cum_gt"],
                     "y2050_global_gt": c["y2050_global_gt"]}
                    for c in de["cases"] if c["combo"] == combo]
    return out


def stats(lik: np.ndarray, totals: np.ndarray, target: float, shift: float) -> dict:
    return {"median": float(np.median(lik)),
            "q10_q90": [float(np.quantile(lik, 0.1)), float(np.quantile(lik, 0.9))],
            "share_ge_50": float(np.mean(lik >= 0.5)), "share_ge_67": float(np.mean(lik >= 0.67)),
            "share_at_or_below_17": float(np.mean(lik <= 0.17 + 1e-9)),
            "total_gt_q10_q90": [float(np.quantile(totals, 0.1)), float(np.quantile(totals, 0.9))],
            "budget50_gt": B.budget_gt(target, 0.5, non_co2_shift=shift), "n": int(lik.size)}


def main() -> None:
    cases = load_cases()
    res = {}
    for set_name, rows in cases.items():
        for kind in KINDS:
            for subset in ("all", "selected"):
                rr = [r for r in rows if r["kind"] == kind and (subset == "all" or r["config"] == SELECTED)]
                if not rr:
                    continue
                for acc, (land, nz) in ACCOUNTING.items():
                    totals = np.array([B.total_until_net_zero(r["cum_gt"], r["y2050_global_gt"], land, nz)
                                       for r in rr])
                    for shift in (0.0, -B.NON_CO2_SHIFT_GT, B.NON_CO2_SHIFT_GT):
                        if shift != 0.0 and acc != "headline":
                            continue
                        for t in TARGETS:
                            lik = B.likelihood_below(t, totals, non_co2_shift=shift)
                            res[f"{set_name}|{kind}|{subset}|{acc}|{int(shift):+d}|{t}"] = stats(lik, totals, t, shift)
    old = {str(t): B.tcre_only_budget(t) for t in TARGETS}
    new = {str(t): {str(q): B.budget_gt(t, q) for q in B.LIKELIHOODS} for t in TARGETS}
    out = {"source": B.SOURCE, "budgets_from_2023": new, "old_tcre_only_budgets": old,
           "accounting": {k: {"land_use": v[0], "net_zero_year": v[1]} for k, v in ACCOUNTING.items()},
           "results": res}
    with open("results/tables/budget_likelihoods.json", "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"budgets from 2023 (GtCO2) at 17/33/50/67/83% [{B.SOURCE}]:")
    for t in TARGETS:
        print(f"  {t} C: " + " / ".join(f"{new[str(t)][str(q)]:.0f}" for q in B.LIKELIHOODS)
              + f"   (old TCRE-only formula: {old[str(t)]:.0f})")

    def show(set_name, subset, acc="headline", shift=0):
        print(f"\n[{set_name} | {subset} models | accounting={acc} | non-CO2 shift {shift:+d}] "
              "likelihood of staying below: median [q10-q90], share of cases >=50%")
        for kind in KINDS:
            key = f"{set_name}|{kind}|{subset}|{acc}|{shift:+d}"
            if f"{key}|1.5" not in res:
                continue
            tot = res[f"{key}|2.0"]["total_gt_q10_q90"]
            cells = []
            for t in TARGETS:
                s = res[f"{key}|{t}"]
                cells.append(f"{t}C {100*s['median']:3.0f}% [{100*s['q10_q90'][0]:2.0f}-{100*s['q10_q90'][1]:2.0f}] "
                             f"{100*s['share_ge_50']:3.0f}%")
            print(f"  {kind:14s} total {tot[0]:5.0f}-{tot[1]:5.0f} Gt | " + " | ".join(cells))

    for set_name in ("as_fitted", "emission_error", "both_paired", "both_independent"):
        show(set_name, "all")
    show("both_paired", "selected")
    for acc in ("lenient", "net_zero_2060", "land_constant"):
        show("both_paired", "all", acc)
    for shift in (-200, 200):
        show("both_paired", "all", "headline", shift)
    print("[ok] wrote results/tables/budget_likelihoods.json")


if __name__ == "__main__":
    main()
