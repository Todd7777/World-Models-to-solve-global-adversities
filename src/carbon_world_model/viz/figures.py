"""Publication-quality figures for the Carbon World Model.

Clean, colour-blind-safe styling (Okabe-Ito palette); light background; minimal
chartjunk. Each function returns a matplotlib Figure and is called by
scripts/05_make_figures.py.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito colour-blind-safe palette.
OI = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
    "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#111111",
    "grey": "#999999",
}
FUEL_COLORS = {
    "coal": "#4d4d4d", "oil": "#8c6d31", "gas": "#E69F00", "nuclear": "#CC79A7",
    "hydro": "#0072B2", "wind": "#56B4E9", "solar": "#F0E442", "biofuel": "#009E73",
    "other_renewable": "#7fd4b0",
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "legend.frameon": False, "figure.autolayout": False,
})


def _finish(fig):
    fig.tight_layout()
    return fig


def fig_reconstruction(panel, ef, rec) -> plt.Figure:
    from ..model.emissions import energy_to_co2
    pred = np.asarray(energy_to_co2(panel.energy, ef.as_full(), panel.process_co2))
    m = panel.mask.astype(bool)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax1.scatter(panel.co2[m], pred[m], s=6, alpha=0.35, color=OI["blue"], edgecolors="none")
    lim = [1, max(panel.co2[m].max(), pred[m].max()) * 1.2]
    ax1.plot(lim, lim, color=OI["red"], lw=1.2, ls="--", label="1:1")
    ax1.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
            xlabel="Reported CO2 (Mt)", ylabel="Mechanistic reconstruction (Mt)",
            title=f"Physics layer: R2 = {rec['r2']:.4f}")
    ax1.legend(loc="upper left")

    fuels = list(ef.fossil.keys())
    vals = [ef.fossil[f] for f in fuels]
    ax2.bar(fuels, vals, color=[FUEL_COLORS[f] for f in fuels])
    ax2.set(ylabel="Emission factor (MtCO2 / TWh)", title="Estimated emission factors")
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    return _finish(fig)


def fig_insample(panel, wm, params) -> plt.Figure:
    tr = wm.rollout(params, panel)
    co2 = np.asarray(tr["co2"])
    yrs = panel.years
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()
    # global total
    axes[0].plot(yrs, panel.co2.sum(0), color=OI["black"], lw=2, label="observed")
    axes[0].plot(yrs, co2.sum(0), color=OI["red"], lw=1.6, ls="--", label="model (free run)")
    axes[0].set_title("All 60 countries"); axes[0].legend()
    for ax, country in zip(axes[1:], ["China", "United States", "India", "Germany", "Russia"]):
        i = panel.country_index(country)
        ax.plot(yrs, panel.co2[i], color=OI["black"], lw=2)
        ax.plot(yrs, co2[i], color=OI["red"], lw=1.6, ls="--")
        ax.set_title(country)
    for ax in axes:
        ax.set_xlabel("year"); ax.set_ylabel("CO2 (Mt)")
    fig.suptitle("In-sample free-running reconstruction (1990-2022)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def fig_backtest(panel, bt) -> plt.Figure:
    yrs = np.array(bt["aggregate"]["years"])
    true = np.array(bt["aggregate"]["true_mt"])
    cutoff = bt["cutoff"]
    k = int(np.where(yrs == cutoff)[0][0])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    styles = {"world_model": (OI["red"], "-"), "trend_kaya": (OI["green"], "--"),
              "growth_persistence": (OI["blue"], ":"), "persistence": (OI["grey"], ":"),
              "frozen_kaya": (OI["orange"], "-.")}
    ax1.plot(yrs, true / 1000, color=OI["black"], lw=2.4, label="observed", zorder=5)
    for name, (c, ls) in styles.items():
        pred = np.array(bt["aggregate"]["pred_mt"][name]) / 1000
        ax1.plot(yrs[k:], pred[k:], color=c, ls=ls, lw=1.8, label=name.replace("_", " "))
    ax1.axvline(cutoff, color="#bbbbbb", lw=1)
    ax1.set(xlabel="year", ylabel="CO2 (Gt)", title=f"Aggregate forecast (train <= {cutoff})")
    ax1.legend(fontsize=9)

    methods = list(bt["methods"].keys())
    mape = [100 * bt["methods"][m]["mape"] for m in methods]
    order = np.argsort(mape)
    methods = [methods[i] for i in order]; mape = [mape[i] for i in order]
    colors = [OI["red"] if m == "world_model" else OI["grey"] for m in methods]
    ax2.barh([m.replace("_", " ") for m in methods], mape, color=colors)
    ax2.invert_yaxis()
    ax2.set(xlabel="MAPE (%)", title="Per-country holdout error (lower = better)")
    for i, v in enumerate(mape):
        ax2.text(v + 0.1, i, f"{v:.1f}", va="center", fontsize=9)
    return _finish(fig)


def fig_transfer(panel, tr_res) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    wm = tr_res["world_model"]; tk = tr_res["trend_kaya"]
    txt = (f"Held-out countries (n={len(tr_res['test_countries'])})\n"
           f"forecast 2016-2022\n\n"
           f"World model:  MAPE {100*wm['mape']:.1f}%,  R2 {wm['r2']:.3f}\n"
           f"Trend-Kaya:   MAPE {100*tk['mape']:.1f}%,  R2 {tk['r2']:.3f}\n\n"
           f"Adaptation fit (<=2015): MAPE {100*tr_res['adapt_insample_mape']:.1f}%")
    ax.axis("off")
    ax.text(0.03, 0.95, "Spatial transfer to unseen countries", fontsize=13,
            fontweight="bold", va="top")
    ax.text(0.03, 0.78, txt, fontsize=11, va="top", family="monospace")
    ax.text(0.03, 0.17, "Shared dynamics trained on 48 countries;\nonly each held-out "
            "country's latent adapted on its\npre-2015 data.", fontsize=10,
            va="top", color="#555555")
    return _finish(fig)


def fig_planning(panel, plan) -> plt.Figure:
    yrs = np.array(plan["scenario"]["years"])
    bau = np.array(plan["bau"]["global_mt"]) / 1000
    mx = np.array(plan["max_ambition"]["global_mt"]) / 1000
    hist_y = panel.years; hist = panel.co2.sum(0) / 1000
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax1.plot(hist_y, hist, color=OI["black"], lw=2.2, label="observed 1990-2022")
    ax1.plot(yrs, bau, color=OI["red"], lw=2, label="business-as-usual")
    ax1.plot(yrs, mx, color=OI["green"], lw=2, label="max feasible effort")
    ax1.fill_between(yrs, mx, bau, color=OI["green"], alpha=0.08)
    ax1.set(xlabel="year", ylabel="CO2 (Gt/yr)", title="Emissions pathways to 2050 (60 countries)")
    ax1.legend(fontsize=9, loc="upper left")

    fr = plan["frontier"]
    cost = [p["cost"] for p in fr]; cum = [p["cum_gt"] for p in fr]
    ax2.plot(cost, cum, "-o", color=OI["blue"], ms=5)
    ax2.set(xlabel="transition effort (model cost units)",
            ylabel="cumulative 2023-2050 CO2 (Gt)",
            title="Cost-emissions efficient frontier")
    return _finish(fig)


def fig_mix(panel, years, bau_agg, max_agg) -> plt.Figure:
    """bau_agg, max_agg: [Tf, F] energy-weighted aggregate primary-energy shares."""
    yrs = years
    fuels = list(panel.fuels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, sh, title in [(ax1, bau_agg, "Business-as-usual mix"),
                           (ax2, max_agg, "Max-feasible mix")]:
        bottom = np.zeros(len(yrs))
        for fi, f in enumerate(fuels):
            ax.fill_between(yrs, bottom, bottom + sh[:, fi], label=f.replace("_", " "),
                            color=FUEL_COLORS[f], lw=0)
            bottom = bottom + sh[:, fi]
        ax.set(xlabel="year", title=title, ylim=(0, 1))
    ax1.set_ylabel("share of primary energy (energy-weighted)")
    ax2.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- v2 figures
SSP_COLORS = {"SSP1": OI["green"], "SSP2": OI["blue"], "SSP3": OI["red"],
              "SSP4": OI["orange"], "SSP5": OI["purple"]}


def fig_ablation(abl: dict, selected: str) -> plt.Figure:
    """Validation score (country MAPE% + mean |aggregate error|%) for every config."""
    names = sorted(abl, key=lambda n: abl[n]["world_model"]["score"])
    wm = [abl[n]["world_model"]["score"] for n in names]
    any_ = abl[names[0]]
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(names) + 1.6))
    colors = [OI["red"] if n == selected else OI["sky"] for n in names]
    ax.barh(names, wm, color=colors)
    ax.invert_yaxis()
    for ref, c, ls in (("trend_kaya", OI["green"], "--"), ("growth_persistence", OI["grey"], ":")):
        if ref in any_:
            ax.axvline(any_[ref]["score"], color=c, ls=ls, lw=1.6, label=ref.replace("_", " "))
    for i, v in enumerate(wm):
        ax.text(v + 0.15, i, f"{v:.1f}", va="center", fontsize=9)
    ax.set(xlabel="validation score (lower = better)",
           title="Ablation on validation folds (cutoffs 2005/2007/2009)")
    ax.legend(loc="lower right", fontsize=9)
    return _finish(fig)


def fig_test(test: dict, china: dict) -> plt.Figure:
    methods = ["world_model", "trend_kaya", "growth_persistence", "persistence", "frozen_kaya"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(methods))
    w = 0.38
    for j, (drv, c) in enumerate((("realized", OI["blue"]), ("ssp", OI["orange"]))):
        s = test[drv]["summary"]
        vals = [100 * s[m]["country_mape"] for m in methods]
        ax1.bar(x + (j - 0.5) * w, vals, w, color=c,
                label="realised drivers" if drv == "realized" else "SSP2 (2014 vintage) ex-ante")
        for xi, v in zip(x + (j - 0.5) * w, vals):
            ax1.text(xi, v + 0.15, f"{v:.1f}", ha="center", fontsize=8)
    ax1.set_xticks(x, [m.replace("_", "\n") for m in methods], fontsize=9)
    ax1.set(ylabel="per-country MAPE (%)", title="Test folds (cutoffs 2015/2016/2017)")
    ax1.legend(fontsize=9)

    cuts = sorted(china, key=int)
    ax2.bar(np.arange(len(cuts)) - 0.2, [100 * china[c]["co2_err"] for c in cuts], 0.4,
            color=OI["red"], label="world model")
    ax2.bar(np.arange(len(cuts)) + 0.2, [100 * china[c]["trend_kaya_co2_err"] for c in cuts], 0.4,
            color=OI["green"], label="trend-Kaya")
    ax2.axhline(0, color="#444444", lw=0.8)
    ax2.set_xticks(np.arange(len(cuts)), [f"from {c}\n(to {china[c]['year']})" for c in cuts])
    ax2.set(ylabel="China final-year CO2 error (%)", title="The China gap, by forecast origin")
    ax2.legend(fontsize=9)
    return _finish(fig)


def fig_china_forecasts(panel, cutoff: int, series: dict) -> plt.Figure:
    """series: name -> (years, co2) forecasts for China from `cutoff`."""
    i = panel.country_index("China")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(panel.years, panel.co2[i] / 1000, color=OI["black"], lw=2.4, label="observed")
    styles = [OI["red"], OI["grey"], OI["green"], OI["orange"], OI["purple"]]
    for (name, (yrs, v)), c in zip(series.items(), styles):
        ax.plot(yrs, np.asarray(v) / 1000, color=c, lw=2, ls="--" if "orig" in name else "-",
                label=name)
    ax.axvline(cutoff, color="#bbbbbb", lw=1)
    ax.set(xlabel="year", ylabel="CO2 (Gt)", title=f"China: forecasts from the observed {cutoff} state",
           xlim=(2000, int(panel.years[-1]) + 0.5))
    ax.legend(fontsize=9)
    return _finish(fig)


def fig_ssp_drivers(panel, v9, v3, ssp_driver_paths, ssps) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    yrs = np.arange(2015, int(panel.years[-1]) + 1)
    for name, c in (("China", OI["red"]), ("Brazil", OI["green"]), ("United States", OI["blue"])):
        i = panel.country_index(name)
        k = int(np.where(panel.years == 2015)[0][0])
        ax1.plot(yrs, panel.gdp[i, k:] / panel.gdp[i, k], color=c, lw=2.4, label=f"{name} realised")
        g, _ = ssp_driver_paths(panel, v9, "SSP2", 2015, yrs)
        ax1.plot(yrs, g[i] / g[i, 0], color=c, lw=1.6, ls="--", label=f"{name} SSP2 (2014 vintage)")
    ax1.set(xlabel="year", ylabel="GDP index (2015 = 1)",
            title="Ex-ante drivers: 2014-vintage SSPs vs what happened")
    ax1.legend(fontsize=8)
    fut = np.arange(int(panel.years[-1]), 2051)
    i = panel.country_index("China")
    for s in ssps:
        g, _ = ssp_driver_paths(panel, v3, s, int(panel.years[-1]), fut)
        ax2.plot(fut, g[i] / g[i, 0], color=SSP_COLORS[s], lw=2, label=s)
    ax2.set(xlabel="year", ylabel="China GDP index (2022 = 1)",
            title="Forward drivers: SSP database v3.2 (2025)")
    ax2.legend(fontsize=9)
    return _finish(fig)


def fig_ssp_pathways(panel, plan: dict) -> plt.Figure:
    yrs = np.array(plan["years"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    ax1.plot(panel.years, panel.co2.sum(0) / 1000, color=OI["black"], lw=2.2, label="observed")
    for name, d in plan["scenarios"].items():
        s = name.split()[0]
        if s not in SSP_COLORS:
            continue
        ax1.plot(yrs, np.array(d["bau_global_mt"]) / 1000, color=SSP_COLORS[s], lw=2, label=f"{s} BAU")
        ax1.plot(yrs, np.array(d["max_global_mt"]) / 1000, color=SSP_COLORS[s], lw=1.4, ls="--")
        if "open_loop_max_global_mt" in d:
            ax1.plot(yrs, np.array(d["open_loop_max_global_mt"]) / 1000, color=SSP_COLORS[s],
                     lw=1.2, ls=":")
    ax1.set(xlabel="year", ylabel="CO2 (Gt/yr)",
            title="BAU (solid); max feasible effort with fitted\ndynamics (dashed) "
                  "and with no learned reaction (dotted)")
    ax1.legend(fontsize=8, ncol=2)

    rb = plan["robust"]
    ssps = list(rb["robust_cum_by_ssp"])
    x = np.arange(len(ssps))
    ax2.bar(x - 0.2, [rb["tailored_cum_by_ssp"][s] for s in ssps], 0.4, color=OI["grey"],
            label=f"plan tuned to SSP2 (effort {rb['tailored_cost']:.2f})")
    ax2.bar(x + 0.2, [rb["robust_cum_by_ssp"][s] for s in ssps], 0.4, color=OI["blue"],
            label=f"robust plan (effort {rb['robust_cost']:.2f})")
    ax2.axhline(rb["budget_gt"], color=OI["red"], lw=1.6, ls="--", label="carbon budget")
    lo = min(min(rb["tailored_cum_by_ssp"].values()), min(rb["robust_cum_by_ssp"].values()))
    ax2.set_ylim(lo * 0.9, None)
    ax2.set_xticks(x, ssps)
    ax2.set(ylabel="cumulative 2023-2050 CO2 (Gt)",
            title=f"Robust planning: +{rb['price_of_robustness_pct']:.0f}% effort buys "
                  "compliance in every SSP")
    ax2.legend(fontsize=8)
    return _finish(fig)


def fig_structural(panel, st: dict) -> plt.Figure:
    """Planning results across the validation-tied models (scripts/07_structural.py)."""
    yrs = np.array(st["years"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    ax1.plot(panel.years, panel.co2.sum(0) / 1000, color=OI["black"], lw=2.2, label="observed")
    groups = [("bau_global_mt", OI["grey"], "business-as-usual"),
              ("max_global_mt", OI["red"], "max effort, fitted dynamics"),
              ("open_loop_global_mt", OI["blue"], "max effort, no learned reaction")]
    for key, col, lab in groups:
        Y = np.array([m[key] for m in st["models"]]) / 1000
        ax1.fill_between(yrs, Y.min(0), Y.max(0), color=col, alpha=0.18, lw=0)
        for i, y in enumerate(Y):
            ax1.plot(yrs, y, color=col, lw=0.9, alpha=0.8, label=lab if i == 0 else None)
    ax1.set(xlabel="year", ylabel="CO2, 60 countries (Gt/yr)",
            title=f"{st['ssp']}: {len(st['models'])} models tied on validation")
    ax1.legend(fontsize=8, loc="upper left")

    s = st["summary"]
    names = [f"{m['config']}\nseed {m['seed']}" for m in st["models"]]
    y = np.arange(len(names))
    for key, col, mk, lab in [("global_cum_bau_gt", OI["grey"], "o", "business-as-usual"),
                              ("global_cum_max_gt", OI["red"], "s", "max effort, fitted"),
                              ("global_cum_open_loop_gt", OI["blue"], "D", "max effort, open loop")]:
        ax2.scatter([m[key] for m in st["models"]], y, color=col, marker=mk, s=38, label=lab, zorder=3)
    for b, lab in [(s["budget_1p5_gt"], "1.5 °C budget, 50% (all CO2 to net zero)"), (s["budget_2p0_gt"], "2 °C budget, 50% (all CO2 to net zero)")]:
        ax2.axvline(b, color=OI["black"], lw=1.1, ls="--")
        ax2.text(b, len(names) - 0.4, f" {lab.split(',')[0]} (50%)", fontsize=8, va="bottom")
    ax2.set_yticks(y, names, fontsize=8)
    ax2.set_ylim(-0.6, len(names) + 0.4)
    ax2.set(xlabel="cumulative 2023-2050 CO2, scaled to global (Gt)",
            title="Carbon budgets vs. every model")
    ax2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    return _finish(fig)


def fig_longrun(lr: dict) -> plt.Figure:
    """Long-run level checks (scripts/08_longrun.py): long-horizon cumulative-CO2 errors,
    and energy-ladder decoupling relative to each country's own past."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.1, 1]})
    bt = lr["backtests"]
    any_cfg = next(iter(bt["longval"]))
    rows = [(c, "world_model", c + (" (selected)" if c == lr.get("selected") else ""))
            for c in bt["longval"]] + \
           [(any_cfg, m, f"baseline: {m.replace('_', ' ')}")
            for m in ("trend_kaya", "growth_persistence", "persistence")]
    y = np.arange(len(rows))
    from ..eval.rolling import PHASES
    hmin, hmax = 99, 0
    for off, phase, col in [(-0.2, "longval", OI["blue"]), (0.2, "longtest", OI["orange"])]:
        cuts, hor = PHASES[phase]
        hmin, hmax = min(hmin, *(hor(c) for c in cuts)), max(hmax, *(hor(c) for c in cuts))
        lab = f"cutoffs {cuts[0]}-{cuts[-1]} → {cuts[0] + hor(cuts[0])}"
        v = np.array([100 * bt[phase][c][m]["cum_err"] for c, m, _ in rows])
        per = [[100 * x[m] for x in bt[phase][c]["cum_err_by_cutoff"].values()] for c, m, _ in rows]
        lo, hi = v - np.array([min(q) for q in per]), np.array([max(q) for q in per]) - v
        ax1.barh(y + off, v, 0.38, color=col, label=lab, xerr=[lo, hi],
                 error_kw={"ecolor": OI["black"], "elinewidth": 0.8, "capsize": 2})
    ax1.axvline(0, color=OI["black"], lw=0.8)
    ax1.set_yticks(y, [r[2] for r in rows], fontsize=8)
    ax1.invert_yaxis()
    ax1.set(xlabel="error in cumulative aggregate CO2 over the horizon (%)\n"
                   "bar: mean over cutoffs; whisker: min-max over cutoffs",
            title=f"Long-horizon backtests ({hmin}-{hmax} years)")
    ax1.legend(fontsize=8, loc="lower right")

    lad = lr["ladder"]
    models = [k for k, v in lad.items() if v["family"] in ("stab", "unconstrained")]
    cols = {"stab": OI["red"], "unconstrained": OI["grey"]}
    seen = set()
    for i, k in enumerate(models):
        fam = lad[k]["family"]
        ax2.scatter(100 * lad[k]["energy_weighted_excess_per_year"], i, color=cols[fam], s=40,
                    zorder=3, label=None if fam in seen else
                    ("stabilised" if fam == "stab" else "unconstrained"))
        seen.add(fam)
    ax2.axvline(0, color=OI["green"], lw=1.4, label="each country repeats its 1994-2022 drift")
    for k, ls, txt in [("ref_own_trend_intensity", ":", "own 10-yr intensity trend continued"),
                       ("ref_constant_intensity", "--", "constant energy intensity")]:
        ax2.axvline(100 * lad[k]["energy_weighted_excess_per_year"], color=OI["black"], lw=1.1,
                    ls=ls, label=txt)

    def label(k):
        if k == "selected":
            return f"{lad[k]['config']} (selected)"
        cfg, seed = k.rsplit("_s", 1)
        return f"{cfg}, seed {seed}"

    ax2.set_yticks(np.arange(len(models)), [label(m) for m in models], fontsize=8)
    ax2.invert_yaxis()
    ax2.set(xlabel="decoupling slower (+) / faster (−) than own past, %/yr (energy-weighted)",
            title="Energy ladder: SSP2 BAU 2022-2050")
    ax2.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=2)
    return _finish(fig)


def fig_backtest_error(pe: dict) -> plt.Figure:
    """Planning ranges with and without long-horizon backtest error
    (scripts/09_backtest_error.py), against the 1.5 C and 2 C budgets."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    labels = {"bau": "business-as-usual", "max_fitted": "max effort, fitted dynamics",
              "max_open_loop": "max effort, no learned reaction"}
    S = pe["summary"]["expanded"]
    rows = []
    for kind in ("bau", "max_fitted", "max_open_loop"):
        rows += [(f"{labels[kind]}\nstabilised tied models, SSP2", S["none"][kind]["global_cum_gt"],
                  S["linear"][kind]["global_cum_gt"]),
                 (f"{labels[kind]}\nselected model, SSP1-5", pe["selected_ssp1to5"][kind]["without_error"],
                  pe["selected_ssp1to5"][kind]["with_error"])]
    for i, (_, base, err) in enumerate(rows):
        ax.plot(err["envelope"], [i, i], color=OI["sky"], lw=9, solid_capstyle="butt", alpha=0.6,
                label="+ backtest error: all cases" if i == 0 else None)
        ax.plot(err["q10_q90"], [i, i], color=OI["blue"], lw=9, solid_capstyle="butt",
                label="+ backtest error: 10th-90th percentile" if i == 0 else None)
        ax.plot(base["envelope"], [i, i], color=OI["black"], lw=2, marker="|", ms=14,
                label="model as fitted" if i == 0 else None)
    for b, txt, ls in [(pe["budget_1p5_gt"], "1.5 °C budget, 50% (all CO2 to net zero)", "--"), (pe["budget_2p0_gt"], "2 °C budget, 50% (all CO2 to net zero)", ":")]:
        ax.axvline(b, color=OI["red"], lw=1.4, ls=ls, label=txt)
    ax.set_yticks(np.arange(len(rows)), [r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    n = pe["delta_stats"][pe["selected"]]["n"]
    ax.set(xlabel="cumulative 2023-2050 CO2, scaled to global (Gt)",
           title=f"Planning ranges with long-horizon backtest error ({n} backtests per configuration)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    return _finish(fig)


def fig_driver_error(pe: dict, de: dict) -> plt.Figure:
    """How each error source widens the planning ranges: model as fitted, + emission backtest
    error (scripts/09), + GDP/population driver error (scripts/10), + both."""
    kinds = [("max_fitted", "max effort, fitted dynamics"),
             ("max_open_loop", "max effort, no learned reaction"),
             ("bau", "business-as-usual")]
    srcs = [("as fitted", de["summary"]["driver:none"]),
            ("+ emission backtest error", pe["summary"]["expanded"]["linear"]),
            ("+ driver error (naive projections, 1997-2007)", de["summary"]["driver:naive_long"]),
            ("+ driver error (2014 SSP projections)", de["summary"]["driver:ssp_v9"]),
            ("+ both, paired by cutoff", de["summary"]["both:paired"]),
            ("+ both, independent", de["summary"]["both:independent"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, (kind, title) in zip(axes, kinds):
        for i, (lab, s) in enumerate(srcs):
            r = s[kind]["global_cum_gt"] if "global_cum_gt" in s[kind] else s[kind]
            ax.plot(r["envelope"], [i, i], color=OI["sky"], lw=8, solid_capstyle="butt", alpha=0.6,
                    label="all cases" if i == 0 else None)
            ax.plot(r["q10_q90"], [i, i], color=OI["blue"], lw=8, solid_capstyle="butt",
                    label="10th-90th percentile" if i == 0 else None)
        for b, txt, ls in [(de["budget_1p5_gt"], "1.5 °C budget, 50% (all CO2 to net zero)", "--"),
                           (de["budget_2p0_gt"], "2 °C budget, 50% (all CO2 to net zero)", ":")]:
            ax.axvline(b, color=OI["red"], lw=1.4, ls=ls, label=txt)
        ax.set(title=title, xlabel="cumulative 2023-2050 CO2, global (Gt)")
    axes[0].set_yticks(np.arange(len(srcs)), [s[0] for s in srcs], fontsize=8)
    axes[0].invert_yaxis()
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4)
    fig.suptitle("Planning ranges by error source: stabilised validation-tied models, SSP2",
                 fontweight="bold")
    return _finish(fig)


def fig_budget_likelihoods(bl: dict) -> plt.Figure:
    """Likelihood of staying below 1.5 / 1.7 / 2 C for every planning case
    (scripts/11_budgets.py): headline accounting and both error sources (paired)."""
    kinds = [("max_open_loop", "max effort, no learned reaction"),
             ("max_fitted", "max effort, fitted dynamics"), ("bau", "business-as-usual")]
    targets = (1.5, 1.7, 2.0)
    cols = {1.5: OI["red"], 1.7: OI["orange"], 2.0: OI["blue"]}
    R = bl["results"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    y, labels = 0, []
    for kind, klab in kinds:
        for t in targets:
            h = R[f"both_paired|{kind}|all|headline|+0|{t}"]
            lo, hi = (100 * v for v in h["q10_q90"])
            ax.plot([lo, hi], [y, y], color=cols[t], lw=7, alpha=0.75, solid_capstyle="butt",
                    label=f"{t} °C: 10th-90th percentile of cases" if (kind, t) in
                    (("max_open_loop", 1.7), ("max_open_loop", 2.0)) else None)
            if h["share_at_or_below_17"] == 1.0:
                ax.text(19, y, "17% or less in every case", fontsize=7.5, va="center",
                        color=OI["grey"])
            ax.plot(100 * h["median"], y, "o", color=OI["black"], ms=5,
                    label="median (headline accounting)" if (kind, t) == ("max_open_loop", 1.5) else None)
            le = R[f"both_paired|{kind}|all|lenient|+0|{t}"]
            ax.plot(100 * le["median"], y, "o", mfc="none", mec=OI["black"], ms=8,
                    label="median, fossil 2023-2050 only (lenient)" if (kind, t) == ("max_open_loop", 1.5) else None)
            for sh, mk in (("-200", "<"), ("+200", ">")):
                v = R[f"both_paired|{kind}|all|headline|{sh}|{t}"]["median"]
                ax.plot(100 * v, y, mk, color=OI["grey"], ms=6,
                        label=f"median, non-CO2 {sh} Gt" if (kind, t) == ("max_open_loop", 1.5) else None)
            labels.append(f"{klab}: {t} °C")
            y += 1
        y += 0.6
    ticks = [i + 0.6 * (i // 3) for i in range(len(labels))]
    ax.set_yticks(ticks, labels, fontsize=8)
    ax.set_ylim(ticks[-1] + 1.1, -0.6)
    for x, txt in ((50, "as likely as not"), (66, "likely")):
        ax.axvline(x, color=OI["grey"], lw=1, ls=":")
        ax.text(x + 0.5, ticks[-1] + 0.95, txt, fontsize=7.5, color=OI["grey"], va="bottom")
    ax.set_xlim(10, 90)
    ax.set(xlabel="likelihood of staying below the threshold (%), published budgets (Forster et al. 2025)\n"
                  "values at 17% mean 17% or less; at 83% mean 83% or more",
           title="Carbon-budget likelihoods: model + driver error (paired), all CO2 to net zero in 2070")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    return _finish(fig)


def fig_to_2100(t: dict) -> plt.Figure:
    """Model runs to 2100 (scripts/12_to_2100.py) against the assumed post-2050 tail, and
    the 2 C likelihoods each implies."""
    yrs = np.array(t["years"])
    k50 = int(np.where(yrs == 2050)[0][0])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    cols = {"bau": OI["grey"], "max_fitted": OI["red"], "max_open_loop": OI["blue"]}
    labs = {"bau": "business-as-usual", "max_fitted": "max effort, fitted dynamics",
            "max_open_loop": "max effort, no learned reaction"}
    for i, (name, paths) in enumerate(t["fitted_paths"].items()):
        for kind, p in paths.items():
            g = np.array(p["global_mt"]) / 1000 / 0.926
            ax1.plot(yrs, g, color=cols[kind], lw=1.4, alpha=0.85, label=labs[kind] if i == 0 else None)
            if kind == "max_fitted":
                tail = g[k50] * np.clip((2070 - yrs[k50:]) / 20, 0, 1)
                ax1.plot(yrs[k50:], tail, color=cols[kind], lw=1.1, ls="--",
                         label="assumed tail: linear to net zero in 2070" if i == 0 else None)
    ax1.axhline(0, color=OI["black"], lw=0.8)
    ax1.axvline(2050, color=OI["grey"], lw=0.8, ls=":")
    ax1.text(2051, ax1.get_ylim()[1] * 0.95, "beyond 2050: extrapolation", fontsize=8, color=OI["grey"], va="top")
    ax1.set(xlabel="year", ylabel="fossil + industry CO2, global (Gt/yr)",
            title="Stabilised tied models as fitted, SSP2, run to 2100")
    ax1.legend(fontsize=8, loc="upper right")

    S = t["summary"]
    rows = [("all tied models, fitted dynamics", "paired|model|all|max_fitted"),
            ("all tied models, no learned reaction", "paired|model|all|max_open_loop"),
            ("selected model, fitted dynamics", "paired|model|selected|max_fitted"),
            ("selected model, no learned reaction", "paired|model|selected|max_open_loop")]
    y = np.arange(len(rows))
    for off, acc, col, lab in [(-0.18, "total_tail_2070", OI["orange"], "post-2050 tail assumed (net zero 2070)"),
                               (0.18, "total_model_to_2100", OI["purple"], "model run to 2100 (upper bound)")]:
        med = [100 * S[k][f"{acc}|2.0"]["median"] for _, k in rows]
        lo = [100 * S[k][f"{acc}|2.0"]["q10_q90"][0] for _, k in rows]
        hi = [100 * S[k][f"{acc}|2.0"]["q10_q90"][1] for _, k in rows]
        ax2.barh(y + off, med, 0.34, color=col, label=lab,
                 xerr=[np.array(med) - lo, np.array(hi) - med],
                 error_kw={"ecolor": OI["black"], "elinewidth": 0.8, "capsize": 2})
    ax2.axvline(50, color=OI["grey"], lw=1, ls=":")
    ax2.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 90)
    ax2.set(xlabel="likelihood of staying below 2 °C (%), max effort, model + driver error\n"
                   "bar: median; whisker: 10th-90th percentile of cases",
            title="2 °C at maximum effort: assumed tail vs model")
    ax2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.22), ncol=2)
    return _finish(fig)
