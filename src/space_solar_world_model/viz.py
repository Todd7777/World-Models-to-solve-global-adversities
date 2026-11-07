"""Figures for the Space Solar World Model (static PNGs, house style)."""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GEO_KM = 35786.0


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.fontsize": 8, "legend.labelcolor": INK2, "font.family": "sans-serif",
        "lines.linewidth": 1.8, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def _no_minor_labels(ax):
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def _altitude_axis(ax):
    ax.set_xscale("log")
    _no_minor_labels(ax)
    ax.set_xlim(300, 42000)
    ax.set_xticks([500, 1000, 2000, 5000, 10000, 20000, GEO_KM])
    ax.set_xticklabels(["500", "1k", "2k", "5k", "10k", "20k", "GEO"])
    ax.set_xlabel("Altitude [km]")


def _end_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center",
                fontsize=7.5, color=INK2)


def fig_environment(tab: dict, path: str):
    alts = np.array(tab["alts"])
    dd = np.array(tab["dd_alts"])
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.2))
    fig.suptitle("The space environment by altitude", x=0.01, ha="left", fontsize=11.5,
                 fontweight="bold", color=INK)
    ax = axs[0, 0]
    for c, key, lab in ((CAT[0], "0", "equatorial"), (CAT[1], "53", "53 deg"), (CAT[2], "90", "polar")):
        y = [d["sunlit_fraction"] for d in tab["illumination"][key]]
        ax.plot(alts, y, color=c, label=lab)
    y = [d["sunlit_fraction"] for d in tab["illumination"]["dawn-dusk"]]
    ax.plot(dd, y, color=CAT[3], label="dawn-dusk sun-synchronous")
    _altitude_axis(ax)
    ax.set_ylabel("Fraction of time in sunlight")
    ax.set_title("Time in sunlight")
    ax.legend(loc="lower right")
    ax = axs[0, 1]
    for c, key, lab in ((CAT[0], "0", "equatorial"), (CAT[1], "53", "53 deg"), (CAT[2], "90", "polar")):
        ax.plot(alts, [d["total"] for d in tab["radiation"][key]], color=c, label=lab)
    an = tab["anchors"]
    ax.scatter([a["alt_km"] for a in an], [a["target"] for a in an], s=22, color=INK, zorder=5,
               edgecolor=SURFACE, linewidth=1.2, label="calibration anchors")
    ax.set_yscale("log")
    _altitude_axis(ax)
    ax.set_ylabel("Damage-equivalent fluence\n[1-MeV e / cm^2 / yr]")
    ax.set_title("Radiation damage to solar cells")
    ax.legend(loc="lower right")
    ax = axs[1, 0]
    drag = np.array(tab["drag_coeff"]) * 0.5 * 0.5
    m = drag > 0
    ax.plot(alts[m], np.maximum(drag[m], 1e-3), color=CAT[0])
    ax.set_yscale("log")
    _altitude_axis(ax)
    ax.set_xlim(300, 3000)
    ax.set_xticks([300, 500, 1000, 2000, 3000])
    ax.set_xticklabels(["300", "500", "1k", "2k", "3k"])
    ax.set_ylabel("Station-keeping delta-v [m/s per year]")
    ax.set_title("Drag on a light array (0.5 m^2/kg)")
    ax = axs[1, 1]
    ax.plot(alts, tab["debris"]["1cm"], color=CAT[0], label="> 1 cm")
    ax.plot(alts, tab["debris"]["10cm"], color=CAT[1], label="> 10 cm (destroys a module)")
    ax.set_yscale("log")
    _altitude_axis(ax)
    ax.set_ylabel("Impacts per m^2 per year")
    ax.set_title("Debris impacts")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_in_space(tab: dict, path: str):
    alts = np.array(tab["alts"])
    incs = tab["incs"]
    dd = np.array(tab["dd_alts"])
    scen = ["today", "starship", "advanced"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.4), sharey=False)
    for ax, s in zip(axs, scen):
        g = tab["grid"][s]
        for c, i in zip((CAT[0], CAT[1], CAT[2]), (0, 2, 4)):
            ax.plot(alts, g["firm"][i], color=c, label=f"{incs[i]:g} deg inclination")
        ax.plot(dd, g["dd_firm"], color=CAT[3], label="dawn-dusk sun-synchronous")
        rows = {r["placement"]: r for r in tab["tables"][s]}
        for k, lab in (("l1", "Sun-Earth L1"), ("lunar_nrho", "Lunar NRHO")):
            if k in rows:
                ax.axhline(rows[k]["lcoe_firm"], color=MUTED, linewidth=0.9)
                ax.annotate(lab, (330, rows[k]["lcoe_firm"]), xytext=(0, 3), textcoords="offset points",
                            fontsize=7.5, color=INK2)
        b = tab["best_dawn_dusk"][s]
        ax.scatter([b["alt_km"]], [b["lcoe_firm"]], s=40, color=CAT[3], edgecolor=SURFACE, linewidth=1.5, zorder=6)
        ax.annotate(f"cheapest dawn-dusk: {b['alt_km']:,.0f} km, ${b['lcoe_firm']:,.0f}/MWh",
                    (b["alt_km"], b["lcoe_firm"]), xytext=(8, 10), textcoords="offset points",
                    fontsize=7.5, color=INK)
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo / 1.6, hi * 1.15)
        _altitude_axis(ax)
        ax.set_title(f"{s.capitalize()} scenario")
        ax.set_ylabel("Cost of firm in-orbit power [$/MWh]")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Power used in orbit: cost of firm electricity by altitude and orbit type", x=0.01, ha="left",
                 fontsize=11.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


NICE = {"geo": "GEO", "meo_20200_55": "MEO 20,200 km", "meo_12000_55": "MEO 12,000 km",
        "meo_8000_0": "MEO 8,000 km (eq.)", "leo_1500_53": "LEO 1,500 km", "leo_1000_53": "LEO 1,000 km",
        "sso18_1000": "Dawn-dusk 1,000 km", "tundra": "Tundra (HEO)", "molniya": "Molniya (HEO)",
        "l1": "Sun-Earth L1"}


def fig_beam_ranking(tab: dict, path: str, site: str = "Las Vegas"):
    scen = ["today", "starship", "advanced"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, s in zip(axs, scen):
        rows = tab[s]["rankings"][site]
        best = {}
        for r in rows:
            if r["placement"] not in best or r["lcoe"] < best[r["placement"]]["lcoe"]:
                best[r["placement"]] = r
        items = sorted(best.values(), key=lambda r: r["lcoe"])[::-1]
        y = np.arange(len(items))
        ax.barh(y, [r["lcoe"] for r in items], height=0.55, color=CAT[0])
        ax.set_yticks(y)
        ax.set_yticklabels([f"{NICE.get(r['placement'], r['placement'])} ({r['tech']})" for r in items], fontsize=7.5)
        g = tab[s]["ground_firm"][site]["lcoe"]
        ax.axvline(g, color=CAT[1], linewidth=1.8)
        ax.annotate(f"firm ground solar\n+ batteries ${g:,.0f}", (g, len(items) - 0.6), xytext=(4, 0),
                    textcoords="offset points", fontsize=7.5, color=INK2, va="top")
        for yi, r in zip(y, items):
            ax.annotate(f"${r['lcoe']:,.0f}", (r["lcoe"], yi), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=7, color=INK2)
        ax.set_xscale("log")
        _no_minor_labels(ax)
        ax.set_xlabel("Levelised cost of delivered electricity [$/MWh]")
        ax.set_title(f"{s.capitalize()}")
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Power beamed to {site}: every placement, design optimised", x=0.01, ha="left",
                 fontsize=11.5, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_profiles(tab: dict, path: str):
    sh = tab["hourly_shapes_las_vegas"]
    names = [("geo", "GEO", CAT[0]), ("meo_20200_55", "MEO 20,200 km network", CAT[1]),
             ("leo_1000_53", "LEO 1,000 km network", CAT[2]), ("ground_pv", "ground solar (tracking)", CAT[3])]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, j, title in zip(axs, range(3), ("March equinox", "June solstice", "December solstice")):
        for k, lab, c in names:
            ax.plot(np.arange(24) + 0.5, sh[k][j], color=c, label=lab)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xlabel("Local solar time [h]")
        ax.set_title(title)
    axs[0].set_ylabel("Delivered power / annual mean")
    axs[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("When does the power arrive in Las Vegas?", x=0.01, ha="left", fontsize=11.5,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_sweeps(tab: dict, path: str):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, s in zip(axs[:2], ("starship", "advanced")):
        rows = tab["launch_sweep"][s]
        for c, (k, lab) in zip(CAT, (("geo", "GEO"), ("meo_20200_55", "MEO 20,200 km"), ("leo_1000_53", "LEO 1,000 km"))):
            pts = sorted((r["value"], r["lcoe"]) for r in rows if r["placement"] == k)
            ax.plot(*zip(*pts), color=c, marker="o", markersize=4, label=lab)
        for site, ls in (("Las Vegas", "-"), ("London", "-")):
            g = tab["breakeven_launch"][f"{s}|{site}"]["ground_firm_lcoe"]
            ax.axhline(g, color=MUTED, linewidth=0.9)
            ax.annotate(f"firm ground solar, {site} ${g:,.0f}", (11, g), xytext=(0, 3),
                        textcoords="offset points", fontsize=7.5, color=INK2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _no_minor_labels(ax)
        ax.set_xlabel("Launch price to LEO [$/kg]")
        ax.set_ylabel("Delivered electricity [$/MWh]")
        ax.set_title(f"do(launch price), {s} technology")
        ax.legend(loc="lower right", fontsize=7.5)
    ax = axs[2]
    lat = tab["latitude_sweep"]
    for c, (k, lab) in zip(CAT, (("geo", "GEO"), ("tundra", "Tundra"), ("molniya", "Molniya"))):
        pts = sorted((r["lat"], r["lcoe"]) for r in lat if r["placement"] == k and np.isfinite(r["lcoe"]))
        if pts:
            ax.plot(*zip(*pts), color=c, marker="o", markersize=4, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("Receiving-site latitude [deg N]")
    ax.set_ylabel("Delivered electricity [$/MWh]")
    ax.set_title("do(site latitude): GEO vs high-elliptical orbits")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_mechanisms(tab: dict, path: str):
    mech = tab["mechanisms"]
    fig, axs = plt.subplots(1, len(mech), figsize=(13, 3.8), sharey=True)
    for ax, (title, rows) in zip(np.atleast_1d(axs), mech.items()):
        rows = [r for r in rows if r["intervention"] != "baseline"]
        y = np.arange(len(rows))[::-1]
        v = [100 * r["change"] for r in rows]
        ax.barh(y, v, height=0.55, color=[CAT[0] if x < 0 else CAT[1] for x in v])
        ax.set_yticks(y)
        ax.set_yticklabels([r["intervention"] for r in rows], fontsize=8)
        for yi, x in zip(y, v):
            ax.annotate(f"{x:+.0f}%", (x, yi), xytext=(-3 if x < 0 else 3, 0), textcoords="offset points",
                        ha="right" if x < 0 else "left", va="center", fontsize=7.5, color=INK2)
        ax.axvline(0, color=AXIS, linewidth=0.8)
        ax.set_xlabel("Change in levelised cost")
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", visible=False)
        lo = min(v + [0])
        ax.set_xlim(lo * 1.25 - 5, max(v + [0]) * 1.3 + 8)
    fig.suptitle("What drives the cost? Knock out one mechanism at a time (do-interventions)",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_counterfactual(tab: dict, path: str):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.0))
    for ax, key, title in ((axs[0], "counterfactual_geo_pilot", "GEO pilot under-performs: would MEO/LEO have done better?"),
                           (axs[1], "counterfactual_orbital_compute", "650 km data-centre arrays degrade: was another orbit better?")):
        cf = tab[key]
        names = ["factual"] + [k for k in cf if isinstance(cf[k], dict) and "posterior" in cf[k] and k != "factual"]
        y = np.arange(len(names))[::-1]
        for yi, n in zip(y, names):
            for off, which, c, lab in ((0.12, "prior", MUTED, "before evidence (prior)"),
                                       (-0.12, "posterior", CAT[0], "after evidence (counterfactual)")):
                s = cf[n][which]
                ax.plot([s["p05"], s["p95"]], [yi + off] * 2, color=c, linewidth=1.8)
                ax.scatter([s["median"]], [yi + off], s=36, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=5,
                           label=lab if yi == y[0] else None)
            if n != "factual":
                ax.annotate(f"P(better than factual) = {cf[n]['p_better_than_factual']:.2f}",
                            (cf[n]["posterior"]["p95"], yi - 0.12), xytext=(5, 0), textcoords="offset points",
                            va="center", fontsize=7.5, color=INK2)
        labels = []
        for n in names:
            if n == "factual":
                labels.append("factual choice" + (" (GEO)" if "geo" in key else " (dawn-dusk 650 km)"))
            else:
                labels.append(NICE.get(n, n).replace("sso18_", "dawn-dusk ").replace("l1", "Sun-Earth L1"))
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xscale("log")
        _no_minor_labels(ax)
        ax.set_xlabel("Levelised cost [$/MWh], median and 90% interval")
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_all(tables_dir: str, figures_dir: str):
    style()
    os.makedirs(figures_dir, exist_ok=True)
    ld = lambda n: json.load(open(os.path.join(tables_dir, n)))
    made = []
    if os.path.exists(os.path.join(tables_dir, "physics_maps.json")):
        fig_environment(ld("physics_maps.json"), os.path.join(figures_dir, "ss1_environment.png")); made.append("ss1")
    if os.path.exists(os.path.join(tables_dir, "in_space.json")):
        fig_in_space(ld("in_space.json"), os.path.join(figures_dir, "ss2_in_space.png")); made.append("ss2")
    if os.path.exists(os.path.join(tables_dir, "beam.json")):
        b = ld("beam.json")
        fig_beam_ranking(b, os.path.join(figures_dir, "ss3_beam_ranking.png")); made.append("ss3")
        fig_profiles(b, os.path.join(figures_dir, "ss4_delivery_profiles.png")); made.append("ss4")
    if os.path.exists(os.path.join(tables_dir, "counterfactuals.json")):
        c = ld("counterfactuals.json")
        fig_sweeps(c, os.path.join(figures_dir, "ss5_counterfactual_sweeps.png")); made.append("ss5")
        fig_mechanisms(c, os.path.join(figures_dir, "ss6_mechanisms.png")); made.append("ss6")
        fig_counterfactual(c, os.path.join(figures_dir, "ss7_rung3_counterfactuals.png")); made.append("ss7")
    return made
