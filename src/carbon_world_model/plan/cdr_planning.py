"""Removals as a planning lever: the optimiser chooses novel CDR alongside abatement.

Until 2026-11-14 negative emissions were only subtracted after the model had run
(climate/cdr.py, scripts/12_to_2100.py). Here the planner chooses a global removal path
r[t] (GtCO2/yr) together with the abatement controls (clean push g, efficiency h, process
abatement p; plan/optimize.py), so plans can trade removals against abatement.

The two levers are measured in different currencies. Abatement effort is the model's
quadratic effort index (optimize._cost); it has no dollar calibration. Removals cost
dollars (AR6 cost ranges). No exchange rate between them is known, so none is assumed.
Instead the planner minimises abatement effort for a given removal spending allowance S,
which traces the frontier between the two. Any exchange rate picks one point on it.

Rules fixed before the first run (2026-11-14):
  * Supply: the four novel, durable methods of AR6 WGIII Ch12 Table 12.6 (climate/cdr.py
    CDR_METHODS), each method's potential (range midpoint) spread evenly over its cost
    range in N_BINS price bins. The biomass land cap (central 2.4 GtCO2/yr) goes to the
    CHEAPEST biomass bins first. There is no price cap: the price is what removals cost.
  * Scale-up: in each year the capacity of every bin is its full-scale capacity times
    the logistic factor of climate/cdr.novel_cdr_path (0.002 GtCO2/yr in 2025, 40%/yr,
    saturating at the full potential), as in the accounting runs.
  * Energy: DACCS capacity in each year is capped at f x the plan's OWN low-carbon
    primary energy / 7 EJ per GtCO2 (f = 10%; cdr.energy_limited_cdr_path). Low-carbon
    energy comes from the rollout, so a stronger clean push also unlocks DACCS.
  * Cost: removals fill the cheapest bins first; spending = sum of price x volume,
    undiscounted, 2023-2100, in USD trillion.
  * Budget: PEAK cumulative net CO2 2023-2100 (global fossil+industry scaled from the
    panel by coverage, plus land-use CO2 falling linearly from 4.2 Gt/yr to zero in 2050,
    minus removals) must not exceed the budget, AND net CO2 must be <= 0 in 2100, so the
    peak is reached within the run (published budgets count CO2 until net zero).
  * Objective: effort / effort at maximum feasible effort (same model, same dynamics)
    + penalties on budget overshoot, on positive net CO2 in 2100 and on spending above
    S, + a tie-break of 1e-4 per USD trillion so money is not spent for nothing.
  * A plan MEETS the budget if its peak is within 1% of it and net CO2 in 2100 is at
    most 0.1 GtCO2/yr. Otherwise the run reports how far it misses (penalty solutions are
    approximate minimum overshoots, not certified).
  * Changed after ONE trial run (selected model, 2C 50% budget), before any grid ran: the
    soft spending cap leaked (an allowance of zero still spent USD 7 trillion, traded
    against budget overshoot). Now an allowance of zero switches removals off, and
    spending above a positive allowance is penalised per percent of the allowance, ten
    times as hard as the same percent of budget overshoot, so the allowance acts as the
    hard limit and the budget as the soft one.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..climate import budgets as B
from ..climate import cdr as CDR
from ..config import FUELS, LOW_CARBON_FUELS
from . import optimize as O
from .scenario import Scenario

N_BINS = 40
TWH_PER_EJ = 277.78
COVERAGE = 0.926                       # panel share of global CO2 (climate/tcre.scale_to_global)
LC_IDX = [FUELS.index(f) for f in LOW_CARBON_FUELS]
MEET_TOL = 0.01                        # peak may exceed the budget by 1%
NET_END_TOL = 0.1                      # GtCO2/yr in 2100
SPEND_TIEBREAK = 1e-4                  # objective units per USD trillion


def supply_bins(land_cap: float = CDR.LAND_CAPS["central"], n_bins: int = N_BINS):
    """Full-scale supply curve as price bins sorted by price.

    Returns (price USD/t [n], capacity GtCO2/yr [n], is_daccs [n]). Biomass methods
    (BECCS, biochar) share the land cap, which is given to their cheapest bins first."""
    rows = []
    for name, (clo, chi, plo, phi, bio) in CDR.CDR_METHODS.items():
        edges = np.linspace(clo, chi, n_bins + 1)
        mids = 0.5 * (edges[1:] + edges[:-1])
        cap = 0.5 * (plo + phi) / n_bins
        rows += [(float(p), cap, bool(bio), name == "daccs") for p in mids]
    rows.sort(key=lambda r: r[0])
    left, out = land_cap, []
    for p, cap, bio, dac in rows:
        if bio:
            cap, left = min(cap, max(left, 0.0)), left - cap
        out.append((p, cap, dac))
    price, cap, dac = (np.array(x) for x in zip(*out))
    return price, cap, dac.astype(bool)


def capacity(years, lowcarbon_ej, bins, frac: float = CDR.ENERGY_FRACTIONS["central"],
             ej_per_gt: float = CDR.DACCS_EJ_PER_GT_CENTRAL):
    """Capacity of each bin in each year [T, n] (GtCO2/yr): logistic scale-up, with
    DACCS capped by its energy limit. Works on numpy or jax arrays of low-carbon EJ."""
    price, cap, dac = bins
    K = float(cap.sum())
    phi = jnp.asarray(CDR.novel_cdr_path(years, K) / K)               # [T]
    capt = phi[:, None] * jnp.asarray(cap)[None, :]
    dac_full = float(cap[dac].sum())
    if dac_full > 0:
        limit = frac * jnp.asarray(lowcarbon_ej) / ej_per_gt
        scale = jnp.minimum(1.0, limit / (phi * dac_full))
        capt = capt * jnp.where(jnp.asarray(dac)[None, :], scale[:, None], 1.0)
    return capt


def spending_bn(r, capt, price):
    """USD billion per year to remove r[t] GtCO2, filling the cheapest bins first."""
    before = jnp.cumsum(capt, axis=1) - capt
    filled = jnp.clip(r[:, None] - before, 0.0, capt)
    return jnp.sum(filled * jnp.asarray(price)[None, :], axis=1)


def land_use_path(years_after_anchor) -> np.ndarray:
    """Land-use CO2 (GtCO2/yr) 2023..: 4.2 falling linearly to zero in 2050 (as in 11/12)."""
    yrs = np.asarray(years_after_anchor)
    n50 = int(np.sum(yrs <= 2050))
    land = np.zeros(len(yrs))
    land[:n50] = B.LAND_USE_CO2_2024 * np.linspace(1.0, 0.0, n50)
    return land


def lowcarbon_ej(traj):
    """Global low-carbon primary energy (EJ/yr) from a rollout, scaled like CO2."""
    E = traj["shares"] * traj["D"][..., None]
    return jnp.sum(E[:, :, LC_IDX], axis=(0, 2)) / TWH_PER_EJ / COVERAGE


def net_path(global_mt, removals_gt, land):
    """Global net CO2 (GtCO2/yr) for 2023.. from a panel path whose index 0 is 2022."""
    return global_mt[1:] / 1000.0 / COVERAGE + land - removals_gt


@dataclass
class CDRPlan:
    budget_gt: float
    spend_cap_tn: float
    met: bool
    peak_gt: float
    net_2100_gt: float
    effort: float              # relative to maximum feasible effort (1.0 = max)
    spend_tn: float            # USD trillion 2023-2100, undiscounted
    removals: np.ndarray       # [T-1] GtCO2/yr, 2023..
    capacity: np.ndarray       # [T-1] available removal capacity
    global_mt: np.ndarray      # [T] panel fossil+industry CO2 (Mt), index 0 = 2022
    lowcarbon_ej: np.ndarray   # [T]
    g: np.ndarray
    h: np.ndarray
    p: np.ndarray

    def summary(self) -> dict:
        yrs = 2023 + np.arange(len(self.removals))
        nz = CDR.net_zero_year(yrs, self.global_mt[1:] / 1000.0 / COVERAGE
                               + land_use_path(yrs) - self.removals)
        return {"budget_gt": self.budget_gt, "spend_cap_tn": self.spend_cap_tn, "met": self.met,
                "peak_gt": self.peak_gt, "net_2100_gt": self.net_2100_gt, "effort": self.effort,
                "spend_tn": self.spend_tn, "removal_2050_gt": float(self.removals[yrs == 2050][0]),
                "removal_2100_gt": float(self.removals[-1]),
                "removal_total_gt": float(self.removals.sum()),
                "gross_2050_gt": float(self.global_mt[1:][yrs == 2050][0] / 1000.0 / COVERAGE),
                "gross_2100_gt": float(self.global_mt[-1] / 1000.0 / COVERAGE),
                "fossil_cum_gt": float(self.global_mt[1:].sum() / 1000.0 / COVERAGE),
                "net_zero_year": nz,
                "mean_g": float(np.mean(self.g)), "mean_h": float(np.mean(self.h)),
                "mean_p": float(np.mean(self.p))}


def plan_with_removals(wm, params, scen: Scenario, budget_gt: float, spend_cap_tn: float,
                       open_loop: bool = False, frac: float = CDR.ENERGY_FRACTIONS["central"],
                       land_cap: float = CDR.LAND_CAPS["central"], steps: int = 1500,
                       lr: float = 0.05, seed: int = 0, pen_budget: float = 10.0,
                       pen_net: float = 10.0, pen_spend: float = 100.0) -> CDRPlan:
    """Minimum abatement effort meeting `budget_gt` (peak cumulative net CO2 from 2023,
    with net CO2 <= 0 in 2100) with removal spending of at most `spend_cap_tn`.

    open_loop=True pins the natural changes to business-as-usual (no learned reaction),
    as optimize.max_ambition_open_loop does."""
    years = np.asarray(scen.years)
    yrs = years[1:]
    land = jnp.asarray(land_use_path(yrs))
    bins = supply_bins(land_cap)
    price = bins[0]
    natural = None
    if open_loop:
        tr = O._rollout(wm, params, scen, None)
        natural = {"dI_nat": tr["dI_nat"], "ds_nat": tr["ds_nat"]}
    mx = (O.max_ambition_open_loop if open_loop else O.max_ambition)(wm, params, scen)
    effort_max = float(mx.cost)

    key = jax.random.PRNGKey(seed)
    shape = (scen.C, scen.Tf)
    raw = (-1.0 + 0.1 * jax.random.normal(key, shape),
           -1.0 + 0.1 * jax.random.normal(jax.random.fold_in(key, 1), shape),
           -1.0 + 0.1 * jax.random.normal(jax.random.fold_in(key, 2), shape),
           jnp.full(len(yrs), -2.0))
    opt = optax.adam(lr)
    st = opt.init(raw)

    def parts(rg, rh, rp, rr):
        controls, g, h = O._controls(rg, rh, scen, rp)
        tr = O._rollout(wm, params, scen, controls, natural)
        gmt = jnp.sum(tr["co2"], axis=0)
        lc = lowcarbon_ej(tr)
        capt = capacity(yrs, lc[1:], bins, frac)
        cap_tot = jnp.sum(capt, axis=1)
        r = cap_tot * jax.nn.sigmoid(rr) * (1.0 if spend_cap_tn > 0 else 0.0)
        spend = jnp.sum(spending_bn(r, capt, price)) / 1000.0
        net = net_path(gmt, r, land)
        peak = jnp.max(jnp.cumsum(net))
        effort = O._cost(g, h, scen, controls["u_proc"]) / effort_max
        return dict(controls=controls, g=g, h=h, gmt=gmt, lc=lc, cap=cap_tot, r=r,
                    spend=spend, net=net, peak=peak, effort=effort)

    def obj(raw):
        q = parts(*raw)
        over = 100.0 * jax.nn.relu(q["peak"] - budget_gt) / budget_gt      # % of budget
        j = (q["effort"] + pen_budget * over ** 2 + pen_net * jax.nn.relu(q["net"][-1]) ** 2
             + SPEND_TIEBREAK * q["spend"])
        if spend_cap_tn > 0 and np.isfinite(spend_cap_tn):
            j = j + pen_spend * (100.0 * jax.nn.relu(q["spend"] - spend_cap_tn) / spend_cap_tn) ** 2
        return j

    @jax.jit
    def step(raw, st):
        j, grads = jax.value_and_grad(obj)(raw)
        up, st = opt.update(grads, st)
        return optax.apply_updates(raw, up), st, j

    for _ in range(steps):
        raw, st, _ = step(raw, st)
    q = jax.tree_util.tree_map(np.asarray, parts(*raw))
    peak, net_end = float(q["peak"]), float(q["net"][-1])
    return CDRPlan(budget_gt=float(budget_gt), spend_cap_tn=float(spend_cap_tn),
                   met=bool(peak <= budget_gt * (1 + MEET_TOL) and net_end <= NET_END_TOL),
                   peak_gt=peak, net_2100_gt=net_end, effort=float(q["effort"]),
                   spend_tn=float(q["spend"]), removals=np.asarray(q["r"]),
                   capacity=np.asarray(q["cap"]), global_mt=np.asarray(q["gmt"]),
                   lowcarbon_ej=np.asarray(q["lc"]), g=np.asarray(q["g"]), h=np.asarray(q["h"]),
                   p=np.asarray(q["controls"]["u_proc"]))
