"""Why the planner keeps the network on the business-as-usual path.

Until 2026-11-13 the planner's rollouts were closed-loop: a control moved a country's
energy intensity or fuel mix away from the path that business as usual (BAU) follows,
and the learned dynamics then computed the next natural change from the controlled
state. Those states, for instance a far larger low-carbon share than any country of that
income has had, lie outside the training data, so the network's response there was
extrapolation, and training seeds extrapolated differently: maximum effort cut 2050
emissions by 26-59% depending on the seed.

This module keeps that closed loop (`closed_loop_members`, with the old maximum controls
`closed_loop_max`) to measure it, against two ways of keeping the network on the BAU
path: `open_loop` holds the natural changes at their BAU values and adds the old
controls on top, so every unit of control persists; the frontier is the planner itself
(plan/optimize.py), which lifts each country's total annual rates to the feasibility
limits, or leaves them where BAU is faster. `control_response` compares the three for
each ensemble member under maximum effort, for both controls together and each alone.
Used by scripts/control_response.py; see the README, "Why the training seeds disagree".
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..config import FUELS, LOW_CARBON_FUELS
from ..model.dynamics import rollout
from . import optimize as O
from .ensemble import max_controls
from .scenario import Scenario

LOWC_IDX = [FUELS.index(f) for f in LOW_CARBON_FUELS]
PLANS = ("both", "efficiency", "clean")  # both controls at maximum, or one of them alone
LOOPS = ("closed", "open", "frontier")


def effective_ef(wm, stacked) -> np.ndarray:
    """[K, C, F] emission factors of each member, with its per-country correction."""
    corr = np.exp(np.asarray(stacked["log_ef_corr"], dtype=float))
    return np.asarray(wm.ef, dtype=float)[None, None, :] * corr[:, :, None]


def log_intensity(run, scen: Scenario) -> np.ndarray:
    """[K, C, Tf] log primary energy per unit of GDP."""
    return np.log(np.asarray(run["D"], dtype=float)) - np.log(scen.gdp)[None]


def lowc_shares(run) -> np.ndarray:
    """[K, C, Tf] each country's low-carbon share of primary energy."""
    return np.asarray(run["shares"], dtype=float)[..., LOWC_IDX].sum(axis=-1)


def lowc_share(run) -> np.ndarray:
    """[K, Tf] energy-weighted low-carbon share of the modelled countries."""
    E = np.asarray(run["shares"], dtype=float) * np.asarray(run["D"], dtype=float)[..., None]
    return E[..., LOWC_IDX].sum(axis=(1, 3)) / E.sum(axis=(1, 3))


def open_loop(wm, stacked, scen: Scenario, bau: dict, g, h) -> dict:
    """co2 and D [K, C, Tf] and shares [K, C, Tf, F] of the plan (g, h), [C, Tf] or per
    member [K, C, Tf], when the natural changes of intensity and mix stay at their BAU
    values (`bau`, from closed_loop_members): the learned dynamics do not react to the
    controls. As in the closed loop, the control of year t moves the state of year t+1;
    with zero controls this is BAU."""
    K, C, Tf, F = bau["shares"].shape
    g = np.broadcast_to(np.asarray(g, dtype=float), (K, C, Tf))
    h = np.broadcast_to(np.asarray(h, dtype=float), (K, C, Tf))
    shift = scen.clean_weights[None, :] - scen.fossil_weights                  # [C, F]
    cum_h = np.concatenate([np.zeros((K, C, 1)), np.cumsum(h, axis=2)[..., :-1]], axis=2)
    D = np.asarray(bau["D"], dtype=float) * np.exp(-cum_h)
    bs = np.asarray(bau["shares"], dtype=float)
    ds = np.diff(bs, axis=2)
    s = np.empty_like(bs)
    s[:, :, 0] = bs[:, :, 0]
    for t in range(Tf - 1):
        raw = np.maximum(s[:, :, t] + ds[:, :, t] + g[:, :, t, None] * shift[None], 0.0)
        s[:, :, t + 1] = raw / raw.sum(axis=-1, keepdims=True)
    co2 = (effective_ef(wm, stacked)[:, :, None, :] * D[..., None] * s).sum(-1)
    return {"co2": co2 + scen.process[None], "D": D, "shares": s}


def closed_loop_max(scen: Scenario):
    """The old planner's maximum controls (g, h), each [C, Tf]: the limits, added to the
    network's own changes."""
    full = np.full((scen.C, scen.Tf), 1.0 / (1.0 + np.exp(-8.0)))
    return scen.g_max * full, scen.h_max * full


def closed_loop_members(wm, stacked, scen: Scenario, g=None, h=None) -> dict:
    """co2 and D [K, C, Tf] and shares [K, C, Tf, F] of the old planner's plan (g, h)
    under every member: the controls enter the network's state, and the network responds
    to it. Without g and h, business as usual."""
    controls = None
    if g is not None:
        shift = scen.clean_weights[None, None, :] - scen.fossil_weights[:, None, :]
        controls = {"u_int": jnp.asarray(h), "u_deploy": jnp.asarray(g)[:, :, None] * shift}
    drivers = {"gdp": jnp.asarray(scen.gdp), "pop": jnp.asarray(scen.pop),
               "process": jnp.asarray(scen.process)}
    inits = {k: jnp.asarray(v) for k, v in scen.inits.items()}

    def one(p):
        return rollout(p, jnp.asarray(wm.static), wm.norm, jnp.asarray(wm.ef), inits, drivers,
                       controls, wm.hp, tfrac=jnp.asarray(scen.tfrac))

    tr = jax.vmap(one)(stacked)
    return {k: np.asarray(tr[k]) for k in ("co2", "D", "shares")}


def control_response(wm, stacked, scen: Scenario, top: int = 8) -> dict:
    """Maximum effort in the closed loop, the open loop and at the frontier (the
    planner), for each member (lists of K), for both controls and each alone.

    cut_2050_pct and energy_change_2050_pct are relative to the member's BAU in 2050.
    intensity_fall_pct_yr and lowc_gain_pp_yr are each country's average annual rates
    over the horizon, weighted by its primary energy in the anchor year. Retention of the
    efficiency control is the closed loop's fall in log energy over the open loop's
    (efficiency alone); of the clean control, its gain in the energy-weighted low-carbon
    share over the open loop's (clean alone). by_country lists the `top` countries whose
    maximum-effort emissions in 2050 vary most between members in the closed loop.
    """
    g, h = closed_loop_max(scen)
    z = np.zeros_like(g)
    old = {"both": (g, h), "efficiency": (z, h), "clean": (g, z)}
    fg, fh = max_controls(scen)
    bau = closed_loop_members(wm, stacked, scen)
    W = O.make_worlds(wm, stacked, [scen])
    runs = {"closed": {k: closed_loop_members(wm, stacked, scen, *p) for k, p in old.items()},
            "open": {k: open_loop(wm, stacked, scen, bau, *p) for k, p in old.items()},
            "frontier": {k: {q: np.asarray(v) for q, v in O.apply_plan(W, *p).items()}
                         for k, p in {"both": (fg, fh), "efficiency": (z, fh),
                                      "clean": (fg, z)}.items()}}

    def last(run, key):
        return np.asarray(run[key], dtype=float).sum(axis=1)[:, -1]       # [K]

    years = scen.Tf - 1
    w = np.asarray(bau["D"], dtype=float)[:, :, 0]
    w = w / w.sum(axis=1, keepdims=True)                                   # [K, C]

    def fall(run):
        li = log_intensity(run, scen)
        return 100 * (w * (li[:, :, 0] - li[:, :, -1])).sum(axis=1) / years

    def gain(run):
        lc = lowc_shares(run)
        return 100 * (w * (lc[:, :, -1] - lc[:, :, 0])).sum(axis=1) / years

    b_co2, b_D, b_lc = last(bau, "co2"), last(bau, "D"), lowc_share(bau)[:, -1]
    out = {"scenario": scen.name, "bounds": {"h_max": scen.h_max, "g_max": scen.g_max},
           "bau_2050_mt": b_co2.tolist(), "lowc_share_2050_bau": b_lc.tolist(),
           "intensity_fall_pct_yr_bau": fall(bau).tolist(),
           "lowc_gain_pp_yr_bau": gain(bau).tolist()}
    for key, f in (("cut_2050_pct", lambda r: 100 * (1 - last(r, "co2") / b_co2)),
                   ("energy_change_2050_pct", lambda r: 100 * (last(r, "D") / b_D - 1)),
                   ("lowc_share_2050", lambda r: lowc_share(r)[:, -1]),
                   ("intensity_fall_pct_yr", fall), ("lowc_gain_pp_yr", gain)):
        out[key] = {loop: {k: f(r).tolist() for k, r in rs.items()} for loop, rs in runs.items()}
    dlog = {loop: np.log(b_D / last(runs[loop]["efficiency"], "D")) for loop in ("closed", "open")}
    dlc = {loop: lowc_share(runs[loop]["clean"])[:, -1] - b_lc for loop in ("closed", "open")}
    out["retention"] = {"efficiency": (dlog["closed"] / dlog["open"]).tolist(),
                        "clean": (dlc["closed"] / dlc["open"]).tolist()}
    m = np.asarray(runs["closed"]["both"]["co2"], dtype=float)[:, :, -1]   # [K, C]
    order = np.argsort(-m.std(axis=0))[:top]
    out["by_country"] = {scen.countries[c]: {
        "bau_2050_mt": np.asarray(bau["co2"], dtype=float)[:, c, -1].tolist(),
        **{f"max_2050_{loop}_mt": np.asarray(runs[loop]["both"]["co2"], dtype=float)[:, c, -1].tolist()
           for loop in LOOPS}}
        for c in order}
    return out
