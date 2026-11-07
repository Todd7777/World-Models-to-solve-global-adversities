"""Planning inside the world model: optimal design for each placement, then ranking.

For every candidate placement the planner finds the cost-minimising design (array
power, transmitter aperture, receiver size and, for shared constellations, the number
of satellites and receiving sites) by a vectorised coarse search followed by L-BFGS-B
on the *exact gradient* of log(LCOE), obtained by automatic differentiation through
the structural model. Transfer options (chemical vs solar-electric tug) are discrete
and enumerated.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .economics.system import placement_vector, shared_availability, system
from .physics.beaming import LINKS
from .world import DEDICATED, Action, SpaceSolarWorld

BOUNDS = dict(P_sat=(1e5, 4e10), d_t=(0.5, 8000.0), d_r=(0.5, 40000.0), T=(1.0, 3e4))


@dataclass
class PlanResult:
    action: Action
    outcome: dict

    @property
    def lcoe(self) -> float:
        return float(self.outcome["lcoe"])


def _var_names(mode: str):
    return ["P_sat", "d_t", "d_r"] + (["T"] if mode == "shared" else [])


def _design_from_x(x, names):
    return {n: 10.0 ** x[i] for i, n in enumerate(names)}


def optimize_beam(world: SpaceSolarWorld, action: Action, u: dict | None = None,
                  n_starts: int = 4) -> PlanResult:
    """Optimise the continuous design of a beaming action (placement/tech/sites fixed)."""
    u = world.central_u() if u is None else u
    tech = LINKS[action.tech]
    link = world.link_arrays(action)
    names = _var_names(action.mode)
    best = None
    for transfer in world.placement(action.placement).transport:
        pv = placement_vector(world.placement(action.placement), transfer)
        D_med = float(np.median(link["D"][0][link["w"][0] > 0]))
        d0 = math.sqrt(tech.wavelength_m * D_med * 4.0 / math.pi)
        grid = dict(P_sat=np.logspace(6, 10.5, 10), d_t=d0 * np.logspace(-1.5, 1.0, 7),
                    d_r=d0 * np.logspace(-1.0, 1.5, 7))
        if action.mode == "shared":
            T_grid = np.logspace(0.5, 4.0, 8)
        else:
            T_grid = [None]
        cands = []
        G = np.array(list(itertools.product(grid["P_sat"], grid["d_t"], grid["d_r"])))
        G[:, 1] = np.clip(G[:, 1], *BOUNDS["d_t"])
        G[:, 2] = np.clip(G[:, 2], *BOUNDS["d_r"])
        for T in T_grid:
            lk = dict(link)
            design = dict(P_sat=G[:, 0], d_t=G[:, 1], d_r=G[:, 2])
            if T is not None:
                lk["avail"] = shared_availability(np.float64(T), link["q"], np)
                design["T"] = np.full(len(G), T)
            out = system(pv, u, design, lk, tech, "beam")
            lc = np.asarray(out["lcoe"])
            lc = np.where(np.isfinite(lc), lc, np.inf)
            for i in np.argsort(lc)[:n_starts]:
                x = np.log10([G[i, 0], G[i, 1], G[i, 2]] + ([T] if T is not None else []))
                cands.append((lc[i], x))
        cands.sort(key=lambda c: c[0])
        starts = [c[1] for c in cands[:n_starts]]

        def f(x):
            d = {n: 10.0 ** x[i] for i, n in enumerate(names)}
            return jnp.log(system(pv, u, d, link, tech, "beam")["lcoe"])

        vg = jax.jit(jax.value_and_grad(f))

        def fun(x):
            v, g = vg(jnp.asarray(x, dtype=jnp.float32))
            v, g = float(v), np.asarray(g, dtype=np.float64)
            if not np.isfinite(v):
                return 1e3, np.zeros_like(x)
            return v, np.nan_to_num(g)

        bounds = [tuple(np.log10(BOUNDS[n])) for n in names]
        for x0 in starts:
            res = minimize(fun, np.asarray(x0, float), jac=True, method="L-BFGS-B", bounds=bounds,
                           options=dict(maxiter=300, ftol=1e-9, gtol=1e-6))
            lc = float(np.exp(res.fun))
            if best is None or lc < best[0]:
                best = (lc, transfer, _design_from_x(res.x, names))
    lc, transfer, design = best
    act = Action(action.placement, "beam", action.tech, transfer, action.site, action.n_sites,
                 tuple(sorted(design.items())))
    return PlanResult(act, world.outcome(act, u))


def best_network(world: SpaceSolarWorld, placement: str, site: str, tech: str,
                 site_options=(1, 4, 12, 24), u: dict | None = None) -> PlanResult:
    """Shared constellations: also choose how many receiving sites to build."""
    best = None
    for n in site_options:
        r = optimize_beam(world, Action(placement, "beam", tech, "sep_tug", site, n), u)
        if best is None or r.lcoe < best.lcoe:
            best = r
    return best


def in_space(world: SpaceSolarWorld, placement: str, firm: bool = True,
             u: dict | None = None) -> PlanResult:
    """Cost of electricity *used in orbit* at a placement (no beaming); picks the transfer."""
    use = "space_firm" if firm else "space_asavail"
    best = None
    for transfer in world.placement(placement).transport:
        act = Action(placement, use, None, transfer, None, 1, (("P_sat", 1e6),))
        out = world.outcome(act, u)
        if best is None or float(out["lcoe"]) < best.lcoe:
            best = PlanResult(act, out)
    return best


def plan_beam(world: SpaceSolarWorld, placement: str, site: str, tech: str,
              u: dict | None = None, site_options=(1, 4, 12, 24)) -> PlanResult:
    if placement in DEDICATED:
        return optimize_beam(world, Action(placement, "beam", tech, "sep_tug", site, 1), u)
    return best_network(world, placement, site, tech, site_options, u)
