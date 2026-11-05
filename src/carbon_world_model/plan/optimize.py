"""Decarbonisation planning: optimise interventions by gradient descent through the
differentiable world model (model-based planning / optimal control).

Controls, per country and year:
  g[c,t] in [0, g_max]  -- clean-transition effort (extra low-carbon-share gain / yr)
  h[c,t] in [0, h_max]  -- energy-efficiency effort (extra intensity decline / yr)

Both are bounded by historically observed maxima (see scenario.feasibility_bounds), so
optimised pathways stay within rates humanity has actually achieved. Because the whole
simulator is differentiable, the cost-optimal pathway is found directly by Adam, not by
scenario enumeration.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .scenario import Scenario


@dataclass
class PlanResult:
    co2: np.ndarray          # [C, Tf] emissions trajectory (Mt)
    shares: np.ndarray       # [C, Tf, F]
    D: np.ndarray            # [C, Tf]
    g: np.ndarray            # [C, Tf] clean-transition effort
    h: np.ndarray            # [C, Tf] efficiency effort
    global_co2: np.ndarray   # [Tf] sum across countries (Mt)
    cum_gt: float            # cumulative GtCO2 over horizon (excl. anchor year)
    cost: float              # total transition effort (model units)


def _controls(raw_g, raw_h, scen: Scenario):
    g = scen.g_max * jax.nn.sigmoid(raw_g)             # [C,Tf]
    h = scen.h_max * jax.nn.sigmoid(raw_h)             # [C,Tf]
    shift = scen.clean_weights[None, None, :] - scen.fossil_weights[:, None, :]  # [C,1,F]-ish
    u_deploy = g[:, :, None] * shift                   # [C,Tf,F], mass-conserving
    return {"u_int": h, "u_deploy": u_deploy}, g, h


def _rollout(wm, params, scen: Scenario, controls):
    from ..model.dynamics import rollout
    drivers = {"gdp": jnp.asarray(scen.gdp), "pop": jnp.asarray(scen.pop),
               "process": jnp.asarray(scen.process)}
    inits = {"I0": jnp.asarray(scen.I0), "s0": jnp.asarray(scen.s0)}
    return rollout(params, jnp.asarray(wm.static), wm.norm, jnp.asarray(wm.ef),
                   inits, drivers, controls, wm.hp, tfrac=jnp.asarray(scen.tfrac))


def _cost(g, h, scen: Scenario):
    """Transition effort, volume-weighted by energy-system size with increasing
    marginal cost (quadratic). Larger systems cost more per share-point to transform."""
    size = jnp.asarray(scen.gdp[:, 0]) / jnp.mean(jnp.asarray(scen.gdp[:, 0]))  # proxy ~ GDP
    w = size[:, None]
    return jnp.sum(w * (g ** 2 + h ** 2))


def _summaries(traj, g, h, scen):
    co2 = np.asarray(traj["co2"])           # [C,Tf]
    gco2 = co2.sum(axis=0)                   # [Tf]
    cum_gt = float(gco2[1:].sum() / 1000.0)  # exclude anchor year; Mt->Gt
    return co2, gco2, cum_gt


def business_as_usual(wm, params, scen: Scenario) -> PlanResult:
    C, Tf = scen.C, scen.Tf
    controls = {"u_int": jnp.zeros((C, Tf)), "u_deploy": jnp.zeros((C, Tf, len(scen.clean_weights)))}
    traj = _rollout(wm, params, scen, controls)
    co2, gco2, cum = _summaries(traj, None, None, scen)
    return PlanResult(co2=co2, shares=np.asarray(traj["shares"]), D=np.asarray(traj["D"]),
                      g=np.zeros((C, Tf)), h=np.zeros((C, Tf)), global_co2=gco2,
                      cum_gt=cum, cost=0.0)


def max_ambition(wm, params, scen: Scenario) -> PlanResult:
    big = 8.0
    raw_g = jnp.full((scen.C, scen.Tf), big)
    raw_h = jnp.full((scen.C, scen.Tf), big)
    controls, g, h = _controls(raw_g, raw_h, scen)
    traj = _rollout(wm, params, scen, controls)
    co2, gco2, cum = _summaries(traj, g, h, scen)
    return PlanResult(co2=co2, shares=np.asarray(traj["shares"]), D=np.asarray(traj["D"]),
                      g=np.asarray(g), h=np.asarray(h), global_co2=gco2, cum_gt=cum,
                      cost=float(_cost(g, h, scen)))


def optimize_pathway(wm, params, scen: Scenario, lam: float = 0.0,
                     budget_gt: float | None = None, penalty: float = 50.0,
                     steps: int = 1500, lr: float = 0.05, seed: int = 0,
                     verbose: bool = False) -> PlanResult:
    """Find controls minimising cumulative emissions + lam*cost, optionally with a hard
    cumulative carbon budget (penalty method)."""
    key = jax.random.PRNGKey(seed)
    raw_g = -1.0 + 0.1 * jax.random.normal(key, (scen.C, scen.Tf))
    raw_h = -1.0 + 0.1 * jax.random.normal(jax.random.fold_in(key, 1), (scen.C, scen.Tf))
    opt = optax.adam(lr)
    st = opt.init((raw_g, raw_h))

    def obj(rg, rh):
        controls, g, h = _controls(rg, rh, scen)
        traj = _rollout(wm, params, scen, controls)
        gco2 = jnp.sum(traj["co2"], axis=0)
        cum_gt = jnp.sum(gco2[1:]) / 1000.0
        cost = _cost(g, h, scen)
        j = cum_gt + lam * cost
        if budget_gt is not None:
            j = cost + penalty * jax.nn.relu(cum_gt - budget_gt) ** 2
        return j, (cum_gt, cost)

    @jax.jit
    def step(rg, rh, st):
        (j, aux), (grg, grh) = jax.value_and_grad(obj, argnums=(0, 1), has_aux=True)(rg, rh)
        up, st = opt.update((grg, grh), st)
        rg, rh = optax.apply_updates((rg, rh), up)
        return rg, rh, st, j, aux

    for i in range(steps):
        raw_g, raw_h, st, j, aux = step(raw_g, raw_h, st)
        if verbose and i % max(1, steps // 10) == 0:
            print(f"  opt step {i:4d} obj={float(j):.3f} cum={float(aux[0]):.1f}Gt cost={float(aux[1]):.2f}")

    controls, g, h = _controls(raw_g, raw_h, scen)
    traj = _rollout(wm, params, scen, controls)
    co2, gco2, cum = _summaries(traj, g, h, scen)
    return PlanResult(co2=co2, shares=np.asarray(traj["shares"]), D=np.asarray(traj["D"]),
                      g=np.asarray(g), h=np.asarray(h), global_co2=gco2, cum_gt=cum,
                      cost=float(_cost(g, h, scen)))


def pareto_frontier(wm, params, scen: Scenario, lams, steps: int = 800):
    """Trace the cost vs cumulative-emissions efficient frontier over a sweep of lam."""
    pts = []
    for lam in lams:
        r = optimize_pathway(wm, params, scen, lam=lam, steps=steps)
        pts.append({"lam": float(lam), "cum_gt": r.cum_gt, "cost": r.cost,
                    "final_global_mt": float(r.global_co2[-1])})
    return pts
