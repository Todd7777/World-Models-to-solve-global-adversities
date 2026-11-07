"""Decarbonisation planning: optimise interventions by gradient descent through the
differentiable world model (model-based planning / optimal control).

Controls, per country and year:
  g[c,t] in [0, g_max]  -- clean-transition effort (extra low-carbon-share gain / yr)
  h[c,t] in [0, h_max]  -- energy-efficiency effort (extra intensity decline / yr)

Both are bounded by the fastest *sustained* national rates on record (see
scenario.feasibility_bounds), so optimised pathways stay within what has been achieved.
Controls do not feed the model's momentum state (dynamics.py), so the optimiser cannot
manufacture "free" future reductions.

Robust planning: because the simulator is batched and differentiable, a single control
trajectory can be optimised against several socioeconomic futures at once (e.g. all
five SSPs stacked along the batch axis), minimising expected or worst-case outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..model.dynamics import rollout
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
    shift = scen.clean_weights[None, None, :] - scen.fossil_weights[:, None, :]
    return {"u_int": h, "u_deploy": g[:, :, None] * shift}, g, h


def _inits(scen: Scenario) -> dict:
    return {k: jnp.asarray(v) for k, v in scen.inits.items()}


def _rollout(wm, params, scen: Scenario, controls):
    drivers = {"gdp": jnp.asarray(scen.gdp), "pop": jnp.asarray(scen.pop),
               "process": jnp.asarray(scen.process)}
    return rollout(params, jnp.asarray(wm.static), wm.norm, jnp.asarray(wm.ef),
                   _inits(scen), drivers, controls, wm.hp, tfrac=jnp.asarray(scen.tfrac))


def _cost(g, h, scen: Scenario):
    """Transition effort, volume-weighted by economic size with increasing marginal cost
    (quadratic). Larger systems cost more per share-point to transform."""
    size = jnp.asarray(scen.gdp[:, 0]) / jnp.mean(jnp.asarray(scen.gdp[:, 0]))
    return jnp.sum(size[:, None] * (g ** 2 + h ** 2))


def _summaries(traj):
    co2 = np.asarray(traj["co2"])
    gco2 = co2.sum(axis=0)
    return co2, gco2, float(gco2[1:].sum() / 1000.0)


def _result(traj, g, h, scen) -> PlanResult:
    co2, gco2, cum = _summaries(traj)
    C, Tf = scen.C, scen.Tf
    return PlanResult(co2=co2, shares=np.asarray(traj["shares"]), D=np.asarray(traj["D"]),
                      g=np.zeros((C, Tf)) if g is None else np.asarray(g),
                      h=np.zeros((C, Tf)) if h is None else np.asarray(h),
                      global_co2=gco2, cum_gt=cum,
                      cost=0.0 if g is None else float(_cost(g, h, scen)))


def business_as_usual(wm, params, scen: Scenario) -> PlanResult:
    return _result(_rollout(wm, params, scen, None), None, None, scen)


def max_ambition(wm, params, scen: Scenario) -> PlanResult:
    big = jnp.full((scen.C, scen.Tf), 8.0)
    controls, g, h = _controls(big, big, scen)
    return _result(_rollout(wm, params, scen, controls), g, h, scen)


def optimize_pathway(wm, params, scen: Scenario, lam: float = 0.0,
                     budget_gt: float | None = None, penalty: float = 50.0,
                     steps: int = 1500, lr: float = 0.05, seed: int = 0,
                     verbose: bool = False) -> PlanResult:
    """Minimise cumulative emissions + lam*cost, or cost subject to a cumulative carbon
    budget (penalty method)."""
    key = jax.random.PRNGKey(seed)
    raw_g = -1.0 + 0.1 * jax.random.normal(key, (scen.C, scen.Tf))
    raw_h = -1.0 + 0.1 * jax.random.normal(jax.random.fold_in(key, 1), (scen.C, scen.Tf))
    opt = optax.adam(lr)
    st = opt.init((raw_g, raw_h))

    def obj(rg, rh):
        controls, g, h = _controls(rg, rh, scen)
        gco2 = jnp.sum(_rollout(wm, params, scen, controls)["co2"], axis=0)
        cum_gt = jnp.sum(gco2[1:]) / 1000.0
        cost = _cost(g, h, scen)
        j = cum_gt + lam * cost
        if budget_gt is not None:
            j = cost + penalty * jax.nn.relu(cum_gt - budget_gt) ** 2
        return j, (cum_gt, cost)

    @jax.jit
    def step(rg, rh, st):
        (j, aux), grads = jax.value_and_grad(obj, argnums=(0, 1), has_aux=True)(rg, rh)
        up, st = opt.update(grads, st)
        rg, rh = optax.apply_updates((rg, rh), up)
        return rg, rh, st, j, aux

    for i in range(steps):
        raw_g, raw_h, st, j, aux = step(raw_g, raw_h, st)
        if verbose and i % max(1, steps // 10) == 0:
            print(f"  opt step {i:4d} obj={float(j):.3f} cum={float(aux[0]):.1f}Gt cost={float(aux[1]):.2f}")
    controls, g, h = _controls(raw_g, raw_h, scen)
    return _result(_rollout(wm, params, scen, controls), g, h, scen)


def pareto_frontier(wm, params, scen: Scenario, budgets_gt, steps: int = 1200,
                    penalty: float = 2.0):
    """Cost-of-abatement frontier: minimum effort that meets each cumulative budget."""
    pts = []
    for bud in budgets_gt:
        r = optimize_pathway(wm, params, scen, budget_gt=float(bud), penalty=penalty, steps=steps)
        pts.append({"budget_gt": float(bud), "cum_gt": r.cum_gt, "cost": r.cost,
                    "final_global_mt": float(r.global_co2[-1])})
    return pts


# ------------------------------------------------------------------ robust planning
def _stack(scens: list[Scenario]) -> dict:
    C = scens[0].C
    cat = lambda xs: jnp.asarray(np.concatenate(xs, axis=0))  # noqa: E731
    return {
        "S": len(scens), "C": C,
        "drivers": {"gdp": cat([s.gdp for s in scens]), "pop": cat([s.pop for s in scens]),
                    "process": cat([s.process for s in scens])},
        "inits": {k: cat([s.inits[k] for s in scens]) for k in scens[0].inits},
        "country_idx": jnp.asarray(np.tile(np.arange(C), len(scens))),
        "tfrac": jnp.asarray(scens[0].tfrac),
    }


def _rollout_stack(wm, params, stk, controls):
    """Run one shared control trajectory [C,...] through all stacked scenarios."""
    S = stk["S"]
    tiled = None if controls is None else {
        "u_int": jnp.tile(controls["u_int"], (S, 1)),
        "u_deploy": jnp.tile(controls["u_deploy"], (S, 1, 1)),
    }
    tr = rollout(params, jnp.asarray(wm.static), wm.norm, jnp.asarray(wm.ef), stk["inits"],
                 stk["drivers"], tiled, wm.hp, tfrac=stk["tfrac"], country_idx=stk["country_idx"])
    return tr["co2"].reshape(S, stk["C"], -1)       # [S, C, Tf]


def evaluate_controls(wm, params, scens: list[Scenario], g=None, h=None) -> dict:
    """Cumulative and 2050 emissions of one control plan in each scenario."""
    stk = _stack(scens)
    controls = None
    if g is not None:
        shift = scens[0].clean_weights[None, None, :] - scens[0].fossil_weights[:, None, :]
        controls = {"u_int": jnp.asarray(h), "u_deploy": jnp.asarray(g)[:, :, None] * shift}
    co2 = np.asarray(_rollout_stack(wm, params, stk, controls))
    g_s = co2.sum(axis=1)                              # [S, Tf]
    return {"cum_gt": (g_s[:, 1:].sum(axis=1) / 1000.0).tolist(),
            "final_mt": g_s[:, -1].tolist(), "global_mt": g_s.tolist()}


def robust_pathway(wm, params, scens: list[Scenario], budget_gt: float, penalty: float = 2.0,
                   mode: str = "all", steps: int = 1500, lr: float = 0.05,
                   seed: int = 0) -> dict:
    """Minimum-effort single plan that keeps cumulative emissions within `budget_gt`
    under every stacked scenario (mode='all') or on average (mode='mean')."""
    s0 = scens[0]
    stk = _stack(scens)
    key = jax.random.PRNGKey(seed)
    raw_g = -1.0 + 0.1 * jax.random.normal(key, (s0.C, s0.Tf))
    raw_h = -1.0 + 0.1 * jax.random.normal(jax.random.fold_in(key, 1), (s0.C, s0.Tf))
    opt = optax.adam(lr)
    st = opt.init((raw_g, raw_h))

    def obj(rg, rh):
        controls, g, h = _controls(rg, rh, s0)
        cum = jnp.sum(_rollout_stack(wm, params, stk, controls)[:, :, 1:], axis=(1, 2)) / 1000.0
        cost = _cost(g, h, s0)
        viol = jax.nn.relu(cum - budget_gt)
        pen = jnp.sum(viol ** 2) if mode == "all" else jax.nn.relu(jnp.mean(cum) - budget_gt) ** 2
        return cost + penalty * pen, (cum, cost)

    @jax.jit
    def step(rg, rh, st):
        (j, aux), grads = jax.value_and_grad(obj, argnums=(0, 1), has_aux=True)(rg, rh)
        up, st = opt.update(grads, st)
        rg, rh = optax.apply_updates((rg, rh), up)
        return rg, rh, st, aux

    for _ in range(steps):
        raw_g, raw_h, st, aux = step(raw_g, raw_h, st)
    _, g, h = _controls(raw_g, raw_h, s0)
    ev = evaluate_controls(wm, params, scens, np.asarray(g), np.asarray(h))
    return {"g": np.asarray(g), "h": np.asarray(h), "cost": float(_cost(g, h, s0)), **ev}
