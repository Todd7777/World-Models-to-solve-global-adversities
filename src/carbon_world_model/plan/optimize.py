"""Decarbonisation planning at the feasibility frontier.

The learned dynamics project business as usual (BAU): each country's natural path of
energy intensity and fuel mix, given its GDP and population. A plan adds policy-driven
change to that path, per country and year:

  g[c,t] >= 0  extra annual gain in the low-carbon share of primary energy
  h[c,t] >= 0  extra annual fall in log energy intensity (efficiency)

The total rates may not exceed the feasibility limits, the rates countries have
sustained for a decade (scenario.feasibility_bounds). In each world a plan therefore
achieves min(g, max(0, g_max - BAU gain)) and min(h, max(0, h_max - BAU fall)), and
maximum effort lifts every country to the limits, or leaves it where BAU is faster.
The planned low-carbon energy accumulates as a share of primary energy, split across
low-carbon sources in a fixed deployable portfolio (scenario.CLEAN_PORTFOLIO), and
displaces fossil fuels in proportion to their shares in BAU that year, until none is
left; effort beyond that is lost.

The network is not run on the planned states. No country has been in them, so its
response there would be extrapolation, and training seeds extrapolate differently: when
the planner fed its controls back into the network (until 2026-11-13), maximum effort
cut 2050 emissions by 26-59% depending on the seed (plan/response.py). Planned change is
therefore assumed to persist, with no rebound and no crowding out of BAU change. Effort
is charged on the plan, whether or not a world has room for all of it.

Everything after BAU is closed-form and differentiable (the emissions layer's fixed
physics applied to the planned energy demand and mix), so plans are optimised by
gradient descent, and one plan can be optimised against several worlds at once: the
five SSPs, the members of a seed ensemble (plan/ensemble.py), or both.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.special
import numpy as np
import optax

from ..config import FOSSIL_FUELS, FUELS, LOW_CARBON_FUELS
from ..model.dynamics import rollout
from .scenario import Scenario

LOWC_IDX = np.array([FUELS.index(f) for f in LOW_CARBON_FUELS])
FOSSIL_IDX = np.array([FUELS.index(f) for f in FOSSIL_FUELS])


@dataclass
class PlanResult:
    co2: np.ndarray          # [C, Tf] emissions trajectory (Mt)
    shares: np.ndarray       # [C, Tf, F]
    D: np.ndarray            # [C, Tf]
    g: np.ndarray            # [C, Tf] extra low-carbon-share gain achieved (fraction/yr)
    h: np.ndarray            # [C, Tf] extra fall in log intensity achieved (per yr)
    global_co2: np.ndarray   # [Tf] sum across countries (Mt)
    cum_gt: float            # cumulative GtCO2 over horizon (excl. anchor year)
    cost: float              # transition effort of the plan (model units)


class Worlds(NamedTuple):
    """Business-as-usual projections that plans are applied to: W worlds (ensemble members
    x scenarios, member-major) along the leading axis."""
    co2: jnp.ndarray         # [W, C, Tf] BAU emissions (Mt)
    D: jnp.ndarray           # [W, C, Tf] BAU primary energy (TWh)
    shares: jnp.ndarray      # [W, C, Tf, F] BAU fuel shares
    ef: jnp.ndarray          # [W, C, F] emission factors with the member's correction
    process: jnp.ndarray     # [W, C, Tf] process CO2 (Mt)
    gap_g: jnp.ndarray       # [W, C, Tf] extra low-carbon gain the limit leaves room for
    gap_h: jnp.ndarray       # [W, C, Tf] extra intensity fall the limit leaves room for
    clean: jnp.ndarray       # [F] how planned low-carbon energy splits across sources


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


def make_worlds(wm, stacked, scens: list[Scenario]) -> Worlds:
    """BAU of every member of `stacked` (parameters with a leading member axis) in every
    scenario, and the room each leaves below the feasibility limits. The network runs
    here only, without controls."""
    stk = _stack(scens)
    K = jax.tree_util.tree_leaves(stacked)[0].shape[0]
    S, C = stk["S"], stk["C"]
    s0 = scens[0]

    def bau(p):
        return rollout(p, jnp.asarray(wm.static), wm.norm, jnp.asarray(wm.ef), stk["inits"],
                       stk["drivers"], None, wm.hp, tfrac=stk["tfrac"],
                       country_idx=stk["country_idx"])

    tr = jax.vmap(bau)(stacked)                                   # [K, S*C, Tf(, F)]
    co2, D, shares, li = (tr[k].reshape((K * S, C) + tr[k].shape[2:])
                          for k in ("co2", "D", "shares", "intensity"))
    fall = li[..., :-1] - li[..., 1:]
    lowc = shares[..., LOWC_IDX].sum(axis=-1)
    gain = lowc[..., 1:] - lowc[..., :-1]
    pad = jnp.zeros((K * S, C, 1))                # the last year's plan acts beyond the horizon
    ef = wm.ef[None, None, :] * jnp.exp(stacked["log_ef_corr"])[:, :, None]       # [K, C, F]
    process = stk["drivers"]["process"].reshape(1, S, C, -1)
    return Worlds(
        co2=co2, D=D, shares=shares, ef=jnp.repeat(ef, S, axis=0),
        process=jnp.broadcast_to(process, (K, S) + process.shape[2:]).reshape(K * S, C, -1),
        gap_g=jnp.concatenate([jnp.maximum(0.0, s0.g_max - gain), pad], axis=-1),
        gap_h=jnp.concatenate([jnp.maximum(0.0, s0.h_max - fall), pad], axis=-1),
        clean=jnp.asarray(s0.clean_weights))


def apply_plan(W: Worlds, g=None, h=None) -> dict:
    """co2 and D [W, C, Tf] and shares [W, C, Tf, F] of the plan (g, h), each [C, Tf], in
    every world; without a plan, BAU. Each world caps the plan's extra rates at the room
    it leaves below the limits, so a plan of unlimited rates is maximum effort. The plan
    of year t changes the state of year t+1."""
    if g is None and h is None:
        return {"co2": W.co2, "D": W.D, "shares": W.shares}
    zero = jnp.zeros(W.gap_g.shape[1:])
    gw = jnp.minimum(zero if g is None else jnp.asarray(g), W.gap_g)
    hw = jnp.minimum(zero if h is None else jnp.asarray(h), W.gap_h)
    D = W.D * jnp.exp(-(jnp.cumsum(hw, axis=-1) - hw))
    fossil = jnp.zeros_like(W.shares).at[..., FOSSIL_IDX].set(W.shares[..., FOSSIL_IDX])
    F = fossil.sum(axis=-1)                                              # [W, C, Tf]
    # Planned low-carbon share so far: X(t+1) = min(X(t) + g(t), F(t+1)), X(0) = 0, so
    # effort beyond the fossil fuel left to displace is lost (closed form: G + cummin).
    G = jnp.cumsum(gw, axis=-1) - gw
    X = G + jnp.minimum(0.0, jax.lax.cummin(F - G, axis=2))
    left = jnp.clip(1.0 - X / jnp.maximum(F, 1e-9), 0.0, 1.0)      # fossil not displaced
    shares = W.shares - fossil + fossil * left[..., None] + X[..., None] * W.clean
    co2 = jnp.sum(W.ef[:, :, None, :] * D[..., None] * shares, axis=-1) + W.process
    return {"co2": co2, "D": D, "shares": shares}


def max_plan(W: Worlds):
    """(g, h), each [C, Tf]: the largest extra rates any world has room for. Applied to
    every world it is maximum effort, and it bounds the plans the optimiser considers."""
    return W.gap_g.max(axis=0), W.gap_h.max(axis=0)


def _cost(g, h, scen: Scenario):
    """Transition effort, volume-weighted by economic size with increasing marginal cost
    (quadratic). Larger systems cost more per share-point to transform."""
    size = jnp.asarray(scen.gdp[:, 0]) / jnp.mean(jnp.asarray(scen.gdp[:, 0]))
    return jnp.sum(size[:, None] * (g ** 2 + h ** 2))


def _cum_gt(co2):
    """[W] cumulative emissions over the horizon (Gt), excluding the anchor year."""
    return jnp.sum(co2[..., 1:], axis=(-2, -1)) / 1000.0


STEPS = 20_000  # Adam steps per plan: at 1,000-1,500 the plans were far from converged


def plan_against(W: Worlds, scen: Scenario, budget_gt: float | None = None, lam: float = 0.0,
                 mode: str = "all", penalty: float = 2.0, steps: int = STEPS, lr: float = 0.05,
                 seed: int = 0, verbose: bool = False, tau: float = 0.5):
    """One plan (g, h), each [C, Tf], optimised against every world of W.

    With a budget: the minimum-effort plan whose cumulative emissions stay within it in
    every world (mode='all') or on average over the worlds ('mean'), by a quadratic
    penalty. 'all' bounds a smooth maximum over the worlds, tau * logsumexp(cum / tau)
    (Gt), which exceeds the largest by at most tau * ln(W): the plan meets the budget in
    every world, with at most that much to spare. Penalising each world's excess
    separately converges far more slowly. Without a budget: minimise mean cumulative
    emissions + lam * effort. Effort is
    weighted by `scen`'s GDP in the anchor year, the same in every scenario. Adam with a
    learning rate decaying from `lr` to zero over the steps (cosine): each world caps the
    plan at its own room, so the objective has kinks that defeat quasi-Newton methods,
    and with a constant rate Adam converges slowly. At 20,000 steps the cost of an
    ensemble-mean plan was within 0.1% of its value at 50,000.
    """
    if mode not in ("mean", "all"):
        raise ValueError(f"mode must be 'mean' or 'all', not {mode!r}")
    G, H = max_plan(W)
    key = jax.random.PRNGKey(seed)
    raw_g = -1.0 + 0.1 * jax.random.normal(key, (scen.C, scen.Tf))
    raw_h = -1.0 + 0.1 * jax.random.normal(jax.random.fold_in(key, 1), (scen.C, scen.Tf))
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps))
    st = opt.init((raw_g, raw_h))

    def obj(rg, rh, W):
        g, h = G * jax.nn.sigmoid(rg), H * jax.nn.sigmoid(rh)
        cum = _cum_gt(apply_plan(W, g, h)["co2"])
        cost = _cost(g, h, scen)
        if budget_gt is None:
            return jnp.mean(cum) + lam * cost, (cum, cost)
        bound = tau * jax.scipy.special.logsumexp(cum / tau) if mode == "all" else jnp.mean(cum)
        return cost + penalty * jax.nn.relu(bound - budget_gt) ** 2, (cum, cost)

    @jax.jit
    def step(rg, rh, st, W):
        (j, aux), grads = jax.value_and_grad(obj, argnums=(0, 1), has_aux=True)(rg, rh, W)
        up, st = opt.update(grads, st)
        rg, rh = optax.apply_updates((rg, rh), up)
        return rg, rh, st, j, aux

    for i in range(steps):
        raw_g, raw_h, st, j, aux = step(raw_g, raw_h, st, W)
        if verbose and i % max(1, steps // 10) == 0:
            print(f"  opt step {i:4d} obj={float(j):.3f} cum={float(jnp.mean(aux[0])):.1f}Gt "
                  f"cost={float(aux[1]):.2f}")
    return np.asarray(G * jax.nn.sigmoid(raw_g)), np.asarray(H * jax.nn.sigmoid(raw_h))


def evaluate_worlds(W: Worlds, g=None, h=None) -> dict:
    """Totals over the countries of one plan in every world: cumulative over the horizon
    (Gt) and final-year (Mt), [W], and trajectories, [W, Tf]."""
    gs = np.asarray(apply_plan(W, g, h)["co2"], dtype=np.float32).sum(axis=1)
    return {"cum_gt": gs[:, 1:].sum(axis=1) / 1000.0, "final_mt": gs[:, -1], "global_mt": gs}


# ------------------------------------------------------- one model (a single member)
def _one(params):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], params)


def _result(W: Worlds, g, h, scen: Scenario) -> PlanResult:
    """The plan in the single world of W."""
    run = apply_plan(W, g, h)
    co2 = np.asarray(run["co2"][0])
    gco2 = co2.sum(axis=0)
    zero = np.zeros((scen.C, scen.Tf))
    gw = zero if g is None else np.minimum(np.asarray(g), np.asarray(W.gap_g[0]))
    hw = zero if h is None else np.minimum(np.asarray(h), np.asarray(W.gap_h[0]))
    return PlanResult(co2=co2, shares=np.asarray(run["shares"][0]), D=np.asarray(run["D"][0]),
                      g=gw, h=hw, global_co2=gco2, cum_gt=float(gco2[1:].sum() / 1000.0),
                      cost=0.0 if g is None and h is None else float(_cost(gw, hw, scen)))


def business_as_usual(wm, params, scen: Scenario) -> PlanResult:
    return _result(make_worlds(wm, _one(params), [scen]), None, None, scen)


def effort_pathway(wm, params, scen: Scenario, frac: float) -> PlanResult:
    """A uniform fraction of maximum effort: every country, every year, closes `frac` of
    the room between its BAU rates and the limits."""
    W = make_worlds(wm, _one(params), [scen])
    g, h = max_plan(W)
    return _result(W, frac * g, frac * h, scen)


def max_ambition(wm, params, scen: Scenario) -> PlanResult:
    return effort_pathway(wm, params, scen, 1.0)


def optimize_pathway(wm, params, scen: Scenario, lam: float = 0.0,
                     budget_gt: float | None = None, penalty: float = 50.0,
                     steps: int = STEPS, lr: float = 0.05, seed: int = 0,
                     verbose: bool = False) -> PlanResult:
    """Minimise cumulative emissions + lam*cost, or cost subject to a cumulative carbon
    budget (penalty method)."""
    W = make_worlds(wm, _one(params), [scen])
    g, h = plan_against(W, scen, budget_gt=budget_gt, lam=lam, penalty=penalty, steps=steps,
                        lr=lr, seed=seed, verbose=verbose)
    return _result(W, g, h, scen)


def pareto_frontier(wm, params, scen: Scenario, budgets_gt, steps: int = STEPS,
                    penalty: float = 2.0):
    """Cost-of-abatement frontier: minimum effort that meets each cumulative budget."""
    pts = []
    for bud in budgets_gt:
        r = optimize_pathway(wm, params, scen, budget_gt=float(bud), penalty=penalty, steps=steps)
        pts.append({"budget_gt": float(bud), "cum_gt": r.cum_gt, "cost": r.cost,
                    "final_global_mt": float(r.global_co2[-1])})
    return pts


# ------------------------------------------------------------------ robust planning
def evaluate_controls(wm, params, scens: list[Scenario], g=None, h=None) -> dict:
    """Cumulative and final-year emissions of one plan in each scenario."""
    ev = evaluate_worlds(make_worlds(wm, _one(params), scens), g, h)
    return {"cum_gt": ev["cum_gt"].tolist(), "final_mt": ev["final_mt"].tolist(),
            "global_mt": ev["global_mt"].tolist()}


def robust_pathway(wm, params, scens: list[Scenario], budget_gt: float, penalty: float = 2.0,
                   mode: str = "all", steps: int = STEPS, lr: float = 0.05,
                   seed: int = 0) -> dict:
    """Minimum-effort single plan that keeps cumulative emissions within `budget_gt`
    under every scenario (mode='all') or on average (mode='mean')."""
    W = make_worlds(wm, _one(params), scens)
    g, h = plan_against(W, scens[0], budget_gt=budget_gt, mode=mode, penalty=penalty,
                        steps=steps, lr=lr, seed=seed)
    ev = evaluate_worlds(W, g, h)
    return {"g": g, "h": h, "cost": float(_cost(g, h, scens[0])),
            "cum_gt": ev["cum_gt"].tolist(), "final_mt": ev["final_mt"].tolist(),
            "global_mt": ev["global_mt"].tolist()}
