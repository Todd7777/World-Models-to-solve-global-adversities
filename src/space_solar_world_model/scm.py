"""Causal queries on the space-solar world model (Pearl's three rungs).

The world model is a structural causal model (SCM): exogenous variables U (launch price,
hardware cost, array specific power, conversion efficiencies, radiation / atmosphere /
debris multipliers, ...) with priors from the scenario; endogenous variables given by the
deterministic structural equations in ``economics.system``; and actions (where to put the
arrays, how to build and use them) as interventions.

* **Rung 2 -- intervention**: ``intervene(action, do=...)`` returns the outcome
  distribution under do(action) (and optional do() on any mechanism node, e.g.
  ``{"fluence_rate": 0}`` for "a world without Van Allen belts").
* **Rung 3 -- counterfactual**: ``counterfactual(factual, evidence, alternatives)`` does
  abduction (importance-weights prior worlds by how well they reproduce what was
  observed under the factual action), action (swap in each alternative) and prediction
  (evaluate the alternative in the *same* posterior worlds). Because the worlds are
  shared, differences between options are paired, which is what makes "what would have
  happened had we put it elsewhere" sharper than comparing two marginal forecasts.
* **Decisions**: ``decision_analysis(actions)`` returns P(best), expected regret and
  quantiles across sampled worlds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .world import Action, SpaceSolarWorld


def wquantile(x, w, q):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x)
    x, w = x[m], w[m]
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    cw /= cw[-1]
    return np.interp(q, cw, x[o])


def summarize(x, w=None) -> dict:
    x = np.asarray(x, float)
    w = np.full(x.shape, 1.0 / x.size) if w is None else np.asarray(w, float) / np.sum(w)
    m = np.isfinite(x)
    mean = float(np.sum(w[m] * x[m]) / np.sum(w[m]))
    p05, p50, p95 = (float(v) for v in wquantile(x, w, [0.05, 0.5, 0.95]))
    return dict(mean=mean, median=p50, p05=p05, p95=p95)


@dataclass
class Posterior:
    u: dict
    w: np.ndarray
    ess: float
    loglik: np.ndarray


class CausalWorld:
    def __init__(self, world: SpaceSolarWorld):
        self.world = world

    # ------------------------------------------------------------- rung 2
    def intervene(self, action: Action, n: int = 4000, seed: int = 0, do: dict | None = None,
                  fixed: dict | None = None, u: dict | None = None, metric: str = "lcoe") -> dict:
        u = self.world.sample_u(n, seed, fixed) if u is None else u
        out = self.world.outcome(action, u, do)
        return dict(summary=summarize(out[metric]), samples=out[metric], outcome=out, u=u)

    # ------------------------------------------------------------- rung 3
    def abduct(self, factual: Action, evidence: dict, n: int = 40000, seed: int = 0,
               fixed: dict | None = None) -> Posterior:
        """Importance-weight prior worlds by the likelihood of ``evidence``.

        ``evidence`` maps a name to (fn(outcome, u) -> predicted value, observed value,
        sigma, kind) with kind "log" (sigma on the log ratio) or "abs".
        """
        u = self.world.sample_u(n, seed, fixed)
        out = self.world.outcome(factual, u)
        ll = np.zeros(n)
        for name, (fn, obs, sig, kind) in evidence.items():
            pred = np.asarray(fn(out, u), float)
            r = np.log(np.maximum(pred, 1e-30) / obs) if kind == "log" else pred - obs
            ll += -0.5 * (r / sig) ** 2
        ll = np.where(np.isfinite(ll), ll, -np.inf)
        w = np.exp(ll - ll.max())
        w /= w.sum()
        return Posterior(u, w, float(1.0 / np.sum(w ** 2)), ll)

    def counterfactual(self, factual: Action, evidence: dict, alternatives: dict, n: int = 40000,
                       seed: int = 0, metric: str = "lcoe", fixed: dict | None = None) -> dict:
        post = self.abduct(factual, evidence, n, seed, fixed)
        fact = self.world.outcome(factual, post.u)[metric]
        res = dict(ess=post.ess, n=n, factual=dict(prior=summarize(fact), posterior=summarize(fact, post.w)))
        for name, act in alternatives.items():
            alt = self.world.outcome(act, post.u)[metric]
            res[name] = dict(prior=summarize(alt), posterior=summarize(alt, post.w),
                             diff_posterior=summarize(alt - fact, post.w),
                             p_better_than_factual=float(np.sum(post.w * (alt < fact))))
        res["posterior_u"] = {k: summarize(v, post.w) for k, v in post.u.items()
                              if np.std(v) > 0}
        return res

    # ------------------------------------------------------------ decisions
    def decision_analysis(self, actions: dict, n: int = 4000, seed: int = 0, metric: str = "lcoe",
                          u: dict | None = None, w: np.ndarray | None = None) -> dict:
        u = self.world.sample_u(n, seed) if u is None else u
        names = list(actions)
        M = np.stack([self.world.outcome(actions[k], u)[metric] for k in names])   # [A, N]
        M = np.where(np.isfinite(M), M, np.inf)
        w = np.full(M.shape[1], 1.0 / M.shape[1]) if w is None else w / w.sum()
        best = np.argmin(M, axis=0)
        regret = M - M.min(axis=0)
        out = {}
        for i, k in enumerate(names):
            out[k] = dict(summarize(M[i], w), p_best=float(np.sum(w * (best == i))),
                          expected_regret=float(np.sum(w * regret[i])))
        return out
