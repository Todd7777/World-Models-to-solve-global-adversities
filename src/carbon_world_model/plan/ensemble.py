"""Planning under model uncertainty: an ensemble of training seeds of one configuration.

Retraining the selected configuration with different seeds gives fitted simulators whose
business-as-usual projections differ (scripts/seed_sensitivity.py), so planning numbers
from a single checkpoint depend on which seed happened to be committed. Here every plan
is evaluated under every member, and one plan (a single control trajectory) can be
optimised against all members at once: to meet a budget on average (mode='mean') or
under every member and scenario ('all'). Plans are applied at the feasibility frontier
of each member's business as usual (plan/optimize.py).

Members share the configuration, the data and therefore the WorldModel; only their
parameters differ. They are stacked along a leading axis and rolled out with jax.vmap.

On disk an ensemble is a directory of params_seed<k>.pkl files and a manifest.json that
records the configuration (without the per-member seed), the seeds and per-member notes.
Its seed-0 member is a copy of the committed single checkpoint, and load_ensemble() can
check that it still is.
"""
from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from ..config import Config, default_config
from ..eval.configs import FIT_SECTIONS, ProvenanceError, config_dict, config_diff
from ..train.fit import load_params, save_params
from . import optimize as O
from .scenario import Scenario

ENSEMBLE_DIR = "results/model/ensemble"
MANIFEST = "manifest.json"


def member_path(root: str, seed: int) -> str:
    return os.path.join(root, f"params_seed{seed}.pkl")


def _without_seed(cfg: dict) -> dict:
    """The sections of a config dict that determine a fit, without the training seed."""
    out = {s: cfg.get(s) for s in FIT_SECTIONS}
    out["train"] = {k: v for k, v in (out["train"] or {}).items() if k != "seed"}
    return out


def save_ensemble(root: str, cfg: Config, members: dict, info: dict | None = None) -> None:
    """Write the members ({seed: params}) and a manifest recording cfg without its seed, the
    seeds, and per-member notes ({seed: dict})."""
    os.makedirs(root, exist_ok=True)
    for s, p in members.items():
        save_params(p, member_path(root, s))
    seeds = sorted(members)
    manifest = {"config": _without_seed(config_dict(cfg)), "seeds": seeds,
                "members": {str(s): (info or {}).get(s, {}) for s in seeds}}
    with open(os.path.join(root, MANIFEST), "w") as fh:
        json.dump(manifest, fh, indent=2)


def _same_params(a, b) -> bool:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(np.array_equal(np.asarray(x), np.asarray(y))
                                      for x, y in zip(la, lb))


def load_ensemble(root: str = ENSEMBLE_DIR, cfg: Config | None = None,
                  checkpoint: str | None = None):
    """(seeds, [params]) of the ensemble in `root`.

    Raises ProvenanceError unless it was trained with `cfg` (default: default_config())
    apart from the seed, every member listed in the manifest exists, and, if `checkpoint`
    is given, its seed-0 member is that checkpoint.
    """
    cfg = default_config() if cfg is None else cfg
    path = os.path.join(root, MANIFEST)
    if not os.path.exists(path):
        raise ProvenanceError(f"{path} not found, so there is no seed ensemble to plan with. "
                              "Train one with scripts/02b_train_ensemble.py.")
    with open(path) as fh:
        manifest = json.load(fh)
    diff = config_diff(manifest["config"], _without_seed(config_dict(cfg)))
    if diff:
        raise ProvenanceError(
            f"the ensemble in {root} was not trained with the default configuration:\n"
            + "\n".join(f"  {k}: ensemble {x!r}, default {y!r}" for k, x, y in diff)
            + "\nRetrain it with scripts/02b_train_ensemble.py.")
    seeds = [int(s) for s in manifest["seeds"]]
    missing = [member_path(root, s) for s in seeds if not os.path.exists(member_path(root, s))]
    if missing:
        raise ProvenanceError(f"ensemble members listed in {path} are missing: "
                              + ", ".join(missing))
    members = [load_params(member_path(root, s)) for s in seeds]
    if checkpoint is not None:
        if 0 not in seeds or not _same_params(members[seeds.index(0)], load_params(checkpoint)):
            raise ProvenanceError(
                f"the seed-0 member of the ensemble in {root} is not {checkpoint}: the "
                "checkpoint was retrained after the ensemble was made. Rerun "
                "scripts/02b_train_ensemble.py.")
    return seeds, members


def stack_members(members: list) -> dict:
    """The members' parameters stacked along a new leading axis."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *members)


def max_controls(scen: Scenario):
    """Maximum effort as a plan (g, h), each [C, Tf]: unlimited extra rates, which every
    member caps at the room its business as usual leaves below the feasibility limits, so
    each country runs at the limits, or at its BAU rates where those are faster."""
    full = np.full((scen.C, scen.Tf), np.inf)
    return full, full.copy()


def rollout_members(wm, stacked, scen: Scenario, g=None, h=None) -> dict:
    """co2 and D [K, C, Tf] and shares [K, C, Tf, F] of one plan under every member;
    without g and h, business-as-usual."""
    run = O.apply_plan(O.make_worlds(wm, stacked, [scen]), g, h)
    return {k: np.asarray(v) for k, v in run.items()}


def _by_member(W: O.Worlds, K: int, S: int, g=None, h=None) -> dict:
    ev = O.evaluate_worlds(W, g, h)
    return {"cum_gt": ev["cum_gt"].reshape(K, S), "final_mt": ev["final_mt"].reshape(K, S),
            "global_mt": ev["global_mt"].reshape(K, S, -1)}


def _n_members(stacked) -> int:
    return jax.tree_util.tree_leaves(stacked)[0].shape[0]


def evaluate_members(wm, stacked, scens: list[Scenario], g=None, h=None) -> dict:
    """60-country emissions of one plan under every member and scenario: cumulative over
    the horizon (Gt) and final-year (Mt), [K, S], and global trajectories, [K, S, Tf]."""
    return _by_member(O.make_worlds(wm, stacked, scens), _n_members(stacked), len(scens), g, h)


def ensemble_pathway(wm, stacked, scens: list[Scenario], budget_gt: float, mode: str = "mean",
                     penalty: float = 2.0, steps: int = O.STEPS, lr: float = 0.05,
                     seed: int = 0) -> dict:
    """Minimum-effort single plan whose cumulative emissions stay within `budget_gt`: on
    average over members and scenarios (mode='mean') or under every one of them ('all').

    With one member this is optimize.optimize_pathway (one scenario, a budget) or
    optimize.robust_pathway (several scenarios, mode='all'), from the same starting plan.
    """
    if mode not in ("mean", "all"):
        raise ValueError(f"mode must be 'mean' or 'all', not {mode!r}")
    W = O.make_worlds(wm, stacked, scens)
    g, h = O.plan_against(W, scens[0], budget_gt=budget_gt, mode=mode, penalty=penalty,
                          steps=steps, lr=lr, seed=seed)
    return {"g": g, "h": h, "cost": float(O._cost(g, h, scens[0])),
            **_by_member(W, _n_members(stacked), len(scens), g, h)}
