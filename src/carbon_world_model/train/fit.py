"""Fit the world model across countries by differentiable simulation (optax Adam)."""
from __future__ import annotations

from dataclasses import dataclass
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..config import Config
from ..model.world_model import WorldModel
from .losses import loss_terms, ef_regulariser


@dataclass
class FitResult:
    params: dict
    history: list[dict]
    wm: WorldModel


def _targets(panel) -> dict:
    return {
        "co2": jnp.asarray(panel.co2),
        "D": jnp.asarray(panel.primary_energy),
        "shares": jnp.asarray(panel.shares()),
    }


def year_window_mask(panel, year_max: int | None) -> jnp.ndarray:
    """[C,T] mask = observed AND year <= year_max (for temporal holdout training)."""
    m = panel.mask.copy()
    if year_max is not None:
        m = m * (panel.years[None, :] <= year_max)
    return jnp.asarray(m)


def fit(panel, cfg: Config, wm: WorldModel | None = None, train_year_max: int | None = None,
        seed: int | None = None, verbose: bool = True) -> FitResult:
    wm = WorldModel.build(panel, cfg) if wm is None else wm
    params = wm.init_params(seed=cfg.train.seed if seed is None else seed)
    targets = _targets(panel)
    mask = year_window_mask(panel, train_year_max)

    sched = optax.cosine_decay_schedule(cfg.train.lr, cfg.train.steps, alpha=0.1)
    opt = optax.chain(
        optax.clip_by_global_norm(cfg.train.grad_clip),
        optax.adamw(sched, weight_decay=cfg.train.weight_decay),
    )
    st = opt.init(params)

    def loss_fn(params):
        traj = wm.rollout(params, panel)
        total, terms = loss_terms(traj, targets, mask, cfg)
        total = total + ef_regulariser(params)
        return total, terms

    @jax.jit
    def step(params, st):
        (l, terms), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, st = opt.update(g, st, params)
        params = optax.apply_updates(params, updates)
        return params, st, l, terms

    history = []
    t0 = time.time()
    for i in range(cfg.train.steps + 1):
        params, st, l, terms = step(params, st)
        if i % max(1, cfg.train.steps // 20) == 0 or i == cfg.train.steps:
            rec = {"step": i, "loss": float(l), **{k: float(v) for k, v in terms.items()}}
            history.append(rec)
            if verbose:
                print(f"step {i:5d}  loss={rec['loss']:.5f}  "
                      f"co2={rec['co2']:.5f} dem={rec['demand']:.5f} "
                      f"mix={rec['mix']:.5f}  ({time.time()-t0:.1f}s)")
    return FitResult(params=params, history=history, wm=wm)


def save_params(params, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(jax.tree_util.tree_map(lambda x: np.asarray(x), params), fh)


def load_params(path: str):
    with open(path, "rb") as fh:
        arrs = pickle.load(fh)
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), arrs)
