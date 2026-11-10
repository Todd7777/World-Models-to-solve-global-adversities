"""Fit the world model across countries by differentiable simulation (optax AdamW).

Two complementary objectives:
  * full run   -- one free-running rollout per country from the first year, fit over
                  the whole training window (teaches long-run consistency);
  * multiple shooting (optional) -- H-year rollouts started from the OBSERVED state in
                  every year of the training window (teaches the thing a forecaster
                  actually does: start from where the world is and roll forward).
Every array used for fitting is restricted to years <= the training cutoff; the model
itself is built with the same cutoff (WorldModel.build(..., upto_year=cutoff)).
"""
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
from ..model.world_model import WorldModel, observed_inits, _year_index
from ..model.dynamics import rollout
from .losses import loss_terms, ef_regulariser, _masked_logmse, _masked_mse


@dataclass
class FitResult:
    params: dict
    history: list[dict]
    wm: WorldModel


def _targets(panel) -> dict:
    return {"co2": jnp.asarray(panel.co2), "D": jnp.asarray(panel.primary_energy),
            "shares": jnp.asarray(panel.shares())}


def year_window_mask(panel, year_max: int | None, recency_halflife: float | None = None):
    """[C,T] weighted mask = observed AND year <= year_max, optional recency weighting."""
    m = panel.mask.copy()
    anchor = year_max if year_max is not None else int(panel.years[-1])
    if year_max is not None:
        m = m * (panel.years[None, :] <= year_max)
    if recency_halflife is not None and recency_halflife > 0:
        age = (anchor - panel.years).clip(min=0)
        m = m * (0.5 ** (age / recency_halflife))[None, :]
    return jnp.asarray(m)


def shooting_batch(wm: WorldModel, panel, cut_idx: int, H: int, stride: int) -> dict:
    """Batch of H-step rollouts from the observed state at every start year < cut_idx.

    Indices past the cutoff are clamped to the cutoff (and masked), so no data after
    the cutoff is ever read. Batch order is start-major: b = start * C + country.
    """
    C, T = panel.co2.shape
    starts = np.arange(0, cut_idx, stride)
    n = len(starts)
    rel = np.arange(H + 1)
    idx = starts[:, None] + rel[None, :]                    # [n, H+1]
    valid = (idx <= cut_idx).astype(float)
    idx_c = np.minimum(idx, cut_idx)

    def g(a):  # [C,T] -> [n*C, H+1]
        return np.transpose(a[:, idx_c], (1, 0, 2)).reshape(n * C, H + 1)

    sh = panel.shares()
    shares = np.transpose(sh[:, idx_c, :], (1, 0, 2, 3)).reshape(n * C, H + 1, sh.shape[2])
    mask = np.transpose(panel.mask[:, idx_c] * valid[None], (1, 0, 2)).reshape(n * C, H + 1)
    inis = [observed_inits(panel, int(t0), wm.hp["momentum"], wm.hp["ema_decay"]) for t0 in starts]
    ini = {k: np.concatenate([d[k] for d in inis], axis=0) for k in inis[0]}
    tf = np.repeat(wm.tfrac(idx_c)[:, None, :], C, axis=1).reshape(n * C, H + 1)
    return {
        "n_starts": n, "C": C,
        "country_idx": jnp.asarray(np.tile(np.arange(C), n)),
        "drivers": {"gdp": jnp.asarray(g(panel.gdp)), "pop": jnp.asarray(g(panel.population)),
                    "process": jnp.asarray(g(panel.process_co2))},
        "inits": {k: jnp.asarray(v) for k, v in ini.items()},
        "tfrac": jnp.asarray(tf),
        "targets": {"co2": jnp.asarray(g(panel.co2)), "D": jnp.asarray(g(panel.primary_energy)),
                    "shares": jnp.asarray(shares)},
        "mask": jnp.asarray(mask),
    }


def shoot_loss(traj, batch, cfg):
    tr = cfg.train
    m, tg = batch["mask"], batch["targets"]
    l_co2 = _masked_logmse(traj["co2"], tg["co2"], m)
    l_dem = _masked_logmse(traj["D"], tg["D"], m)
    l_mix = _masked_mse(traj["shares"], tg["shares"], m[..., None])
    n, C = batch["n_starts"], batch["C"]
    Hp1 = m.shape[1]
    agg_p = jnp.sum((traj["co2"] * m).reshape(n, C, Hp1), axis=1)
    agg_t = jnp.sum((tg["co2"] * m).reshape(n, C, Hp1), axis=1)
    l_agg = _masked_logmse(agg_p, agg_t, (jnp.sum(m.reshape(n, C, Hp1), axis=1) > 0).astype(m.dtype))
    w = m * jnp.sqrt(tg["co2"] / (jnp.mean(tg["co2"]) + 1e-6))
    l_level = jnp.sum(w * ((traj["co2"] - tg["co2"]) / (tg["co2"] + 1.0)) ** 2) / (jnp.sum(w) + 1e-8)
    total = (tr.w_emissions * l_co2 + tr.w_demand * l_dem + tr.w_mix * l_mix
             + tr.w_aggregate * l_agg + tr.w_level * l_level)
    return total, {"s_co2": l_co2, "s_demand": l_dem, "s_mix": l_mix}


def fit(panel, cfg: Config, wm: WorldModel | None = None, train_year_max: int | None = None,
        seed: int | None = None, verbose: bool = True) -> FitResult:
    wm = WorldModel.build(panel, cfg, upto_year=train_year_max) if wm is None else wm
    params = wm.init_params(seed=cfg.train.seed if seed is None else seed)
    targets = _targets(panel)
    mask = year_window_mask(panel, train_year_max, cfg.train.recency_halflife)
    cut_idx = panel.n_years - 1 if train_year_max is None else _year_index(panel, train_year_max)
    batch = (shooting_batch(wm, panel, cut_idx, cfg.train.shoot_horizon, cfg.train.shoot_stride)
             if cfg.train.multi_shoot else None)
    static, ef = jnp.asarray(wm.static), jnp.asarray(wm.ef)

    sched = optax.cosine_decay_schedule(cfg.train.lr, cfg.train.steps, alpha=0.1)
    opt = optax.chain(optax.clip_by_global_norm(cfg.train.grad_clip),
                      optax.adamw(sched, weight_decay=cfg.train.weight_decay))
    st = opt.init(params)

    def loss_fn(p):
        terms = {}
        total = 0.0
        if cfg.train.w_full > 0 or batch is None:
            traj = wm.rollout(p, panel)
            lf, tf_ = loss_terms(traj, targets, mask, cfg)
            total = total + (cfg.train.w_full if batch is not None else 1.0) * lf
            terms.update(tf_)
        if batch is not None:
            trs = rollout(p, static, wm.norm, ef, batch["inits"], batch["drivers"], None,
                          wm.hp, tfrac=batch["tfrac"], country_idx=batch["country_idx"])
            ls, ts_ = shoot_loss(trs, batch, cfg)
            total = total + cfg.train.w_shoot * ls
            terms.update(ts_)
        return total + ef_regulariser(p), terms

    @jax.jit
    def step(p, st):
        (l, terms), g = jax.value_and_grad(loss_fn, has_aux=True)(p)
        updates, st = opt.update(g, st, p)
        return optax.apply_updates(p, updates), st, l, terms

    history = []
    t0 = time.time()
    for i in range(cfg.train.steps + 1):
        params, st, l, terms = step(params, st)
        if i % max(1, cfg.train.steps // 10) == 0 or i == cfg.train.steps:
            rec = {"step": i, "loss": float(l), **{k: float(v) for k, v in terms.items()}}
            history.append(rec)
            if verbose:
                body = " ".join(f"{k}={v:.5f}" for k, v in rec.items() if k not in ("step",))
                print(f"step {i:5d} {body} ({time.time()-t0:.1f}s)")
    return FitResult(params=params, history=history, wm=wm)


def save_params(params, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(jax.tree_util.tree_map(lambda x: np.asarray(x), params), fh)


def load_params(path: str):
    with open(path, "rb") as fh:
        arrs = pickle.load(fh)
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), arrs)


# ---------------------------------------------------------------------------
# Checkpoints that remember which configuration produced them
# ---------------------------------------------------------------------------
# A checkpoint is only meaningful together with the model structure it was trained
# for (time features, level mode, momentum, damped trend). The name of the registered
# configuration (eval/configs.py) is stored next to the parameters, and every consumer
# rebuilds the model from that name instead of assuming the default configuration.
DEFAULT_CHECKPOINT = "results/model/params_full.pkl"


def checkpoint_meta_path(path: str) -> str:
    return os.path.splitext(path)[0] + ".json"


def save_checkpoint(params, path: str, config_name: str, extra: dict | None = None) -> None:
    import json
    save_params(params, path)
    with open(checkpoint_meta_path(path), "w") as fh:
        json.dump({"config": config_name, **(extra or {})}, fh, indent=2)


def checkpoint_config_name(path: str = DEFAULT_CHECKPOINT) -> str:
    """Configuration name recorded with a checkpoint ('orig' for legacy checkpoints)."""
    import json
    meta = checkpoint_meta_path(path)
    if not os.path.exists(meta):
        return "orig"
    with open(meta) as fh:
        return json.load(fh)["config"]


def load_trained(panel, path: str = DEFAULT_CHECKPOINT):
    """(config, world model, parameters) for a checkpoint, built with ITS configuration."""
    from ..eval.configs import make_config
    cfg = make_config(checkpoint_config_name(path))
    return cfg, WorldModel.build(panel, cfg), load_params(path)
