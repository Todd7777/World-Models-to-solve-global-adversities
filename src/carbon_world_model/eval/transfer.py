"""Spatial-transfer test: does the learned world model generalise to new countries?

Strict leave-countries-out AND leave-future-out protocol:
  1. Build the model from data up to the cutoff only. Train the shared dynamics and
     embeddings on TRAIN countries, years <= cutoff. The shared network never sees the
     held-out countries, nor any year after the cutoff.
  2. Freeze the shared network and the train embeddings. For each HELD-OUT country fit
     only its low-dimensional latent (embedding + emission-factor correction) on its
     own data up to the cutoff.
  3. Forecast the held-out countries from their OBSERVED cutoff state and score against
     truth and a trend baseline.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..config import Config
from ..model.world_model import WorldModel, _year_index
from ..model.dynamics import rollout
from ..train.fit import shooting_batch, shoot_loss, year_window_mask, _targets
from ..train.losses import loss_terms, ef_regulariser
from .baselines import trend_kaya
from . import metrics as M


def _fit(wm, panel, cfg, params, country_w, cut_idx, adapt_idx, steps, lr):
    """Fit with the loss restricted to countries with weight 1; if adapt_idx is given,
    only those countries' embedding / EF-correction rows receive gradient."""
    cut_year = int(panel.years[cut_idx])
    targets = _targets(panel)
    cw = jnp.asarray(country_w)
    mask = year_window_mask(panel, cut_year) * cw[:, None]
    batch = None
    if cfg.train.multi_shoot:
        batch = shooting_batch(wm, panel, cut_idx, cfg.train.shoot_horizon, cfg.train.shoot_stride)
        batch["mask"] = batch["mask"] * jnp.tile(cw, batch["n_starts"])[:, None]
    static, ef = jnp.asarray(wm.static), jnp.asarray(wm.ef)

    gmask = None
    if adapt_idx is not None:
        sel = np.zeros(panel.n_countries)
        sel[adapt_idx] = 1.0
        sel = jnp.asarray(sel)
        gmask = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
        gmask["embed"] = jnp.broadcast_to(sel[:, None], params["embed"].shape)
        gmask["log_ef_corr"] = sel

    def loss_fn(p):
        total = 0.0
        if cfg.train.w_full > 0 or batch is None:
            lf, _ = loss_terms(wm.rollout(p, panel), targets, mask, cfg)
            total = total + (cfg.train.w_full if batch is not None else 1.0) * lf
        if batch is not None:
            trs = rollout(p, static, wm.norm, ef, batch["inits"], batch["drivers"], None,
                          wm.hp, tfrac=batch["tfrac"], country_idx=batch["country_idx"])
            ls, _ = shoot_loss(trs, batch, cfg)
            total = total + cfg.train.w_shoot * ls
        return total + ef_regulariser(p)

    opt = optax.adam(lr)
    st = opt.init(params)

    @jax.jit
    def step(p, st):
        g = jax.grad(loss_fn)(p)
        if gmask is not None:
            g = jax.tree_util.tree_map(lambda gi, mi: gi * mi, g, gmask)
        up, st = opt.update(g, st)
        return optax.apply_updates(p, up), st

    for _ in range(steps):
        params, st = step(params, st)
    return params


def spatial_transfer(panel, cfg: Config, test_every: int = 5, cutoff: int = 2015,
                     seed: int = 0, adapt_steps: int = 2000, verbose: bool = True) -> dict:
    wm = WorldModel.build(panel, cfg, upto_year=cutoff)
    k = _year_index(panel, cutoff)
    C = panel.n_countries
    test_idx = list(range(0, C, test_every))          # spread across the emission-rank order
    train_idx = [i for i in range(C) if i not in test_idx]
    cw_train = np.zeros(C); cw_train[train_idx] = 1.0
    cw_test = np.zeros(C); cw_test[test_idx] = 1.0
    if verbose:
        print(f"[transfer] train {len(train_idx)} countries (<= {cutoff}); hold out "
              f"{[panel.countries[i] for i in test_idx]}")

    params = wm.init_params(seed)
    params = _fit(wm, panel, cfg, params, cw_train, k, None, cfg.train.steps, cfg.train.lr)
    params = _fit(wm, panel, cfg, params, cw_test, k, test_idx, adapt_steps, 5e-3)

    co2p = np.asarray(wm.forecast(params, panel, cutoff)["co2"])     # [C, H+1]
    H = co2p.shape[1] - 1
    true = panel.co2[:, k + 1:k + H + 1]
    hmask = np.zeros_like(true)
    hmask[test_idx] = panel.mask[test_idx, k + 1:k + H + 1]
    tk = trend_kaya(panel, cutoff)[:, k + 1:k + H + 1]
    insample = np.asarray(wm.rollout(params, panel)["co2"])[:, :k + 1]
    return {
        "test_countries": [panel.countries[i] for i in test_idx],
        "holdout_years": [int(y) for y in panel.years[k + 1:k + H + 1]],
        "world_model": {"mape": M.mape(co2p[:, 1:], true, hmask),
                        "rmse_mt": M.rmse(co2p[:, 1:], true, hmask),
                        "r2": M.r2(co2p[:, 1:], true, hmask)},
        "trend_kaya": {"mape": M.mape(tk, true, hmask), "rmse_mt": M.rmse(tk, true, hmask),
                       "r2": M.r2(tk, true, hmask)},
        "adapt_insample_mape": M.mape(insample[test_idx], panel.co2[test_idx, :k + 1],
                                      panel.mask[test_idx, :k + 1]),
    }
