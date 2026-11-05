"""Spatial-transfer test: does the learned world model generalise to new countries?

Protocol (strict leave-countries-out):
  1. Train the shared dynamics + embeddings on TRAIN countries only (all years). The
     shared network never sees the held-out countries' data.
  2. Freeze the shared network and the train embeddings. For each HELD-OUT country,
     fit only its low-dimensional latent (embedding + emission-factor correction) on
     its data up to the cutoff year -- a handful of parameters, a few years of data.
  3. Forecast the held-out countries past the cutoff and score against truth and a
     trend baseline.

If step 2 can make the shared dynamics forecast a country it never trained on, the
model has learned transferable structure about how energy systems evolve -- the
defining property of a world model.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..config import Config
from ..model.world_model import WorldModel
from ..train.losses import loss_terms, ef_regulariser
from .baselines import trend_kaya
from . import metrics as M


def _fit(wm, panel, cfg, params, country_w, year_max, train_only_test_latents, test_idx,
         steps, lr, seed):
    targets = {"co2": jnp.asarray(panel.co2), "D": jnp.asarray(panel.primary_energy),
               "shares": jnp.asarray(panel.shares())}
    base = panel.mask * country_w[:, None]
    if year_max is not None:
        base = base * (panel.years[None, :] <= year_max)
    mask = jnp.asarray(base)

    # Gradient mask: when adapting, only test-country embedding/ef rows may move.
    gmask = None
    if train_only_test_latents:
        sel = np.zeros(panel.n_countries)
        sel[test_idx] = 1.0
        sel = jnp.asarray(sel)
        gmask = {
            "embed": sel[:, None],
            "log_ef_corr": sel,
            "trunk": [{"w": 0.0, "b": 0.0} for _ in params["trunk"]],
            "mix_head": {"w": 0.0, "b": 0.0},
            "int_head": {"w": 0.0, "b": 0.0},
        }

    opt = optax.adam(lr)
    st = opt.init(params)

    def loss_fn(p):
        traj = wm.rollout(p, panel)
        total, _ = loss_terms(traj, targets, mask, cfg)
        return total + ef_regulariser(p)

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
                     seed: int = 0, verbose: bool = True) -> dict:
    wm = WorldModel.build(panel, cfg)  # norm + static standardised over the full panel
    C = panel.n_countries
    test_idx = list(range(0, C, test_every))          # spread across the emission-rank order
    train_idx = [i for i in range(C) if i not in test_idx]
    cw_train = np.zeros(C); cw_train[train_idx] = 1.0
    cw_test = np.zeros(C); cw_test[test_idx] = 1.0

    if verbose:
        print(f"[transfer] train on {len(train_idx)} countries, hold out {len(test_idx)}: "
              f"{[panel.countries[i] for i in test_idx]}")

    params = wm.init_params(seed)
    # Phase 1: shared dynamics on train countries (all years).
    params = _fit(wm, panel, cfg, params, cw_train, None, False, test_idx,
                  steps=cfg.train.steps, lr=cfg.train.lr, seed=seed)
    # Phase 2: adapt only held-out latents on their <=cutoff data.
    params = _fit(wm, panel, cfg, params, cw_test, cutoff, True, test_idx,
                  steps=2500, lr=5e-3, seed=seed)

    tr = wm.rollout(params, panel)
    co2p = np.asarray(tr["co2"])
    tk = trend_kaya(panel, cutoff)

    k = int(np.where(panel.years == cutoff)[0][0])
    hmask = np.zeros_like(panel.mask)
    hmask[test_idx, k + 1:] = panel.mask[test_idx, k + 1:]

    out = {
        "test_countries": [panel.countries[i] for i in test_idx],
        "holdout_years": [int(y) for y in panel.years[k + 1:]],
        "world_model": {
            "mape": M.mape(co2p, panel.co2, hmask),
            "rmse_mt": M.rmse(co2p, panel.co2, hmask),
            "r2": M.r2(co2p, panel.co2, hmask),
        },
        "trend_kaya": {
            "mape": M.mape(tk, panel.co2, hmask),
            "rmse_mt": M.rmse(tk, panel.co2, hmask),
            "r2": M.r2(tk, panel.co2, hmask),
        },
        # In-sample fit on the adapted countries (<=cutoff) shows adaptation worked.
        "adapt_insample_mape": M.mape(
            co2p[test_idx, :k + 1], panel.co2[test_idx, :k + 1], panel.mask[test_idx, :k + 1]
        ),
    }
    return out
