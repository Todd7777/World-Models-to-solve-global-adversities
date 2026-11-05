"""Loss functions for fitting the world model by differentiable simulation."""
from __future__ import annotations

import jax.numpy as jnp


def _masked_logmse(pred, true, mask):
    d = (jnp.log(pred + 1.0) - jnp.log(true + 1.0)) ** 2
    return jnp.sum(mask * d) / (jnp.sum(mask) + 1e-8)


def _masked_mse(pred, true, mask):
    return jnp.sum(mask * (pred - true) ** 2) / (jnp.sum(mask) + 1e-8)


def loss_terms(traj, targets, mask, cfg):
    """Return (scalar_loss, dict_of_terms).

    traj:    model output dict (co2 [C,T], D [C,T], shares [C,T,F], intensity[C,T])
    targets: dict with co2, D, shares (same shapes)
    mask:    [C,T] 1.0 where a year counts toward the loss (observed AND in-window)
    """
    tr = cfg.train
    l_co2 = _masked_logmse(traj["co2"], targets["co2"], mask)
    l_dem = _masked_logmse(traj["D"], targets["D"], mask)
    l_mix = _masked_mse(traj["shares"], targets["shares"], mask[..., None])

    # Smoothness: penalise large curvature in predicted log-intensity (discourages
    # oscillating trajectories not seen in data).
    d2 = traj["intensity"][:, 2:] - 2 * traj["intensity"][:, 1:-1] + traj["intensity"][:, :-2]
    l_smooth = jnp.mean(d2 ** 2)

    total = (tr.w_emissions * l_co2 + tr.w_demand * l_dem
             + tr.w_mix * l_mix + tr.w_smooth * l_smooth)
    terms = {"co2": l_co2, "demand": l_dem, "mix": l_mix, "smooth": l_smooth}
    return total, terms


def ef_regulariser(params, weight: float = 1e-3):
    """Keep the learned emission-factor corrections near 1 (log near 0)."""
    return weight * jnp.mean(params["log_ef_corr"] ** 2)
