"""Loss functions for fitting the world model by differentiable simulation."""
from __future__ import annotations

import jax.numpy as jnp


def _masked_logmse(pred, true, mask):
    d = (jnp.log(pred + 1.0) - jnp.log(true + 1.0)) ** 2
    return jnp.sum(mask * d) / (jnp.sum(mask) + 1e-8)


def _masked_mse(pred, true, mask):
    return jnp.sum(mask * (pred - true) ** 2) / (jnp.sum(mask) + 1e-8)


def loss_terms(traj, targets, mask, cfg, window=None):
    """Return (scalar_loss, dict_of_terms).

    traj:    model output dict (co2 [C,T], D [C,T], shares [C,T,F], intensity[C,T])
    targets: dict with co2, D, shares (same shapes)
    mask:    [C,T] 1.0 where a year counts toward the loss (observed AND in-window)
    window:  None for a fit on all the data. Otherwise a [C,T] 0/1 array marking the fit
             window: the countries being fitted, in years up to the cutoff. The free
             rollout behind `traj` runs past the cutoff on realised drivers, so the two
             terms that `mask` does not confine (the curvature penalty and the level
             term's normaliser) are confined to the window. No value outside it can then
             reach the loss. With None both are computed over everything, as before.
    """
    tr = cfg.train
    l_co2 = _masked_logmse(traj["co2"], targets["co2"], mask)
    l_dem = _masked_logmse(traj["D"], targets["D"], mask)
    l_mix = _masked_mse(traj["shares"], targets["shares"], mask[..., None])

    # Smoothness: penalise large curvature in predicted log-intensity (discourages
    # oscillating trajectories not seen in data).
    d2 = traj["intensity"][:, 2:] - 2 * traj["intensity"][:, 1:-1] + traj["intensity"][:, :-2]
    if window is None:
        l_smooth = jnp.mean(d2 ** 2)
        co2_scale = jnp.mean(targets["co2"])
    else:
        w3 = window[:, 2:] * window[:, 1:-1] * window[:, :-2]   # whole stencil in the window
        l_smooth = jnp.sum(w3 * d2 ** 2) / (jnp.sum(w3) + 1e-8)
        co2_scale = jnp.sum(window * targets["co2"]) / (jnp.sum(window) + 1e-8)

    # Aggregate: fit the global-sum emissions path (the policy-relevant quantity, and
    # the one scale-free per-country losses neglect). Year counts if any country is in
    # the mask that year.
    year_mask = (jnp.sum(mask, axis=0) > 0).astype(mask.dtype)
    agg_pred = jnp.sum(traj["co2"] * mask, axis=0)
    agg_true = jnp.sum(targets["co2"] * mask, axis=0)
    l_agg = _masked_logmse(agg_pred, agg_true, year_mask)

    # Emissions-weighted level term: large emitters must be right in absolute terms, so
    # they do not drift over long rollouts. Weight ~ sqrt(share of panel emissions).
    w = mask * jnp.sqrt(targets["co2"] / (co2_scale + 1e-6))
    rel = ((traj["co2"] - targets["co2"]) / (targets["co2"] + 1.0)) ** 2
    l_level = jnp.sum(w * rel) / (jnp.sum(w) + 1e-8)

    total = (tr.w_emissions * l_co2 + tr.w_demand * l_dem + tr.w_mix * l_mix
             + tr.w_smooth * l_smooth + tr.w_aggregate * l_agg + tr.w_level * l_level)
    terms = {"co2": l_co2, "demand": l_dem, "mix": l_mix, "smooth": l_smooth,
             "agg": l_agg, "level": l_level}
    return total, terms


def ef_regulariser(params, weight: float = 1e-3):
    """Keep the learned emission-factor corrections near 1 (log near 0)."""
    return weight * jnp.mean(params["log_ef_corr"] ** 2)
