"""Learned differentiable dynamics (Module B): the world-model core.

Each trajectory (one country, starting from some year) is described by:

    I_t  = log energy intensity   = log(primary_energy_t / GDP_t)      [scalar]
    s_t  = fuel-mix shares on the simplex                              [F]
    m_t  = momentum: recent *natural* rate of change of (I, s)         [1 + F]

Primary energy is reconstructed mechanistically as  D_t = GDP_t * exp(I_t), and
emissions follow the fixed physics of Module A. GDP and population are *drivers*
(observed in training; realised, SSP or other scenario paths when forecasting). A
shared neural network predicts the bounded annual change in intensity and mix; the
bounds encode capital-stock inertia. Per-country embeddings let one set of shared
dynamics specialise to many countries.

Feature/structure options (``hp``), all switchable for ablation:
  time_features  calendar-time inputs (t, t^2). Extrapolate poorly beyond training.
  level_mode     'absolute'   : log energy, log GDP, log population (size-dependent)
                 'per_capita' : log energy/cap, log GDP/cap (development stage) and a
                                squashed log population; keeps the largest economies
                                in-distribution when they grow past historic maxima.
  momentum       'none' | '1yr' (last natural change) | 'ema' (exponentially smoothed)
  damped_trend   change = phi * momentum + bounded learned correction, with learnable
                 damping phi in (0,1): the damped-trend structure known to be among the
                 most robust multi-year forecasters, nested inside the world model.

Controls enter additively and are zero during fitting; momentum is updated from the
*natural* (uncontrolled) change only. The planner does not use them: it keeps the
network on business as usual and applies plans on top (plan/optimize.py), because the
network's response to controlled states is extrapolation. plan/response.py uses them to
measure that closed loop, which the planner used until 2026-11-13.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .nn import mlp_apply


@dataclass(frozen=True)
class Norm:
    """Standardisation statistics, computed only from data up to the fit cutoff."""
    logD: tuple[float, float]
    logGDP: tuple[float, float]
    growth: tuple[float, float]
    logPop: tuple[float, float]
    intensity: tuple[float, float]
    logDpc: tuple[float, float]
    logGDPpc: tuple[float, float]
    dI_scale: float            # typical |annual change| in log intensity
    ds_scale: tuple[float, ...]  # typical |annual change| per fuel share


def _z(x, stat):
    return (x - stat[0]) / (stat[1] + 1e-8)


def softclip(x, m):
    return m * jnp.tanh(x / m)


def n_features(n_fuels: int, n_static: int, embed_dim: int, hp: dict) -> int:
    n = n_fuels + 5                      # shares + 5 continuous level/growth features
    if hp["momentum"] != "none":
        n += 1 + n_fuels
    if hp["time_features"]:
        n += 2
    return n + n_static + embed_dim


def _features(embed_b, static_b, norm, hp, s_t, I_t, logD, logGDP, growth, logPop, tfrac_t,
              mom_I, mom_s):
    B = s_t.shape[0]
    if hp["level_mode"] == "absolute":
        cont = [_z(logD, norm.logD), _z(logGDP, norm.logGDP), _z(growth, norm.growth),
                _z(logPop, norm.logPop), _z(I_t, norm.intensity)]
    elif hp["level_mode"] == "per_capita":
        cont = [_z(logD - logPop, norm.logDpc), _z(logGDP - logPop, norm.logGDPpc),
                _z(growth, norm.growth), _z(I_t, norm.intensity),
                jnp.tanh(0.5 * _z(logPop, norm.logPop))]
    else:
        raise ValueError(hp["level_mode"])
    parts = [s_t, jnp.stack(cont, axis=1)]
    if hp["momentum"] != "none":
        parts.append((mom_I / norm.dI_scale)[:, None])
        parts.append(mom_s / jnp.asarray(norm.ds_scale)[None, :])
    if hp["time_features"]:
        tt = jnp.broadcast_to(tfrac_t, (B,))
        parts.append(jnp.stack([tt, tt ** 2], axis=1))
    parts += [static_b, embed_b]
    return jnp.concatenate(parts, axis=1)


def rollout(params, static, norm, ef, inits, drivers, controls, hp, tfrac=None,
            country_idx=None):
    """Autoregressive differentiable rollout.

    Shapes: B trajectories (countries, or country x start-year pairs), T steps, F fuels.
      inits:    I0 [B], s0 [B,F], optional mom_I0 [B], mom_s0 [B,F] (observed warm start)
      drivers:  gdp [B,T], pop [B,T], process [B,T]
      controls: None or dict(u_int [B,T], u_deploy [B,T,F])
      tfrac:    calendar time feature, [T] or [B,T]; defaults to t/(T-1)
      country_idx: [B] int; gathers per-country embedding / static / EF correction.
    Returns dict of trajectories, each [B,T] (or [B,T,F]).
    """
    gdp, pop, process = drivers["gdp"], drivers["pop"], drivers["process"]
    B, T = gdp.shape
    F = inits["s0"].shape[1]
    logGDP = jnp.log(gdp + 1e-6)
    logPop = jnp.log(pop + 1e-6)
    growth = jnp.concatenate([logGDP[:, 1:], logGDP[:, -1:]], axis=1) - logGDP

    if country_idx is None:
        embed_b, static_b, efc_b = params["embed"], static, params["log_ef_corr"]
    else:
        embed_b = params["embed"][country_idx]
        static_b = static[country_idx]
        efc_b = params["log_ef_corr"][country_idx]
    ef_eff = ef[None, :] * jnp.exp(efc_b)[:, None]          # [B,F]

    if controls is None:
        u_int_all = jnp.zeros((B, T))
        u_dep_all = jnp.zeros((B, T, F))
    else:
        u_int_all, u_dep_all = controls["u_int"], controls["u_deploy"]

    if tfrac is None:
        tfrac = jnp.arange(T) / max(T - 1, 1)
    tfrac = jnp.asarray(tfrac, dtype=jnp.float32)
    tf_bt = tfrac if tfrac.ndim == 2 else jnp.broadcast_to(tfrac[None, :], (B, T))

    max_share, max_int = hp["max_share"], hp["max_int"]
    rho = hp.get("ema_decay", 0.7)
    mode = hp["momentum"]
    damped = hp.get("damped_trend", False) and mode != "none"
    phi_I = jax.nn.sigmoid(params["phi"][0]) if damped else 0.0
    phi_s = jax.nn.sigmoid(params["phi"][1]) if damped else 0.0

    mom_I0 = inits.get("mom_I0", jnp.zeros(B))
    mom_s0 = inits.get("mom_s0", jnp.zeros((B, F)))

    def step(carry, t):
        I_t, s_t, mom_I, mom_s = carry
        D_t = gdp[:, t] * jnp.exp(I_t)
        logD = jnp.log(D_t + 1e-6)
        x = _features(embed_b, static_b, norm, hp, s_t, I_t, logD, logGDP[:, t],
                      growth[:, t], logPop[:, t], tf_bt[:, t], mom_I, mom_s)
        h = jax.nn.silu(mlp_apply(params["trunk"], x))
        mix_logits = h @ params["mix_head"]["w"] + params["mix_head"]["b"]
        int_raw = (h @ params["int_head"]["w"] + params["int_head"]["b"])[:, 0]

        energy_t = D_t[:, None] * s_t
        co2_t = jnp.sum(ef_eff * energy_t, axis=1) + process[:, t]

        # Natural (uncontrolled) proposed changes.
        if damped:
            dI_nat = softclip(phi_I * mom_I + max_int * jnp.tanh(int_raw), max_int)
            ds_nat = softclip(phi_s * mom_s + max_share * jnp.tanh(mix_logits), max_share)
        else:
            dI_nat = max_int * jnp.tanh(int_raw)
            ds_nat = max_share * jnp.tanh(mix_logits)

        # Actual next state includes controls.
        I_next = I_t + dI_nat - u_int_all[:, t]
        s_raw = jax.nn.relu(s_t + ds_nat + u_dep_all[:, t])
        s_next = s_raw / (jnp.sum(s_raw, axis=1, keepdims=True) + 1e-9)

        # Momentum from the natural step only.
        s_nat_raw = jax.nn.relu(s_t + ds_nat)
        s_nat = s_nat_raw / (jnp.sum(s_nat_raw, axis=1, keepdims=True) + 1e-9)
        nI, ns = dI_nat, s_nat - s_t
        if mode == "1yr":
            mom_I_next, mom_s_next = nI, ns
        elif mode == "ema":
            mom_I_next = rho * mom_I + (1 - rho) * nI
            mom_s_next = rho * mom_s + (1 - rho) * ns
        else:
            mom_I_next, mom_s_next = mom_I, mom_s

        out = {"D": D_t, "shares": s_t, "energy": energy_t, "co2": co2_t, "intensity": I_t}
        return (I_next, s_next, mom_I_next, mom_s_next), out

    init_carry = (inits["I0"], inits["s0"], mom_I0, mom_s0)
    _, outs = jax.lax.scan(step, init_carry, jnp.arange(T))
    return {k: jnp.moveaxis(v, 0, 1) for k, v in outs.items()}
