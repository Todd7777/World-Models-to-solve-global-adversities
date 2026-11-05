"""Learned differentiable dynamics (Module B): the world-model core.

Each country's energy system is described by two state variables:

    I_t  = log energy intensity   = log(primary_energy_t / GDP_t)      [scalar]
    s_t  = fuel-mix shares on the simplex                              [F]

Primary energy is reconstructed mechanistically as  D_t = GDP_t * exp(I_t), and
emissions follow the fixed physics of Module A. GDP and population are *drivers*
(observed in training, scenario inputs in planning). A shared neural network predicts
the bounded annual change in intensity and mix; inertia bounds encode that energy
systems turn over slowly (capital stock lifetimes). Per-country embeddings let one set
of shared dynamics specialise to 60 different countries -- the "world model" that
transfers structure across places.

Controls (for the planner) enter additively and are zero during fitting, so the fitted
model reproduces observed behaviour and the planner perturbs it causally:
    u_int    extra intensity improvement  (energy efficiency effort)
    u_deploy extra per-fuel share push     (clean-energy deployment effort)
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .nn import mlp_apply


@dataclass(frozen=True)
class Norm:
    """Standardisation stats for continuous network inputs (plain floats/arrays)."""
    logD: tuple[float, float]
    logGDP: tuple[float, float]
    growth: tuple[float, float]
    logPop: tuple[float, float]
    intensity: tuple[float, float]


def _std(x, stat):
    return (x - stat[0]) / (stat[1] + 1e-8)


def _features(params, static, norm, s_t, I_t, logD_t, logGDP_t, growth_t, logPop_t, tfrac):
    """Build the per-country network input matrix [C, din]."""
    C = s_t.shape[0]
    cont = jnp.stack(
        [
            _std(logD_t, norm.logD),
            _std(logGDP_t, norm.logGDP),
            _std(growth_t, norm.growth),
            _std(logPop_t, norm.logPop),
            _std(I_t, norm.intensity),
        ],
        axis=1,
    )  # [C, 5]
    time = jnp.stack([tfrac * jnp.ones(C), (tfrac ** 2) * jnp.ones(C)], axis=1)  # [C,2]
    x = jnp.concatenate([s_t, cont, time, static, params["embed"]], axis=1)
    return x


def effective_ef(params, ef):
    """Per-country emission factors with a learned multiplicative correction (fossil)."""
    # log_ef_corr: [C]; applies to all fuels but EF is 0 for low-carbon so only fossil moves.
    return ef[None, :] * jnp.exp(params["log_ef_corr"])[:, None]  # [C, F]


def rollout(params, static, norm, ef, inits, drivers, controls, hp):
    """Autoregressive differentiable rollout over the whole horizon.

    Shapes: C countries, T years, F fuels.
      inits:    dict(I0 [C], s0 [C,F])
      drivers:  dict(gdp [C,T], pop [C,T], process [C,T])
      controls: dict(u_int [C,T], u_deploy [C,T,F])
      hp:       dict(max_share, max_int)
    Returns dict of trajectories, each [C,T] (or [C,T,F] for mix/energy).
    """
    gdp, pop, process = drivers["gdp"], drivers["pop"], drivers["process"]
    C, T = gdp.shape
    logGDP = jnp.log(gdp + 1e-6)
    logPop = jnp.log(pop + 1e-6)
    gdp_next = jnp.concatenate([logGDP[:, 1:], logGDP[:, -1:]], axis=1)
    growth = gdp_next - logGDP  # [C,T] log-growth to next year
    ef_eff = effective_ef(params, ef)  # [C,F]
    max_share, max_int = hp["max_share"], hp["max_int"]

    def step(carry, t):
        I_t, s_t = carry
        logGDP_t = logGDP[:, t]
        D_t = gdp[:, t] * jnp.exp(I_t)          # [C] TWh
        logD_t = jnp.log(D_t + 1e-6)
        tfrac = t / (T - 1)

        x = _features(params, static, norm, s_t, I_t, logD_t, logGDP_t,
                      growth[:, t], logPop[:, t], tfrac)
        h = jax.nn.silu(mlp_apply(params["trunk"], x))
        mix_logits = h @ params["mix_head"]["w"] + params["mix_head"]["b"]   # [C,F]
        int_raw = h @ params["int_head"]["w"] + params["int_head"]["b"]      # [C,1]

        # Observables at time t (from current state).
        energy_t = D_t[:, None] * s_t                                        # [C,F] TWh
        co2_t = jnp.sum(ef_eff * energy_t, axis=1) + process[:, t]           # [C] Mt

        # Advance state to t+1.
        dI = max_int * jnp.tanh(int_raw[:, 0]) - controls["u_int"][:, t]
        I_next = I_t + dI
        d_s = max_share * jnp.tanh(mix_logits) + controls["u_deploy"][:, t]  # [C,F]
        s_raw = jax.nn.relu(s_t + d_s)
        s_next = s_raw / (jnp.sum(s_raw, axis=1, keepdims=True) + 1e-9)

        out = {
            "D": D_t,
            "shares": s_t,
            "energy": energy_t,
            "co2": co2_t,
            "intensity": I_t,
        }
        return (I_next, s_next), out

    (_, _), outs = jax.lax.scan(step, (inits["I0"], inits["s0"]), jnp.arange(T))
    # scan stacks along axis 0 (time) -> move to [C, T, ...].
    traj = {}
    for k, v in outs.items():
        traj[k] = jnp.moveaxis(v, 0, 1)
    return traj
