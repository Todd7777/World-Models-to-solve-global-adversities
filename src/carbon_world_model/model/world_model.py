"""World-model container: wiring the panel, physics, and learned dynamics together."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..config import Config, FUELS
from .emissions import estimate_emission_factors, EmissionFactors
from .dynamics import Norm, rollout
from .nn import init_mlp


def _norm_from_panel(panel) -> Norm:
    m = panel.mask.astype(bool)
    pe, gdp, pop = panel.primary_energy, panel.gdp, panel.population
    logD = np.log(pe[m] + 1e-6)
    logGDP = np.log(gdp[m] + 1e-6)
    logPop = np.log(pop[m] + 1e-6)
    intensity = (np.log(pe + 1e-6) - np.log(gdp + 1e-6))[m]
    glog = np.log(gdp + 1e-6)
    growth = (np.concatenate([glog[:, 1:], glog[:, -1:]], axis=1) - glog)[m]

    def ms(x):
        return (float(np.mean(x)), float(np.std(x)))

    return Norm(logD=ms(logD), logGDP=ms(logGDP), growth=ms(growth),
               logPop=ms(logPop), intensity=ms(intensity))


@dataclass
class WorldModel:
    cfg: Config
    ef: np.ndarray              # [F] fixed emission factors
    norm: Norm
    static: np.ndarray          # [C, S]
    n_countries: int
    n_fuels: int
    din: int
    hp: dict

    @classmethod
    def build(cls, panel, cfg: Config) -> "WorldModel":
        ef = estimate_emission_factors(panel, method="nnls").as_full()
        norm = _norm_from_panel(panel)
        C, S = panel.static.shape
        F = panel.n_fuels
        din = F + 5 + 2 + S + cfg.model.embed_dim
        hp = {
            "max_share": cfg.model.max_share_change_per_year,
            "max_int": cfg.model.max_intensity_change_per_year,
        }
        return cls(cfg=cfg, ef=ef, norm=norm, static=panel.static.astype(np.float32),
                   n_countries=C, n_fuels=F, din=din, hp=hp)

    def init_params(self, seed: int | None = None) -> dict:
        key = jax.random.PRNGKey(self.cfg.model.seed if seed is None else seed)
        k_embed, k_trunk, k_mix, k_int = jax.random.split(key, 4)
        hidden = self.cfg.model.hidden
        trunk_sizes = [self.din] + [hidden] * self.cfg.model.depth
        params = {
            "embed": 0.1 * jax.random.normal(k_embed, (self.n_countries, self.cfg.model.embed_dim)),
            "trunk": init_mlp(k_trunk, trunk_sizes),
            "mix_head": {"w": 0.01 * jax.random.normal(k_mix, (hidden, self.n_fuels)),
                          "b": jnp.zeros(self.n_fuels)},
            "int_head": {"w": 0.01 * jax.random.normal(k_int, (hidden, 1)),
                          "b": jnp.zeros(1)},
            "log_ef_corr": jnp.zeros(self.n_countries),
        }
        return params

    # ---- driver / init construction from a panel ----
    def drivers(self, panel) -> dict:
        return {
            "gdp": jnp.asarray(panel.gdp),
            "pop": jnp.asarray(panel.population),
            "process": jnp.asarray(panel.process_co2),
        }

    def inits(self, panel) -> dict:
        pe0 = panel.primary_energy[:, 0]
        gdp0 = panel.gdp[:, 0]
        # log intensity as a difference of logs so that gdp0 * exp(I0) == pe0 exactly
        # (adding eps *inside* the ratio would inject a gdp0-scaled error).
        I0 = np.log(pe0 + 1e-6) - np.log(gdp0 + 1e-6)
        s0 = panel.shares()[:, 0, :]
        return {"I0": jnp.asarray(I0), "s0": jnp.asarray(s0)}

    def zero_controls(self, C: int, T: int) -> dict:
        return {"u_int": jnp.zeros((C, T)), "u_deploy": jnp.zeros((C, T, self.n_fuels))}

    def rollout(self, params, panel, controls: dict | None = None, inits: dict | None = None,
                drivers: dict | None = None) -> dict:
        drivers = self.drivers(panel) if drivers is None else drivers
        inits = self.inits(panel) if inits is None else inits
        C, T = drivers["gdp"].shape
        controls = self.zero_controls(C, T) if controls is None else controls
        return rollout(params, jnp.asarray(self.static), self.norm, jnp.asarray(self.ef),
                        inits, drivers, controls, self.hp)
