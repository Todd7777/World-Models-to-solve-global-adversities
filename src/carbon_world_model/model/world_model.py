"""World-model container: wiring the panel, physics, and learned dynamics together.

Leak-free by construction: ``WorldModel.build(panel, cfg, upto_year=Y)`` derives the
emission factors, standardisation statistics and per-country static features from data
up to and including year Y only. Backtests build with Y = cutoff; deployment builds
with the last data year.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..config import Config
from .emissions import estimate_emission_factors
from .dynamics import Norm, rollout, n_features
from .nn import init_mlp


# ----------------------------------------------------------------------------- helpers
def _year_index(panel, year: int) -> int:
    hits = np.where(panel.years == year)[0]
    if len(hits) == 0:
        raise ValueError(f"year {year} not in panel {panel.years[0]}-{panel.years[-1]}")
    return int(hits[0])


def _log_intensity(panel) -> np.ndarray:
    return np.log(panel.primary_energy + 1e-6) - np.log(panel.gdp + 1e-6)


def _upto_mask(panel, upto_idx: int) -> np.ndarray:
    m = panel.mask.astype(bool).copy()
    m[:, upto_idx + 1:] = False
    return m


def compute_norm(panel, upto_idx: int) -> Norm:
    m = _upto_mask(panel, upto_idx)
    lpe = np.log(panel.primary_energy + 1e-6)
    lg = np.log(panel.gdp + 1e-6)
    lp = np.log(panel.population + 1e-6)
    inten = lpe - lg
    growth = np.concatenate([lg[:, 1:], lg[:, -1:]], axis=1) - lg
    mg = m.copy()
    mg[:, upto_idx] = False  # growth at the cutoff would read the next (future) year

    def ms(x, mask):
        v = x[mask]
        return (float(np.mean(v)), float(np.std(v)))

    pair = m[:, 1:] & m[:, :-1]
    dI = np.diff(inten, axis=1)[pair]
    ds = np.diff(panel.shares(), axis=1)[pair]          # [N, F]
    return Norm(
        logD=ms(lpe, m), logGDP=ms(lg, m), growth=ms(growth, mg), logPop=ms(lp, m),
        intensity=ms(inten, m), logDpc=ms(lpe - lp, m), logGDPpc=ms(lg - lp, m),
        dI_scale=float(max(np.std(dI), 1e-3)),
        ds_scale=tuple(float(max(v, 1e-3)) for v in np.std(ds, axis=0)),
    )


STATIC_NAMES = (
    "log_mean_primary_energy", "fossil_share_start", "fossil_share_last",
    "low_carbon_share_start", "log_gdp_per_capita_start", "log_pop_start",
    "mean_log_energy_intensity",
)


def static_features(panel, upto_idx: int) -> np.ndarray:
    """Per-country static covariates from data up to `upto_idx` only, standardised."""
    sl = slice(0, upto_idx + 1)
    pe, gdp, pop = panel.primary_energy[:, sl], panel.gdp[:, sl], panel.population[:, sl]
    sh = panel.shares()[:, sl]
    fossil = sh[:, :, :3].sum(axis=2)
    eps = 1e-6
    X = np.stack([
        np.log(pe.mean(axis=1) + eps),
        fossil[:, 0],
        fossil[:, -1],
        1.0 - fossil[:, 0],
        np.log(gdp[:, 0] / (pop[:, 0] + eps) + eps),
        np.log(pop[:, 0] + eps),
        (np.log(pe + eps) - np.log(gdp + eps)).mean(axis=1),
    ], axis=1)
    return ((X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)).astype(np.float32)


def observed_inits(panel, k: int, momentum: str, rho: float) -> dict:
    """Observed state at index k, with momentum warm-started from observed history."""
    inten = _log_intensity(panel)
    sh = panel.shares()
    C, F = sh.shape[0], sh.shape[2]
    mI, ms_ = np.zeros(C), np.zeros((C, F))
    if momentum == "1yr" and k > 0:
        mI = inten[:, k] - inten[:, k - 1]
        ms_ = sh[:, k] - sh[:, k - 1]
    elif momentum == "ema":
        for j in range(1, k + 1):
            mI = rho * mI + (1 - rho) * (inten[:, j] - inten[:, j - 1])
            ms_ = rho * ms_ + (1 - rho) * (sh[:, j] - sh[:, j - 1])
    return {"I0": inten[:, k], "s0": sh[:, k], "mom_I0": mI, "mom_s0": ms_}


# ----------------------------------------------------------------------------- model
@dataclass
class WorldModel:
    cfg: Config
    ef: np.ndarray        # [F] fixed emission factors
    norm: Norm
    static: np.ndarray    # [C, S]
    n_countries: int
    n_fuels: int
    din: int
    hp: dict
    upto_idx: int
    T_ref: int            # panel length; calendar time feature = index / (T_ref - 1)

    @classmethod
    def build(cls, panel, cfg: Config, upto_year: int | None = None) -> "WorldModel":
        upto_idx = panel.n_years - 1 if upto_year is None else _year_index(panel, upto_year)
        ef = estimate_emission_factors(panel, "nnls", upto_idx=upto_idx).as_full()
        norm = compute_norm(panel, upto_idx)
        static = static_features(panel, upto_idx)
        mc = cfg.model
        hp = {
            "max_share": mc.max_share_change_per_year,
            "max_int": mc.max_intensity_change_per_year,
            "time_features": mc.time_features,
            "level_mode": mc.level_mode,
            "momentum": mc.momentum,
            "ema_decay": mc.ema_decay,
            "damped_trend": mc.damped_trend,
        }
        C, S = static.shape
        din = n_features(panel.n_fuels, S, mc.embed_dim, hp)
        return cls(cfg=cfg, ef=ef, norm=norm, static=static, n_countries=C,
                   n_fuels=panel.n_fuels, din=din, hp=hp, upto_idx=upto_idx,
                   T_ref=panel.n_years)

    def init_params(self, seed: int | None = None) -> dict:
        key = jax.random.PRNGKey(self.cfg.model.seed if seed is None else seed)
        k_embed, k_trunk, k_mix, k_int = jax.random.split(key, 4)
        hidden = self.cfg.model.hidden
        logit = float(np.log(self.cfg.model.phi_init / (1 - self.cfg.model.phi_init)))
        return {
            "embed": 0.1 * jax.random.normal(k_embed, (self.n_countries, self.cfg.model.embed_dim)),
            "trunk": init_mlp(k_trunk, [self.din] + [hidden] * self.cfg.model.depth),
            "mix_head": {"w": 0.01 * jax.random.normal(k_mix, (hidden, self.n_fuels)),
                          "b": jnp.zeros(self.n_fuels)},
            "int_head": {"w": 0.01 * jax.random.normal(k_int, (hidden, 1)), "b": jnp.zeros(1)},
            "log_ef_corr": jnp.zeros(self.n_countries),
            "phi": jnp.full((2,), logit),
        }

    # ---- full-horizon rollout from the first year (used for the long-run fit) ----
    def drivers(self, panel) -> dict:
        return {"gdp": jnp.asarray(panel.gdp), "pop": jnp.asarray(panel.population),
                "process": jnp.asarray(panel.process_co2)}

    def inits(self, panel) -> dict:
        ini = observed_inits(panel, 0, self.hp["momentum"], self.hp["ema_decay"])
        return {k: jnp.asarray(v) for k, v in ini.items()}

    def tfrac(self, idx) -> np.ndarray:
        return np.asarray(idx, dtype=float) / (self.T_ref - 1)

    def rollout(self, params, panel, controls: dict | None = None) -> dict:
        T = panel.n_years
        return rollout(params, jnp.asarray(self.static), self.norm, jnp.asarray(self.ef),
                       self.inits(panel), self.drivers(panel), controls, self.hp,
                       tfrac=jnp.asarray(self.tfrac(np.arange(T))))

    # ---- forecast from the OBSERVED state at a given year ----
    def forecast(self, params, panel, start_year: int, end_year: int | None = None,
                 gdp=None, pop=None, process=None, controls=None,
                 tfrac_cap: float | None = None) -> dict:
        """Roll forward from the observed state in `start_year`.

        Drivers default to the panel's realised GDP/population (a *conditional*
        forecast); pass SSP or other scenario paths [C, H] covering start..end to make
        an *ex-ante* or scenario forecast. Process CO2 defaults to being held at its
        start-year value (no future information). Output index 0 is the start year.
        """
        k = _year_index(panel, start_year)
        if gdp is None:
            end_idx = panel.n_years - 1 if end_year is None else _year_index(panel, end_year)
            gdp = panel.gdp[:, k:end_idx + 1]
            pop = panel.population[:, k:end_idx + 1]
        H = gdp.shape[1]
        if process is None:
            process = np.repeat(panel.process_co2[:, k:k + 1], H, axis=1)
        ini = observed_inits(panel, k, self.hp["momentum"], self.hp["ema_decay"])
        tf = self.tfrac(k + np.arange(H))
        if tfrac_cap is not None:
            tf = np.minimum(tf, tfrac_cap)
        drivers = {"gdp": jnp.asarray(gdp), "pop": jnp.asarray(pop),
                   "process": jnp.asarray(process)}
        return rollout(params, jnp.asarray(self.static), self.norm, jnp.asarray(self.ef),
                       {kk: jnp.asarray(v) for kk, v in ini.items()}, drivers, controls,
                       self.hp, tfrac=jnp.asarray(tf))
