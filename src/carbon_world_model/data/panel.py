"""Build a clean, aligned multi-country panel of energy and emissions.

The panel is the single tensor object the rest of the model consumes. All arrays
share the country axis (C) and year axis (T); fuel-resolved arrays add a fuel axis
(F). Units: energy in TWh (primary-energy, substitution method), CO2 in million
tonnes (Mt), GDP in constant international dollars, population in persons.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os

import numpy as np
import pandas as pd

from ..config import (
    Config,
    FUELS,
    FOSSIL_FUELS,
    CONSUMPTION_COL,
    CO2_SOURCE_COL,
    PROCESS_CO2_COLS,
)
from .load import load_raw, select_countries


@dataclass
class Panel:
    countries: list[str]
    years: np.ndarray          # [T] int
    fuels: tuple[str, ...]     # [F]
    energy: np.ndarray         # [C, T, F]  consumption by fuel (TWh), gap-filled, >=0
    primary_energy: np.ndarray # [C, T]     sum over fuels (TWh)
    co2: np.ndarray            # [C, T]     reported production CO2 (Mt)
    co2_by_source: np.ndarray  # [C, T, 3]  coal/oil/gas CO2 (Mt)
    process_co2: np.ndarray    # [C, T]     cement+flaring+other industry CO2 (Mt)
    gdp: np.ndarray            # [C, T]     constant intl $
    population: np.ndarray     # [C, T]
    mask: np.ndarray           # [C, T]     1.0 where co2 & energy originally observed
    static: np.ndarray         # [C, S]     per-country static features (standardised)
    static_names: list[str]

    # ---- derived convenience ----
    @property
    def n_countries(self) -> int:
        return len(self.countries)

    @property
    def n_years(self) -> int:
        return len(self.years)

    @property
    def n_fuels(self) -> int:
        return len(self.fuels)

    def country_index(self, name: str) -> int:
        return self.countries.index(name)

    def shares(self) -> np.ndarray:
        """Fuel shares of primary energy, [C, T, F], rows summing to 1."""
        pe = self.primary_energy[..., None]
        return np.divide(self.energy, pe, out=np.zeros_like(self.energy), where=pe > 0)

    def save(self, cfg: Config) -> None:
        os.makedirs(os.path.dirname(cfg.data.processed_npz), exist_ok=True)
        np.savez_compressed(
            cfg.data.processed_npz,
            years=self.years,
            energy=self.energy,
            primary_energy=self.primary_energy,
            co2=self.co2,
            co2_by_source=self.co2_by_source,
            process_co2=self.process_co2,
            gdp=self.gdp,
            population=self.population,
            mask=self.mask,
            static=self.static,
        )
        meta = {
            "countries": self.countries,
            "fuels": list(self.fuels),
            "static_names": self.static_names,
            "year_start": int(self.years[0]),
            "year_end": int(self.years[-1]),
        }
        with open(cfg.data.meta_json, "w") as fh:
            json.dump(meta, fh, indent=2)

    @classmethod
    def load(cls, cfg: Config) -> "Panel":
        d = np.load(cfg.data.processed_npz, allow_pickle=False)
        with open(cfg.data.meta_json) as fh:
            meta = json.load(fh)
        return cls(
            countries=list(meta["countries"]),
            years=d["years"],
            fuels=tuple(meta["fuels"]),
            energy=d["energy"],
            primary_energy=d["primary_energy"],
            co2=d["co2"],
            co2_by_source=d["co2_by_source"],
            process_co2=d["process_co2"],
            gdp=d["gdp"],
            population=d["population"],
            mask=d["mask"],
            static=d["static"],
            static_names=list(meta["static_names"]),
        )


def _series_on_grid(sub: pd.DataFrame, col: str, years: np.ndarray) -> np.ndarray:
    """Align one country's column to the fixed year grid; return raw values (NaN gaps)."""
    s = sub.set_index("year")[col].reindex(years)
    return s.to_numpy(dtype=float)


def _fill_gaps(x: np.ndarray, clip_nonneg: bool = True) -> np.ndarray:
    """Linear-interpolate internal NaNs, edge-fill ends. 1-D array over years."""
    x = x.astype(float).copy()
    idx = np.arange(len(x))
    good = np.isfinite(x)
    if good.sum() == 0:
        return np.zeros_like(x)
    x = np.interp(idx, idx[good], x[good])  # interp also edge-fills with nearest
    if clip_nonneg:
        x = np.clip(x, 0.0, None)
    return x


def build_panel(cfg: Config, verbose: bool = True) -> Panel:
    co2_df, en_df = load_raw(cfg)
    countries = select_countries(en_df, cfg)
    years = np.arange(cfg.data.year_start, cfg.data.year_end + 1)
    C, T, F = len(countries), len(years), len(FUELS)

    energy = np.zeros((C, T, F))
    co2 = np.full((C, T), np.nan)
    co2_src = np.zeros((C, T, len(FOSSIL_FUELS)))
    process = np.zeros((C, T))
    gdp = np.full((C, T), np.nan)
    pop = np.full((C, T), np.nan)
    mask = np.zeros((C, T))

    en_by_c = {k: v for k, v in en_df.groupby("country")}
    co2_by_c = {k: v for k, v in co2_df.groupby("country")}

    for ci, country in enumerate(countries):
        e = en_by_c[country]
        c = co2_by_c.get(country)
        # Energy mix (observed-ness before fill -> mask on co2+pe below)
        for fi, fuel in enumerate(FUELS):
            raw = _series_on_grid(e, CONSUMPTION_COL[fuel], years)
            energy[ci, :, fi] = _fill_gaps(raw, clip_nonneg=True)
        pe_raw = _series_on_grid(e, "primary_energy_consumption", years)
        gdp[ci] = _fill_gaps(_series_on_grid(e, "gdp", years), clip_nonneg=True)
        pop[ci] = _fill_gaps(_series_on_grid(e, "population", years), clip_nonneg=True)

        if c is not None:
            co2_raw = _series_on_grid(c, "co2", years)
            co2[ci] = _fill_gaps(co2_raw, clip_nonneg=True)
            for si, fuel in enumerate(FOSSIL_FUELS):
                co2_src[ci, :, si] = _fill_gaps(
                    _series_on_grid(c, CO2_SOURCE_COL[fuel], years), clip_nonneg=True
                )
            proc = np.zeros(T)
            for pcol in PROCESS_CO2_COLS:
                if pcol in c.columns:
                    proc = proc + _fill_gaps(_series_on_grid(c, pcol, years), clip_nonneg=True)
            process[ci] = proc
            # Original observed-ness: both reported co2 and primary energy present.
            mask[ci] = (np.isfinite(co2_raw) & np.isfinite(pe_raw)).astype(float)
        else:
            mask[ci] = 0.0

    primary_energy = energy.sum(axis=2)  # internal consistency: PE == sum of fuels

    static, static_names = _build_static_features(
        energy, primary_energy, gdp, pop, co2
    )

    panel = Panel(
        countries=countries,
        years=years,
        fuels=FUELS,
        energy=energy,
        primary_energy=primary_energy,
        co2=co2,
        co2_by_source=co2_src,
        process_co2=process,
        gdp=gdp,
        population=pop,
        mask=mask,
        static=static,
        static_names=static_names,
    )
    if verbose:
        _report(panel)
    return panel


def _build_static_features(energy, primary_energy, gdp, pop, co2):
    """Per-country standardised static covariates used by the dynamics model."""
    C, T, F = energy.shape
    shares = np.divide(
        energy, primary_energy[..., None],
        out=np.zeros_like(energy), where=primary_energy[..., None] > 0,
    )
    fossil = shares[:, :, :3].sum(axis=2)  # coal/oil/gas are first three
    low_carbon = 1.0 - fossil
    eps = 1e-6
    feats = {
        "log_mean_primary_energy": np.log(primary_energy.mean(axis=1) + eps),
        "fossil_share_start": fossil[:, 0],
        "fossil_share_end": fossil[:, -1],
        "low_carbon_share_start": low_carbon[:, 0],
        "log_gdp_per_capita_start": np.log(gdp[:, 0] / (pop[:, 0] + eps) + eps),
        "log_pop_start": np.log(pop[:, 0] + eps),
        "mean_energy_intensity": (primary_energy / (gdp + eps)).mean(axis=1),
    }
    names = list(feats.keys())
    X = np.stack([feats[n] for n in names], axis=1)  # [C, S]
    # Standardise each feature across countries.
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    Xs = (X - mu) / sd
    return Xs, names


def _report(p: Panel) -> None:
    tot22 = np.nansum(p.co2[:, -1])
    print(f"[panel] countries={p.n_countries} years={p.years[0]}-{p.years[-1]} fuels={p.n_fuels}")
    print(f"[panel] energy {p.energy.shape}  co2 {p.co2.shape}  mask mean={p.mask.mean():.3f}")
    print(f"[panel] total CO2 {p.years[-1]} across panel = {tot22:,.0f} Mt")
    sh = p.shares()
    print(f"[panel] mean fossil share {p.years[0]}={sh[:,0,:3].sum(1).mean():.3f} "
          f"{p.years[-1]}={sh[:,-1,:3].sum(1).mean():.3f}")
