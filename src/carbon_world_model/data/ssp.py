"""Shared Socioeconomic Pathway (SSP) drivers: GDP and population for SSP1-5.

Two vintages of the IIASA SSP database are supported, both as shipped in GCAM's
open-source data system (JGCRI/gcamdata and JGCRI/gcam-core):

  * ``v9``   -- SSP database v9, downloaded February 2014 (GDP: OECD Env-Growth,
               GDP|PPP in billion US$2005; population: IIASA-WiC POP). Its projections
               predate the 2016-2022 holdout, so it supports an honest *ex-ante*
               forecast test ("what could a forecaster have known in 2015?").
  * ``v3.2`` -- SSP database v3.2 beta, June 2025 (GDP: OECD ENV-Growth 2025, GDP|PPP
               in billion USD_2017; population: IIASA-WiC POP 2025). The current
               release; used for forward scenarios (2022-2100).

SSP series come in 5-year steps; we interpolate log-linearly to annual values (the
natural choice for exponentially growing quantities) and *harmonise* to the panel by
the ratio method standard in integrated assessment: driver(t) = observed(base) *
SSP(t) / SSP(base). Levels therefore stay consistent with the data the world model
was trained on, while growth follows the SSP. Units cancel in the ratio.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os

import numpy as np
import pandas as pd

SSPS: tuple[str, ...] = ("SSP1", "SSP2", "SSP3", "SSP4", "SSP5")

V9_CSV = "data/raw/ssp_v9/SSP_database_v9.csv"
V32_CSV = "data/raw/ssp_2025/SSP_database_2025.csv.gz"
V32_REGMAP = "data/raw/ssp_2025/iso_SSP_regID.csv"
PROCESSED = "data/processed/ssp_series.npz"
PROCESSED_META = "data/processed/ssp_series_meta.json"

# Annual grid on which SSP series are stored (covers both vintages' projection spans).
GRID = np.arange(2005, 2101)

# Legacy ISO3 codes still used by some GCAM mapping files -> current ISO 3166-1 alpha-3.
LEGACY_ISO = {"ROM": "ROU", "ZAR": "COD", "TMP": "TLS", "YUG": "SRB"}


def _year_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).isdigit()]


def loglinear_interp(years_src, vals_src, years_tgt) -> np.ndarray:
    """Log-linear interpolation of a positive series; NaN outside the source span."""
    years_src = np.asarray(years_src, float)
    vals_src = np.asarray(vals_src, float)
    ok = np.isfinite(vals_src) & (vals_src > 0)
    out = np.full(len(years_tgt), np.nan)
    if ok.sum() < 2:
        return out
    ys, lv = years_src[ok], np.log(vals_src[ok])
    yt = np.asarray(years_tgt, float)
    inside = (yt >= ys.min()) & (yt <= ys.max())
    out[inside] = np.exp(np.interp(yt[inside], ys, lv))
    return out


def _scenario_root(s: str) -> str:
    """'SSP2_v9_130325' -> 'SSP2'; 'SSP4d_v9_130115' -> 'SSP4'."""
    return s[:4]


def _series_table(rows: pd.DataFrame, region_col: str, scen_col: str) -> dict:
    """{(scenario, region): (years[int], values[float])} for SSP1-5 rows."""
    yc = _year_cols(rows)
    years = np.array([int(c) for c in yc])
    out = {}
    for _, r in rows.iterrows():
        scen = _scenario_root(str(r[scen_col]))
        if scen not in SSPS:
            continue
        out[(scen, str(r[region_col]))] = (years, r[yc].to_numpy(dtype=float))
    return out


def load_v9(path: str = V9_CSV) -> tuple[dict, dict, dict]:
    """Return (gdp, pop, pop_fallback) tables keyed by (scenario, ISO3)."""
    d = pd.read_csv(path, comment="#", low_memory=False)
    gdp = d[(d.MODEL == "OECD Env-Growth") & (d.VARIABLE == "GDP|PPP")]
    pop = d[(d.MODEL == "IIASA-WiC POP") & (d.VARIABLE == "Population")]
    pop_fb = d[(d.MODEL == "OECD Env-Growth") & (d.VARIABLE == "Population")]
    return (_series_table(gdp, "REGION", "SCENARIO"),
            _series_table(pop, "REGION", "SCENARIO"),
            _series_table(pop_fb, "REGION", "SCENARIO"))


def load_v32(path: str = V32_CSV, regmap_path: str = V32_REGMAP) -> tuple[dict, dict, dict]:
    """Return (gdp, pop, empty_fallback) keyed by (scenario, ISO3).

    v3.2 stores history in a 'Historical Reference' scenario and SSP projections from
    2025; we splice them so each SSP series runs continuously from 2005.
    """
    d = pd.read_csv(path, comment="#", low_memory=False)
    reg = pd.read_csv(regmap_path, comment="#")
    name_to_iso = {str(n): LEGACY_ISO.get(str(i).upper(), str(i).upper())
                   for i, n in zip(reg["iso"], reg["ssp_country_name"])}
    yc = _year_cols(d)
    years = np.array([int(c) for c in yc])

    def table(rows):
        out = {}
        hist = {str(r.Region): r[yc].to_numpy(dtype=float)
                for _, r in rows[rows.Scenario == "Historical Reference"].iterrows()}
        for _, r in rows[rows.Scenario.isin(SSPS)].iterrows():
            iso = name_to_iso.get(str(r.Region))
            if iso is None:
                continue
            v = r[yc].to_numpy(dtype=float)
            h = hist.get(str(r.Region))
            if h is not None:
                v = np.where(np.isfinite(v), v, h)  # splice history before 2025
            out[(str(r.Scenario), iso)] = (years, v)
        return out

    gdp = d[(d.Model == "OECD ENV-Growth 2025") & (d.Variable == "GDP|PPP")
            & (d.Unit == "billion USD_2017/yr")]
    pop = d[(d.Model == "IIASA-WiC POP 2025") & (d.Variable == "Population")]
    return table(gdp), table(pop), {}


@dataclass
class SSPSeries:
    """Annual SSP series on GRID for the panel's countries.

    gdp, pop: [S, C, Y] (NaN where unavailable); avail: [S, C] bool;
    source: per (scenario, country) provenance string for the audit trail.
    """
    vintage: str
    years: np.ndarray
    gdp: np.ndarray
    pop: np.ndarray
    avail: np.ndarray
    source: list[list[str]]


def build_series(iso_codes: list[str], vintage: str) -> SSPSeries:
    if vintage == "v9":
        gdp_t, pop_t, pop_fb = load_v9()
    elif vintage == "v3.2":
        gdp_t, pop_t, pop_fb = load_v32()
    else:
        raise ValueError(vintage)
    S, C, Y = len(SSPS), len(iso_codes), len(GRID)
    gdp = np.full((S, C, Y), np.nan)
    pop = np.full((S, C, Y), np.nan)
    avail = np.zeros((S, C), bool)
    source = [["" for _ in range(C)] for _ in range(S)]
    for si, scen in enumerate(SSPS):
        for ci, iso in enumerate(iso_codes):
            src = []
            g = gdp_t.get((scen, iso))
            if g is not None:
                gdp[si, ci] = loglinear_interp(g[0], g[1], GRID)
                src.append("gdp:primary")
            p = pop_t.get((scen, iso))
            if p is None and (scen, iso) in pop_fb:
                p = pop_fb[(scen, iso)]
                src.append("pop:fallback(OECD Env-Growth population)")
            elif p is not None:
                src.append("pop:primary")
            if p is not None:
                pop[si, ci] = loglinear_interp(p[0], p[1], GRID)
            avail[si, ci] = g is not None and p is not None
            source[si][ci] = ";".join(src)
    return SSPSeries(vintage=vintage, years=GRID.copy(), gdp=gdp, pop=pop, avail=avail,
                     source=source)


def harmonise(base_value: np.ndarray, series: np.ndarray, years: np.ndarray,
              base_year: int, out_years: np.ndarray) -> np.ndarray:
    """Ratio harmonisation: base_value[c] * series[c, t] / series[c, base_year].

    base_value: [C]; series: [C, Y] on `years`; returns [C, len(out_years)].
    """
    bi = int(np.where(years == base_year)[0][0])
    idx = np.searchsorted(years, out_years)
    ref = series[:, bi]
    return base_value[:, None] * series[:, idx] / ref[:, None]


def save_processed(all_series: dict[str, SSPSeries], iso_codes: list[str]) -> None:
    os.makedirs(os.path.dirname(PROCESSED), exist_ok=True)
    arrays = {"years": GRID}
    meta = {"iso_codes": iso_codes, "ssps": list(SSPS), "vintages": {}}
    for v, s in all_series.items():
        key = v.replace(".", "")
        arrays[f"{key}_gdp"] = s.gdp.astype(np.float32)   # ratios only need ~7 sig. digits
        arrays[f"{key}_pop"] = s.pop.astype(np.float32)
        arrays[f"{key}_avail"] = s.avail
        meta["vintages"][v] = {"key": key, "source": s.source}
    np.savez_compressed(PROCESSED, **arrays)
    with open(PROCESSED_META, "w") as fh:
        json.dump(meta, fh)


def load_processed(vintage: str) -> SSPSeries:
    d = np.load(PROCESSED, allow_pickle=False)
    with open(PROCESSED_META) as fh:
        meta = json.load(fh)
    info = meta["vintages"][vintage]
    k = info["key"]
    return SSPSeries(vintage=vintage, years=d["years"], gdp=d[f"{k}_gdp"].astype(float),
                     pop=d[f"{k}_pop"].astype(float),
                     avail=d[f"{k}_avail"], source=info["source"])


def ssp_driver_paths(panel, series: SSPSeries, ssp: str, base_year: int,
                     out_years: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Harmonised GDP and population paths [C, len(out_years)] for one SSP.

    Base values are the panel's observed GDP/population in `base_year`.
    """
    si = SSPS.index(ssp)
    bk = int(np.where(panel.years == base_year)[0][0])
    gdp = harmonise(panel.gdp[:, bk], series.gdp[si], series.years, base_year, out_years)
    pop = harmonise(panel.population[:, bk], series.pop[si], series.years, base_year, out_years)
    return gdp, pop
