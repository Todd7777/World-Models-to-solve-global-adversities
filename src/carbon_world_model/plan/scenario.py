"""Future scenario construction for decarbonisation planning.

A scenario supplies the *exogenous* drivers the world model needs to project forward
(GDP and population paths, process emissions), the observed starting state, and the
physical feasibility limits on how fast the energy system can be changed. Drivers come
from the Shared Socioeconomic Pathways (SSP1-5, IIASA SSP database v3.2, 2025) by
default; a simple documented heuristic remains available for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config, FUELS, FOSSIL_FUELS, LOW_CARBON_FUELS
from ..data.ssp import SSPS, load_processed, ssp_driver_paths
from ..model.world_model import observed_inits

FOSSIL_IDX = tuple(FUELS.index(f) for f in FOSSIL_FUELS)
LOWC_IDX = tuple(FUELS.index(f) for f in LOW_CARBON_FUELS)

# A deployable low-carbon portfolio (shares within low-carbon capacity added each
# year). Solar/wind-led, consistent with the composition of the 2010s-2020s build-out.
CLEAN_PORTFOLIO = {
    "nuclear": 0.15, "hydro": 0.05, "wind": 0.30,
    "solar": 0.43, "biofuel": 0.03, "other_renewable": 0.04,
}


@dataclass
class Scenario:
    name: str
    years: np.ndarray            # [Tf] e.g. 2022..2050 (index 0 = observed anchor year)
    gdp: np.ndarray              # [C, Tf]
    pop: np.ndarray              # [C, Tf]
    process: np.ndarray          # [C, Tf]
    inits: dict                  # observed state at the anchor: I0, s0, mom_I0, mom_s0
    tfrac: np.ndarray            # [Tf] time feature, capped at the end-of-history level
    g_max: float                 # max sustained annual low-carbon-share gain
    h_max: float                 # max sustained annual intensity improvement
    fossil_weights: np.ndarray   # [C, F] where fossil reduction is taken from
    clean_weights: np.ndarray    # [F] where low-carbon additions go
    countries: list[str]
    p_max: float = 0.0           # max sustained annual process-CO2 reduction (0 = lever off)

    @property
    def C(self):
        return self.gdp.shape[0]

    @property
    def Tf(self):
        return self.gdp.shape[1]

    # Backwards-compatible accessors.
    @property
    def I0(self):
        return self.inits["I0"]

    @property
    def s0(self):
        return self.inits["s0"]


def _cagr(series, k0, k1):
    a, b = series[:, k0], series[:, k1]
    yrs = max(1, k1 - k0)
    return np.where((a > 0) & (b > 0), (b / np.clip(a, 1e-9, None)) ** (1 / yrs) - 1.0, 0.0)


def _best_sustained_rate(x, window=10):
    """For each country, the fastest average annual change sustained over any `window`
    years (a defensible proxy for a sustainable maximum, unlike single-year spikes)."""
    C, T = x.shape
    best = np.zeros(C)
    for c in range(C):
        for a in range(0, T - window):
            best[c] = max(best[c], (x[c, a + window] - x[c, a]) / window)
    return best


def feasibility_bounds(panel, window: int = 10) -> tuple[float, float]:
    """Feasible *sustained* annual change limits from the fastest decade-long national
    rates (see README). Single-year maxima are not sustainable and are avoided; raw
    upper percentiles of intensity change are contaminated by petrostate GDP swings,
    so the median sustained rate, capped at 3.5%/yr, is used for efficiency."""
    sh = panel.shares()
    lowc = sh[:, :, list(LOWC_IDX)].sum(axis=2)
    g_max = float(min(np.percentile(_best_sustained_rate(lowc, window), 98), 0.020))
    ei = np.log(np.clip(panel.primary_energy / np.clip(panel.gdp, 1e-6, None), 1e-9, None))
    h_max = float(min(np.percentile(_best_sustained_rate(-ei, window), 50), 0.035))
    return g_max, h_max


def process_bound(panel, window: int = 10, cap: float = 0.035) -> float:
    """Feasible sustained annual reduction of process CO2 (cement, flaring, other
    industry), by the same rule as the efficiency bound: the median, over countries, of
    the fastest decade-long decline in absolute process CO2, capped at 3.5%/yr.

    Historical declines in process CO2 mostly reflect demand collapses (construction
    busts, post-Soviet contraction, a plant closure in Hong Kong) rather than abatement,
    and the largest emitters (China, India, Brazil, Turkey) never sustained any decline. So
    this is a weak, history-based proxy; deep abatement (CCS, clinker substitution) is a
    technology question these data cannot answer."""
    lp = np.log(np.clip(panel.process_co2, 1e-3, None))
    return float(min(np.percentile(_best_sustained_rate(-lp, window), 50), cap))


# Technology-based process bound: published roadmaps, read 2026-11-13.
#   cement      GCCA (2021), "Concrete Future: Roadmap to Net Zero": 20% CO2 reduction per
#               tonne of cement by 2030 against 2020, and "net zero concrete by 2050" with
#               on-site carbon capture counted (offsets are not).
#   flaring     World Bank, Zero Routine Flaring by 2030: governments and oil companies
#               commit "to end routine flaring no later than 2030".
# Assumptions made here, not sourced:
#   * no progress counted for 2020-2022, so the 2030 milestone is applied from 2022;
#   * "net zero" is represented as a 1% residual of 2022 direct emissions from 2050;
#   * 5% of flaring is non-routine (safety) flaring and remains after 2030;
#   * other industrial process CO2 follows the cement roadmap.
# Milestones are fractions of the 2022 level, interpolated linearly in between.
TECH_PROCESS_MILESTONES = {
    "cement": {2022: 1.0, 2030: 0.80, 2050: 0.01},
    "flaring": {2022: 1.0, 2030: 0.05},
    "other_industry": {2022: 1.0, 2030: 0.80, 2050: 0.01},
}
PROCESS_COMPONENTS_JSON = "data/processed/process_components.json"


def technology_process_floor(years, shares: np.ndarray) -> np.ndarray:
    """[C, Tf] lowest process CO2 (fraction of the anchor-year level) that the roadmaps allow
    in each year: each country's 2022 component shares applied to the milestone paths."""
    years = np.asarray(years, dtype=float)
    paths = []
    for comp in ("cement", "flaring", "other_industry"):
        m = TECH_PROCESS_MILESTONES[comp]
        xs, ys = np.array(sorted(m)), np.array([m[k] for k in sorted(m)])
        paths.append(np.interp(years, xs, ys))          # held at the last value afterwards
    return np.asarray(shares) @ np.stack(paths)          # [C,3] @ [3,Tf]


def technology_process_rates(years, shares: np.ndarray) -> np.ndarray:
    """[C, Tf] per-year maximum abatement rates that make maximum effort follow the floor
    exactly: p[t] = log(F[t] / F[t+1]); the last year's rate is unused (0)."""
    F = technology_process_floor(years, shares)
    p = np.zeros_like(F)
    p[:, :-1] = np.log(F[:, :-1] / F[:, 1:])
    return p


def load_process_shares(panel) -> np.ndarray:
    import json
    d = json.load(open(PROCESS_COMPONENTS_JSON))
    if d["iso_codes"] != list(getattr(panel, "iso_codes", d["iso_codes"])) or \
            d["countries"] != list(panel.countries):
        raise ValueError("process_components.json does not match the panel's countries")
    return np.asarray(d["shares"], dtype=float)


def _heuristic_drivers(panel, Tf, gdp_longrun, pop_longrun):
    k = panel.n_years - 1
    C = panel.n_countries
    g_recent = np.clip(_cagr(panel.gdp, max(0, k - 10), k), -0.02, 0.10)
    p_recent = np.clip(_cagr(panel.population, max(0, k - 10), k), -0.02, 0.05)
    gdp = np.zeros((C, Tf)); pop = np.zeros((C, Tf))
    gdp[:, 0], pop[:, 0] = panel.gdp[:, k], panel.population[:, k]
    for t in range(1, Tf):
        frac = t / (Tf - 1)
        gdp[:, t] = gdp[:, t - 1] * (1 + (1 - frac) * g_recent + frac * gdp_longrun)
        pop[:, t] = pop[:, t - 1] * (1 + (1 - frac) * p_recent + frac * pop_longrun)
    return gdp, pop


def build_scenario(panel, cfg: Config, end_year: int = 2050, ssp: str | None = "SSP2",
                   ssp_vintage: str = "v3.2", gdp_longrun: float = 0.015,
                   pop_longrun: float = 0.0, process_lever: str = "historical") -> Scenario:
    """Anchor at the last data year (observed state) and project drivers forward.

    ssp='SSP1'..'SSP5' uses that SSP's GDP and population (ratio-harmonised to the
    panel's last year); ssp=None uses the heuristic (recent CAGR tapering to a long-run
    rate). Process (cement/flaring) CO2 is held at its last observed value.

    process_lever: 'historical' (default) caps process abatement at process_bound(panel), a
    constant rate; 'technology' lets maximum effort follow the roadmap floor
    (technology_process_floor), with per-country, per-year rates. The technology bound is
    meant for maximum-effort runs: its rates become very large near 2050, which would
    dominate the optimisers' effort cost.
    """
    k = panel.n_years - 1
    anchor = int(panel.years[k])
    years = np.arange(anchor, end_year + 1)
    Tf = len(years)
    if ssp is None:
        gdp, pop = _heuristic_drivers(panel, Tf, gdp_longrun, pop_longrun)
        name = "heuristic"
    else:
        gdp, pop = ssp_driver_paths(panel, load_processed(ssp_vintage), ssp, anchor, years)
        name = f"{ssp} ({ssp_vintage})"
    process = np.repeat(panel.process_co2[:, k:k + 1], Tf, axis=1)
    inits = observed_inits(panel, k, cfg.model.momentum, cfg.model.ema_decay)

    g_max, h_max = feasibility_bounds(panel)
    s0 = inits["s0"]
    foss = s0[:, list(FOSSIL_IDX)]
    foss_norm = foss / np.clip(foss.sum(axis=1, keepdims=True), 1e-9, None)
    fossil_weights = np.zeros((panel.n_countries, len(FUELS)))
    for j, idx in enumerate(FOSSIL_IDX):
        fossil_weights[:, idx] = foss_norm[:, j]
    clean_weights = np.zeros(len(FUELS))
    for f, w in CLEAN_PORTFOLIO.items():
        clean_weights[FUELS.index(f)] = w

    return Scenario(name=name, years=years, gdp=gdp, pop=pop, process=process, inits=inits,
                    tfrac=np.ones(Tf), g_max=g_max, h_max=h_max,
                    fossil_weights=fossil_weights, clean_weights=clean_weights,
                    countries=list(panel.countries),
                    p_max=(process_bound(panel) if process_lever == "historical" else
                           technology_process_rates(years, load_process_shares(panel))))


def ssp_ensemble(panel, cfg: Config, end_year: int = 2050, vintage: str = "v3.2") -> list[Scenario]:
    """One scenario per SSP (SSP1..SSP5), identical except for the socioeconomic drivers."""
    return [build_scenario(panel, cfg, end_year, ssp=s, ssp_vintage=vintage) for s in SSPS]
