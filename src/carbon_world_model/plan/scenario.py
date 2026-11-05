"""Future scenario construction for decarbonisation planning.

A scenario supplies the *exogenous* drivers the world model needs to project forward
(GDP and population paths, process emissions) plus the physical feasibility limits on
how fast the energy system can be changed. Everything here is a transparent, documented
assumption -- the planner's outputs are conditional on it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config, FUELS, FOSSIL_FUELS, LOW_CARBON_FUELS

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
    years: np.ndarray            # [Tf] e.g. 2022..2050 (2022 is the init anchor)
    gdp: np.ndarray              # [C, Tf]
    pop: np.ndarray              # [C, Tf]
    process: np.ndarray          # [C, Tf]
    I0: np.ndarray               # [C] log-intensity at anchor year (from data)
    s0: np.ndarray               # [C, F] fuel shares at anchor year (from data)
    tfrac: np.ndarray            # [Tf] time feature, capped at 1.0 (freeze secular trend)
    g_max: float                 # max annual low-carbon-share gain (feasibility)
    h_max: float                 # max annual intensity-improvement (feasibility)
    fossil_weights: np.ndarray   # [C, F] where fossil reduction is taken from
    clean_weights: np.ndarray    # [F] where low-carbon additions go
    countries: list[str]

    @property
    def C(self):
        return self.gdp.shape[0]

    @property
    def Tf(self):
        return self.gdp.shape[1]


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
            rate = (x[c, a + window] - x[c, a]) / window
            best[c] = max(best[c], rate)
    return best


def feasibility_bounds(panel, window: int = 10) -> tuple[float, float]:
    """Feasible *sustained* annual change limits, from the fastest decade-long rates
    countries have actually sustained (p90 across countries). Single-year maxima are
    much larger but not sustainable, so we deliberately avoid them. Hard caps guard
    against data artefacts.
    """
    sh = panel.shares()
    lowc = sh[:, :, list(LOWC_IDX)].sum(axis=2)        # [C,T] low-carbon share
    g_sustained = _best_sustained_rate(lowc, window)
    # Frontier of sustained low-carbon primary-energy share gains (~2%/yr, Denmark).
    g_max = float(min(np.percentile(g_sustained, 98), 0.020))

    ei = np.log(np.clip(panel.primary_energy / np.clip(panel.gdp, 1e-6, None), 1e-9, None))
    h_sustained = _best_sustained_rate(-ei, window)    # -log intensity rising = improving
    # Genuine efficiency frontier (~3-3.5%/yr, China). Raw upper percentiles are
    # inflated by oil-price-driven GDP swings in petrostates, so we use the median
    # sustained rate capped at 3.5%/yr rather than the contaminated tail.
    h_max = float(min(np.percentile(h_sustained, 50), 0.035))
    return g_max, h_max


def build_scenario(panel, cfg: Config, end_year: int = 2050,
                   gdp_longrun: float = 0.015, pop_longrun: float = 0.0) -> Scenario:
    """Anchor at the last data year and project drivers forward.

    GDP and population grow from their last observed value at each country's recent
    (10-year) CAGR, tapering linearly to a common long-run rate by `end_year`. Process
    (cement/flaring) emissions are held at their last value. These are deliberately
    simple, documented assumptions; swap them for SSP pathways for scenario analysis.
    """
    k = panel.n_years - 1
    anchor = int(panel.years[k])
    fut_years = np.arange(anchor, end_year + 1)
    Tf = len(fut_years)
    C = panel.n_countries

    g_recent = np.clip(_cagr(panel.gdp, max(0, k - 10), k), -0.02, 0.10)
    p_recent = np.clip(_cagr(panel.population, max(0, k - 10), k), -0.02, 0.05)

    gdp = np.zeros((C, Tf)); pop = np.zeros((C, Tf)); process = np.zeros((C, Tf))
    gdp[:, 0] = panel.gdp[:, k]
    pop[:, 0] = panel.population[:, k]
    process[:, 0] = panel.process_co2[:, k]
    for t in range(1, Tf):
        frac = t / (Tf - 1)
        g_rate = (1 - frac) * g_recent + frac * gdp_longrun
        p_rate = (1 - frac) * p_recent + frac * pop_longrun
        gdp[:, t] = gdp[:, t - 1] * (1 + g_rate)
        pop[:, t] = pop[:, t - 1] * (1 + p_rate)
        process[:, t] = process[:, 0]

    I0 = np.log(panel.primary_energy[:, k] + 1e-6) - np.log(panel.gdp[:, k] + 1e-6)
    s0 = panel.shares()[:, k, :]

    # Freeze the secular time trend at its end-of-history level.
    tfrac = np.ones(Tf)

    g_max, h_max = feasibility_bounds(panel)

    # Where fossil reduction is drawn from: each country's current fossil composition.
    foss = s0[:, list(FOSSIL_IDX)]
    foss_norm = foss / np.clip(foss.sum(axis=1, keepdims=True), 1e-9, None)
    fossil_weights = np.zeros((C, len(FUELS)))
    for j, idx in enumerate(FOSSIL_IDX):
        fossil_weights[:, idx] = foss_norm[:, j]
    clean_weights = np.zeros(len(FUELS))
    for f, w in CLEAN_PORTFOLIO.items():
        clean_weights[FUELS.index(f)] = w

    return Scenario(years=fut_years, gdp=gdp, pop=pop, process=process, I0=I0, s0=s0,
                    tfrac=tfrac, g_max=g_max, h_max=h_max, fossil_weights=fossil_weights,
                    clean_weights=clean_weights, countries=list(panel.countries))
