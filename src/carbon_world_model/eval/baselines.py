"""Baseline emission forecasters for the temporal backtest.

Every baseline sees only data up to and including the cutoff year, plus the realised
GDP and population paths over the forecast horizon (the same "socioeconomic scenario"
the world model is given). This isolates the energy-system model from macroeconomic
forecasting, which is the standard integrated-assessment framing.

All baselines return a full [C, T] CO2 array; callers slice out the horizon years.
"""
from __future__ import annotations

import numpy as np


def _cut_idx(panel, cutoff_year: int) -> int:
    return int(np.where(panel.years == cutoff_year)[0][0])


def persistence(panel, cutoff_year: int) -> np.ndarray:
    """Hold CO2 flat at its cutoff-year value."""
    k = _cut_idx(panel, cutoff_year)
    out = panel.co2.copy()
    out[:, k + 1:] = panel.co2[:, k][:, None]
    return out


def growth_persistence(panel, cutoff_year: int, lookback: int = 5) -> np.ndarray:
    """Extrapolate the recent average annual CO2 growth rate."""
    k = _cut_idx(panel, cutoff_year)
    co2 = panel.co2
    j = max(0, k - lookback)
    # Geometric mean growth over the lookback window.
    ratio = np.clip(co2[:, k], 1e-6, None) / np.clip(co2[:, j], 1e-6, None)
    g = ratio ** (1.0 / max(1, k - j)) - 1.0
    out = co2.copy()
    for t in range(k + 1, panel.n_years):
        out[:, t] = out[:, t - 1] * (1.0 + g)
    return out


def _kaya_components(panel):
    """Return (energy_intensity E/GDP, carbon_intensity CO2/E)."""
    ei = panel.primary_energy / np.clip(panel.gdp, 1e-6, None)
    ci = panel.co2 / np.clip(panel.primary_energy, 1e-6, None)
    return ei, ci


def frozen_kaya(panel, cutoff_year: int) -> np.ndarray:
    """Freeze energy- and carbon-intensity at cutoff; emissions scale with GDP."""
    k = _cut_idx(panel, cutoff_year)
    ei, ci = _kaya_components(panel)
    out = panel.co2.copy()
    for t in range(k + 1, panel.n_years):
        out[:, t] = panel.gdp[:, t] * ei[:, k] * ci[:, k]
    return out


def trend_kaya(panel, cutoff_year: int, lookback: int = 10) -> np.ndarray:
    """Extrapolate log-linear trends in energy- and carbon-intensity, combine with GDP.

    This is a strong baseline: it encodes the ongoing efficiency gains and
    decarbonisation that have actually been happening.
    """
    k = _cut_idx(panel, cutoff_year)
    ei, ci = _kaya_components(panel)
    j = max(0, k - lookback + 1)
    yrs = panel.years[j:k + 1].astype(float)
    out = panel.co2.copy()

    def fit_rate(series):
        # slope of log(series) vs year over the lookback window
        y = np.log(np.clip(series[:, j:k + 1], 1e-9, None))  # [C, L]
        x = yrs - yrs.mean()
        denom = np.sum(x ** 2)
        slope = (y * x).sum(axis=1) / denom  # [C]  d log / d year
        return slope

    r_ei = fit_rate(ei)
    r_ci = fit_rate(ci)
    for t in range(k + 1, panel.n_years):
        dt = panel.years[t] - cutoff_year
        ei_t = ei[:, k] * np.exp(r_ei * dt)
        ci_t = ci[:, k] * np.exp(r_ci * dt)
        out[:, t] = panel.gdp[:, t] * ei_t * ci_t
    return out


BASELINES = {
    "persistence": persistence,
    "growth_persistence": growth_persistence,
    "frozen_kaya": frozen_kaya,
    "trend_kaya": trend_kaya,
}
