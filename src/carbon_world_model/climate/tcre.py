"""Climate translation via the Transient Climate Response to cumulative Emissions.

Peak CO2-induced warming is, to first order, proportional to *cumulative* CO2
emissions -- the TCRE relationship (IPCC AR6). This lets the model express emissions
pathways in the currency that matters: degrees of warming, and avoided warming.

TCRE is a global quantity; this model covers ~93% of global CO2 (60 countries), so
helpers that need a global total scale the panel sum by `global_coverage`.
"""
from __future__ import annotations

from ..config import ClimateConfig


def warming_from_cumulative(cum_gtco2: float, clim: ClimateConfig) -> dict:
    """Warming contribution (degC) from a cumulative CO2 amount (GtCO2), with the
    TCRE uncertainty range. Note: 1 TtCO2 = 1000 GtCO2."""
    f = cum_gtco2 / 1000.0  # -> TtCO2
    return {
        "central": f * clim.tcre_degC_per_ttco2,
        "low": f * clim.tcre_low,
        "high": f * clim.tcre_high,
    }


def avoided_warming(cum_bau_gt: float, cum_scenario_gt: float, clim: ClimateConfig) -> dict:
    """Warming avoided (degC) by a scenario relative to business-as-usual."""
    return warming_from_cumulative(cum_bau_gt - cum_scenario_gt, clim)


def remaining_budget_gtco2(target_degC: float, clim: ClimateConfig | None = None,
                           likelihood: float = 0.5) -> float:
    """Remaining global CO2 budget (GtCO2, total CO2 incl. land use) from the start of 2023
    for a `likelihood` of holding peak warming below `target_degC`: the published
    estimate (climate/budgets.py, Forster et al. 2025), which allows for non-CO2 warming.

    Until 2026-11-12 this returned (target - 1.24) / TCRE, which attributes all remaining
    warming to CO2 and gave 578 / 1689 GtCO2 for 1.5 / 2 C, about 2.7x / 1.5x the published
    50% values (see budgets.tcre_only_budget)."""
    from .budgets import budget_gt
    return budget_gt(target_degC, likelihood, base_year=2023)


def scale_to_global(panel_cum_gt: float, coverage: float = 0.926) -> float:
    """Scale a 60-country cumulative to an approximate global total by CO2 coverage."""
    return panel_cum_gt / coverage
