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


def remaining_budget_gtco2(target_degC: float, clim: ClimateConfig) -> float:
    """Remaining global CO2 budget (GtCO2) from the base year to hold peak warming at
    `target_degC` above pre-industrial, using TCRE and the realised warming to date."""
    headroom = target_degC - clim.warming_baseline_degC
    return (headroom / clim.tcre_degC_per_ttco2) * 1000.0


def scale_to_global(panel_cum_gt: float, coverage: float = 0.926) -> float:
    """Scale a 60-country cumulative to an approximate global total by CO2 coverage."""
    return panel_cum_gt / coverage
