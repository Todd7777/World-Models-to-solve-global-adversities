"""Novel carbon dioxide removal (CDR) and peak-warming accounting.

Source (read 2026-11-13): The State of Carbon Dioxide Removal, 3rd edition, executive
summary (June 2026), stateofcdr.org:
  * "Total removal is 2.2 GtCO2 per year ... Conventional CDR represents 99.9% of this
    total";
  * "Novel CDR is 0.002 GtCO2 per year and has been growing at 40% per year";
  * projects built and in construction "would reach 0.008 GtCO2 per year of capacity in
    2030";
  * in the assessed Paris-compatible scenarios, novel CDR "accelerates to over 3.5 GtCO2
    per year by 2050".

Only NOVEL (durable) CDR is a lever here. Conventional CDR (forests, soils) is already
netted inside the land-use CO2 assumption (climate/budgets.land_use_co2), so adding it
would double-count.

Maximum feasible novel CDR follows a logistic curve that starts from the measured level
(0.002 GtCO2/yr in 2025), grows at the measured rate (40%/yr), and saturates at a
ceiling. The central ceiling is the scenario level (3.5 GtCO2/yr), which the curve nears
around 2050, as in the report. The other ceilings (half, double) are sensitivities chosen
here, not sourced.

With removals, net emissions can turn negative and cumulative CO2 falls again. Peak
warming is set by PEAK cumulative CO2, so budgets are compared with the peak, not the
end total.
"""
from __future__ import annotations

import numpy as np

CDR_START_YEAR = 2025
CDR_START_GT = 0.002        # novel CDR, GtCO2/yr
CDR_GROWTH = 0.40           # per year, observed recent growth
CDR_CEILINGS = {"low": 1.75, "central": 3.5, "high": 7.0}   # GtCO2/yr; central = scenarios


def novel_cdr_path(years, ceiling: float = CDR_CEILINGS["central"],
                   start_year: int = CDR_START_YEAR, start: float = CDR_START_GT,
                   rate: float = CDR_GROWTH) -> np.ndarray:
    """Maximum feasible novel CDR (GtCO2/yr, positive = removal) in each year: logistic
    growth from `start` in `start_year` at `rate`, saturating at `ceiling`."""
    t = np.asarray(years, dtype=float) - start_year
    logistic = ceiling / (1.0 + (ceiling / start - 1.0) * np.exp(-rate * t))
    # before the start year: measured level extrapolated back at the measured rate, so the
    # (hypothetical) ceiling does not change the past
    return np.where(t < 0, start * np.exp(rate * t), logistic)


def peak_cumulative(net_gt) -> tuple[float, bool]:
    """(peak cumulative CO2 over the path, whether the peak was reached within it).

    net_gt: annual net CO2 (GtCO2/yr). If net emissions are still positive in the last
    year, cumulative CO2 is still rising, so the returned value is a lower bound."""
    cum = np.cumsum(np.asarray(net_gt, dtype=float))
    return float(cum.max()), bool(np.asarray(net_gt)[-1] <= 0.0)


def net_zero_year(years, net_gt) -> int | None:
    """First year in which annual net CO2 is zero or negative (None if never)."""
    net = np.asarray(net_gt, dtype=float)
    idx = np.where(net <= 0.0)[0]
    return int(np.asarray(years)[idx[0]]) if idx.size else None


# ---------------------------------------------------------------------------
# Land and cost limits (bottom-up supply ceiling)
# ---------------------------------------------------------------------------
# Source (read 2026-11-14): IPCC AR6 WGIII, Chapter 12 (Cross-sectoral perspectives),
# Table 12.6, author-judgement ranges for novel, durable CDR methods:
#   DACCS              TRL 6    USD 100-300/tCO2 (full 84-386)   5-40 GtCO2/yr
#   BECCS              TRL 5-6  USD 15-400                       0.5-11
#   biochar            TRL 6-7  USD 10-345                       0.3-6.6
#   enhanced weathering TRL 3-4 USD 50-200 (24-578)              2-4 (<1-95)
# Ocean alkalinity enhancement and ocean fertilisation (TRL 1-2) are left out here.
# The same chapter: biomass for BECCS in 2050 corresponds to 5.3 GtCO2/yr (1.1-18,
# 5-95th pct) in 1.5C scenarios (C1) and 2.4 GtCO2/yr (0-8.3) in 2C scenarios (C3);
# DACCS needs 40-100 EJ/yr of energy at 10 GtCO2/yr, and its land needs "are not
# large enough to be of concern".
# Choices made here, not sourced:
#   * each method's potential is the midpoint of its range, and is spread evenly over
#     its cost range, so a price cap admits the share of potential that costs less;
#   * the land limit caps BECCS and biochar TOGETHER (both draw on biomass) at the C3
#     BECCS level (central 2.4), with the C1 5th percentile (1.1) and median (5.3) as
#     sensitivities;
#   * enhanced weathering and DACCS have no land cap here; DACCS energy is reported by
#     cdr_spending_and_energy and limited separately (energy_limited_cdr_path, below).
CDR_METHODS = {  # name: (cost_lo, cost_hi USD/t, potential_lo, potential_hi Gt/yr, biomass)
    "daccs": (100.0, 300.0, 5.0, 40.0, False),
    "beccs": (15.0, 400.0, 0.5, 11.0, True),
    "biochar": (10.0, 345.0, 0.3, 6.6, True),
    "enhanced_weathering": (50.0, 200.0, 2.0, 4.0, False),
}
LAND_CAPS = {"low": 1.1, "central": 2.4, "high": 5.3}       # GtCO2/yr, biomass-based CDR
DACCS_EJ_PER_GT = (4.0, 10.0)                              # 40-100 EJ/yr at 10 GtCO2/yr


def cdr_supply(price_cap: float, land_cap: float = LAND_CAPS["central"]) -> dict:
    """Supply of each method at full scale (GtCO2/yr) under a price cap and a land cap."""
    out = {}
    for name, (clo, chi, plo, phi, _) in CDR_METHODS.items():
        out[name] = 0.5 * (plo + phi) * float(np.clip((price_cap - clo) / (chi - clo), 0.0, 1.0))
    bio = sum(v for k, v in out.items() if CDR_METHODS[k][4])
    if bio > land_cap:
        for k in out:
            if CDR_METHODS[k][4]:
                out[k] *= land_cap / bio
    return out


def cdr_ceiling(price_cap: float, land_cap: float = LAND_CAPS["central"]) -> float:
    return float(sum(cdr_supply(price_cap, land_cap).values()))


def cdr_spending_and_energy(price_cap: float, land_cap: float = LAND_CAPS["central"]) -> dict:
    """At full scale: average cost (USD/t), annual spending (USD trillion/yr), and DACCS
    energy demand (EJ/yr, low-high). Uniform costs within each range, as in cdr_supply."""
    s = cdr_supply(price_cap, land_cap)
    spend = sum(v * 0.5 * (CDR_METHODS[k][0] + min(price_cap, CDR_METHODS[k][1]))
                for k, v in s.items())          # Gt x USD/t = USD bn; /1000 -> tn
    total = sum(s.values())
    return {"avg_cost_usd_t": spend / total if total else 0.0, "spend_usd_tn": spend / 1000.0,
            "daccs_ej": [s["daccs"] * e for e in DACCS_EJ_PER_GT], "supply_gt": s}


# Variants used in planning runs (scripts/12_to_2100.py): name -> ceiling (GtCO2/yr).
CDR_VARIANTS = {
    **{k: v for k, v in CDR_CEILINGS.items()},                       # scenario-level ceilings
    "cost100_land_central": cdr_ceiling(100.0, LAND_CAPS["central"]),
    "cost100_land_low": cdr_ceiling(100.0, LAND_CAPS["low"]),
    "cost100_land_high": cdr_ceiling(100.0, LAND_CAPS["high"]),
    "cost200_land_central": cdr_ceiling(200.0, LAND_CAPS["central"]),
    "cost300_land_central": cdr_ceiling(300.0, LAND_CAPS["central"]),
}


# ---------------------------------------------------------------------------
# Energy limit for direct air capture
# ---------------------------------------------------------------------------
# DACCS needs 40-100 EJ/yr of energy at 10 GtCO2/yr (AR6 WGIII Ch12; ~80% heat, ~20%
# electricity), i.e. 4-10 EJ per GtCO2; 7 is used here. It must run on low-carbon energy
# or it emits much of what it removes. The limit ties DACCS to the modelled energy
# system: in each year DACCS may use at most a fraction f of the case's own low-carbon
# primary energy (f = 10% central, 5% and 20% as sensitivities; choices made here, not
# sourced; for scale, AR6's 40-100 EJ for 10 GtCO2 is 7-17% of today's ~600 EJ primary
# supply). Primary energy is the panel's substitution-method measure, so this is an
# approximation of the heat and electricity DACCS would need. When the limit binds, only
# the DACCS share of removals is cut; other methods are not scaled up to replace it.
DACCS_EJ_PER_GT_CENTRAL = 7.0
ENERGY_FRACTIONS = {"low": 0.05, "central": 0.10, "high": 0.20}


def energy_limited_cdr_path(years, price_cap: float, land_cap: float, lowcarbon_ej,
                            frac: float = ENERGY_FRACTIONS["central"],
                            ej_per_gt: float = DACCS_EJ_PER_GT_CENTRAL) -> tuple[np.ndarray, np.ndarray]:
    """(removals GtCO2/yr, DACCS removals GtCO2/yr) per year: the land- and cost-limited
    logistic path, with DACCS capped at frac * low-carbon energy / ej_per_gt."""
    supply = cdr_supply(price_cap, land_cap)
    K = sum(supply.values())
    x = novel_cdr_path(years, K)
    dac = x * (supply["daccs"] / K if K > 0 else 0.0)
    dac_cap = frac * np.asarray(lowcarbon_ej, dtype=float) / ej_per_gt
    dac_ok = np.minimum(dac, dac_cap)
    return x - (dac - dac_ok), dac_ok
