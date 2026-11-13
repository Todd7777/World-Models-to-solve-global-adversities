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
