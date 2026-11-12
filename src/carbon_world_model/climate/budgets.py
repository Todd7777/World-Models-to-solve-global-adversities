"""Remaining carbon budgets with their uncertainty, and the accounting needed to compare a
projection against them.

Source (read 2026-11-12): Forster, P. M. et al. (2025), Indicators of Global Climate Change
2024, Earth System Science Data 17, 2641-2680, doi:10.5194/essd-17-2641-2025, Table 8.
Remaining carbon budgets (GtCO2, total CO2 including land use) from the start of 2025, by
likelihood of limiting warming to each threshold, "considering only uncertainty in TCRE"
(median TCRE 0.45 C per 1000 GtCO2). They include future non-CO2 warming from AR6
net-zero-aligned scenarios (0.11 C for 1.5 C) and an Earth-system-feedback adjustment, and
"can be higher or lower by around 200 Gt CO2" depending on non-CO2 reductions. (The table
caption says "start of 2024"; its header, the main text and the paper's summary table
say 2025, which is used here.) For comparison, the paper gives the AR6 WG1 value for a 50%
chance of 1.5 C as 500 GtCO2 from the start of 2020.

Rebasing to the start of 2023 adds the global CO2 emitted in 2023 and 2024: 40.6 and
41.6 GtCO2 (Global Carbon Project, Global Carbon Budget 2024 press release, 2024 projected).

Why this replaced the earlier budget: the project used (target - 1.24 C) / TCRE, which
attributes all remaining warming to CO2. With no allowance for non-CO2 warming it gave
578 and 1689 GtCO2 from 2023 for 1.5 and 2 C, about 2.7x and 1.5x the published 50%
values (212 and 1132 GtCO2 from 2023). `tcre_only_budget` keeps that formula for
comparison.
"""
from __future__ import annotations

import numpy as np

SOURCE = ("Forster et al. (2025), Indicators of Global Climate Change 2024, ESSD 17, 2641-2680, "
          "Table 8 (from the start of 2025, TCRE uncertainty only)")
LIKELIHOODS = (0.17, 0.33, 0.50, 0.67, 0.83)
TABLE_BASE_YEAR = 2025
RCB_TABLE = {  # target C -> budgets (GtCO2) at LIKELIHOODS, from the start of 2025
    1.5: (320, 200, 130, 80, 30),
    1.6: (620, 420, 310, 240, 160),
    1.7: (910, 640, 490, 390, 290),
    2.0: (1790, 1310, 1050, 870, 690),
}
NON_CO2_SHIFT_GT = 200.0  # "higher or lower by around 200 Gt CO2" (non-CO2 reductions)
# Global total CO2 (fossil + land use), GtCO2: Global Carbon Budget 2024 (2024 projected).
HISTORICAL_TOTAL_CO2 = {2023: 40.6, 2024: 41.6}
LAND_USE_CO2_2024 = 4.2  # GtCO2/yr, Global Carbon Budget 2024 (projected)


def _shift_to(base_year: int) -> float:
    """GtCO2 to ADD to a start-of-2025 budget to express it from the start of base_year."""
    if base_year > TABLE_BASE_YEAR:
        raise ValueError("budgets after the table's base year need projected emissions")
    return float(sum(HISTORICAL_TOTAL_CO2[y] for y in range(base_year, TABLE_BASE_YEAR)))


def budget_gt(target: float, likelihood: float = 0.5, base_year: int = 2023,
              non_co2_shift: float = 0.0) -> float:
    """Remaining budget (GtCO2) for a `likelihood` of staying below `target`, from the start
    of `base_year`. Likelihoods between table rows are interpolated linearly."""
    b = np.interp(likelihood, LIKELIHOODS, RCB_TABLE[target])
    return float(b + _shift_to(base_year) + non_co2_shift)


def likelihood_below(target: float, total_gt, base_year: int = 2023,
                     non_co2_shift: float = 0.0) -> np.ndarray:
    """Likelihood of staying below `target` given total future CO2 (GtCO2 from the start of
    `base_year` until net zero), interpolated in the published table. Values beyond the
    table are clipped to its range: 0.17 means "17% or less", 0.83 "83% or more"."""
    b = np.asarray(RCB_TABLE[target], dtype=float) + _shift_to(base_year) + non_co2_shift
    # budgets fall as likelihood rises; np.interp needs increasing x
    return np.interp(np.asarray(total_gt, dtype=float), b[::-1], np.asarray(LIKELIHOODS)[::-1])


def tcre_only_budget(target: float, tcre_per_ttco2: float = 0.45,
                     warming_to_date: float = 1.24) -> float:
    """The project's earlier budget: all remaining warming attributed to CO2 (no non-CO2
    allowance). Kept only for comparison."""
    return (target - warming_to_date) / tcre_per_ttco2 * 1000.0


def land_use_co2(start: int = 2023, end: int = 2050, mode: str = "linear_to_zero") -> float:
    """Assumed global land-use CO2, GtCO2 summed over start..end. 'zero', 'constant' (at the
    2024 level), or 'linear_to_zero' (from the 2024 level in `start` to zero in `end`)."""
    n = end - start + 1
    if mode == "zero":
        return 0.0
    if mode == "constant":
        return LAND_USE_CO2_2024 * n
    if mode == "linear_to_zero":
        return float(LAND_USE_CO2_2024 * np.sum(np.linspace(1.0, 0.0, n)))
    raise ValueError(mode)


def tail_after(level_end: float, net_zero_year: int | None, end: int = 2050) -> float:
    """CO2 after `end` if emissions fall linearly from `level_end` (in `end`) to zero in
    `net_zero_year`. None means no tail (the projection is treated as ending at net zero)."""
    if net_zero_year is None:
        return 0.0
    n = net_zero_year - end
    return float(level_end * np.sum(1.0 - np.arange(1, n + 1) / n))


def total_until_net_zero(fossil_cum: float, fossil_end_level: float, land_use: str,
                         net_zero_year: int | None) -> float:
    """Global CO2 from 2023 until net zero (GtCO2): projected fossil and industry CO2
    2023-2050, plus assumed land-use CO2 2023-2050, plus an assumed post-2050 tail."""
    return fossil_cum + land_use_co2(mode=land_use) + tail_after(fossil_end_level, net_zero_year)
