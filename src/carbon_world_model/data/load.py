"""Load and filter the raw OWID CO2 and energy CSVs."""
from __future__ import annotations

import pandas as pd

from ..config import Config, CONSUMPTION_COL, FUELS


def _is_country(df: pd.DataFrame) -> pd.Series:
    """True for rows that are real countries (3-letter ISO), not aggregates/regions."""
    iso = df["iso_code"].astype("string")
    ok = iso.notna() & (iso.str.len() == 3)
    # OWID uses pseudo codes like 'OWID_WRL' (len 8) for aggregates; the len==3 test
    # already excludes them, but guard the common ones explicitly.
    ok &= ~df["country"].isin(["World"])
    return ok.fillna(False)


def load_raw(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (co2_df, energy_df) filtered to real countries within the year window."""
    co2 = pd.read_csv(cfg.data.co2_csv, low_memory=False)
    energy = pd.read_csv(cfg.data.energy_csv, low_memory=False)

    y0, y1 = cfg.data.year_start, cfg.data.year_end
    co2 = co2[(co2.year >= y0) & (co2.year <= y1) & _is_country(co2)].copy()
    energy = energy[(energy.year >= y0) & (energy.year <= y1) & _is_country(energy)].copy()
    return co2, energy


def select_countries(energy: pd.DataFrame, cfg: Config) -> list[str]:
    """Pick countries with good energy-mix coverage and material energy use.

    Ranked by average primary-energy consumption so the selection captures the
    bulk of global emissions. Returns country names (OWID 'country' field).
    """
    cons_cols = [CONSUMPTION_COL[f] for f in FUELS]
    need = cons_cols + ["primary_energy_consumption", "gdp", "population"]
    n_years = cfg.data.year_end - cfg.data.year_start + 1

    def covered_years(d: pd.DataFrame) -> int:
        return int(d.dropna(subset=need).year.nunique())

    coverage = energy.groupby("country").apply(covered_years, include_groups=False)
    mean_pe = energy.groupby("country")["primary_energy_consumption"].mean()

    eligible = [
        c
        for c in mean_pe.sort_values(ascending=False).index
        if coverage.get(c, 0) >= cfg.data.min_year_coverage * n_years
        and mean_pe[c] >= cfg.data.min_primary_energy_twh
    ]
    if cfg.data.max_countries is not None:
        eligible = eligible[: cfg.data.max_countries]
    return eligible
