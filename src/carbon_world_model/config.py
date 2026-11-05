"""Central configuration for the Carbon World Model.

Everything that defines *what* we model (fuels, countries, time window) and the
default hyper-parameters lives here so experiments are reproducible from a single
object. Values are plain Python (no YAML dependency) and fully typed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Sequence
import json


# ---------------------------------------------------------------------------
# Fuel taxonomy
# ---------------------------------------------------------------------------
# Primary-energy consumption columns in the OWID energy dataset (units: TWh,
# input-equivalent / "substitution" method). These partition primary energy.
FUELS: tuple[str, ...] = (
    "coal",
    "oil",
    "gas",
    "nuclear",
    "hydro",
    "wind",
    "solar",
    "biofuel",
    "other_renewable",
)

# The three combustion fossil fuels that carry (almost) all energy CO2.
FOSSIL_FUELS: tuple[str, ...] = ("coal", "oil", "gas")

# Low-carbon sources (zero direct combustion CO2 in this model).
LOW_CARBON_FUELS: tuple[str, ...] = (
    "nuclear",
    "hydro",
    "wind",
    "solar",
    "biofuel",
    "other_renewable",
)

# OWID consumption column name for each fuel (TWh, substitution method).
CONSUMPTION_COL = {f: f"{f}_consumption" for f in FUELS}
CONSUMPTION_COL["other_renewable"] = "other_renewable_consumption"

# OWID CO2-by-source column name for each fossil fuel (million tonnes CO2).
CO2_SOURCE_COL = {"coal": "coal_co2", "oil": "oil_co2", "gas": "gas_co2"}

# Non-energy / process CO2 columns (cement + flaring + other industry), MtCO2.
PROCESS_CO2_COLS: tuple[str, ...] = ("cement_co2", "flaring_co2", "other_industry_co2")


@dataclass(frozen=True)
class DataConfig:
    co2_csv: str = "data/raw/owid_co2/owid-co2-data.csv"
    energy_csv: str = "data/raw/owid_energy/owid-energy-data.csv"
    processed_npz: str = "data/processed/panel.npz"
    meta_json: str = "data/processed/panel_meta.json"

    # Modelling window. Modern energy-mix data (wind/solar) is reliable from ~1990.
    year_start: int = 1990
    year_end: int = 2022  # last year with near-complete coverage for most countries

    # Country selection thresholds.
    min_primary_energy_twh: float = 50.0  # average over window; drops micro-states
    min_year_coverage: float = 0.90  # fraction of years with usable energy+co2 data
    # Hard cap on number of countries (by average primary energy). None = no cap.
    max_countries: int | None = 60


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 64
    embed_dim: int = 8
    depth: int = 2
    # Inertia: max absolute year-on-year change in a fuel's share of primary energy
    # (as a fraction). Calibrated loosely to the fastest historical transitions.
    max_share_change_per_year: float = 0.05
    # Bounds on annual energy-intensity improvement the dynamics can express.
    max_intensity_change_per_year: float = 0.08
    seed: int = 0


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 3e-3
    steps: int = 4000
    weight_decay: float = 1e-4
    # Loss weights.
    w_emissions: float = 1.0
    w_demand: float = 0.5
    w_mix: float = 1.0
    w_smooth: float = 1e-2
    w_aggregate: float = 0.5  # fit the global-sum emissions trajectory (policy quantity)
    w_level: float = 0.2      # emissions-weighted level term (stops big emitters drifting)
    grad_clip: float = 1.0
    seed: int = 0
    # Recency weighting: recent years carry more weight in the loss (0 = off). Tested
    # and left off: emphasising recent years worsened China's extrapolation.
    recency_halflife: float = 0.0


@dataclass(frozen=True)
class ClimateConfig:
    # Transient Climate Response to cumulative Emissions (IPCC AR6 central estimate):
    # ~1.65 degC per 1000 PgC == 0.45 degC per 1000 GtCO2. Likely range 0.27-0.63.
    tcre_degC_per_ttco2: float = 0.45  # degC per 1000 GtCO2 (= per TtCO2)
    tcre_low: float = 0.27
    tcre_high: float = 0.63
    # Approx warming already realised relative to 1850-1900 by the base year.
    warming_baseline_degC: float = 1.24  # ~2022 best estimate


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    climate: ClimateConfig = field(default_factory=ClimateConfig)

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)


def default_config() -> Config:
    return Config()
