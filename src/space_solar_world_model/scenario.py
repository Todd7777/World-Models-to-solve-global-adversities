"""Exogenous world parameters: technology/market scenarios and their uncertainty.

Each ``Scenario`` gives *central values* of the exogenous variables of the structural
causal model; ``SIGMA`` gives their uncertainty (log-normal sigma unless noted). The three
presets bracket the plausible range:

* ``today``     -- 2025-26 reality: Falcon-class launch (~$2.5k/kg; Falcon 9 list is
                   ~$74M for 22.8 t), flight arrays ~100-120 W/kg (ROSA class),
                   mass-produced spacecraft ~$1.5k/kg (Starlink-class).
* ``starship``  -- fully reusable heavy lift in routine service (~$300/kg; the one public
                   Starship price, $90M for Starlab, implies ~$600/kg), 300 W/kg arrays.
* ``advanced``  -- the optimistic end used by SBSP advocates and NASA's combined-
                   sensitivity case: ~$50/kg launch, 1 kW/kg thin-film arrays, $200/kg
                   hardware, 30-year life.

Ground costs (utility single-axis-tracking PV and 4-h batteries) fall too, so space
solar is always judged against a moving target.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class Scenario:
    name: str
    # --- launch & in-space transport
    launch_usd_kg: float          # to a reference 400 km LEO
    isp_ep: float                 # electric propulsion specific impulse [s]
    isp_chem: float               # in-space chemical stage [s]
    ep_tug_overhead: float        # SEP tug amortisation, fraction of launched cost
    ep_accel: float               # SEP tug transfer acceleration [m/s^2]
    # --- space hardware
    hw_usd_kg: float              # manufacturing cost of dry space hardware
    pv_w_kg: float                # array specific power, BOL, 1 AU
    pv_eff: float                 # array efficiency (sets area / drag / SRP)
    pmad_kg_kw: float             # power management & distribution
    tx_kg_kw: float               # microwave DC->RF electronics + their thermal control
    ant_kg_m2: float              # transmitting aperture areal density
    laser_kg_kw: float            # laser transmitter incl. thermal
    struct_frac: float            # structure, ADCS, avionics as fraction of the above
    battery_wh_kg: float          # space battery pack (in-space firm power)
    # --- efficiencies
    eta_pmad: float
    eta_dcrf: float               # microwave DC->RF
    eta_rfdc: float               # rectenna RF->DC
    eta_laser: float              # laser wall-plug
    eta_laser_pv: float           # monochromatic PV receiver
    eta_grid: float               # ground power conditioning & grid interface
    eta_batt_rt: float            # battery round trip
    # --- ground segment
    rect_usd_m2: float            # rectenna incl. land and structure
    grid_usd_kw: float            # substation / grid connection per kW peak
    laser_rx_usd_kw: float        # laser receiver per kW peak
    site_fixed_usd: float         # per receiving site (control, safety zone, permits)
    # --- operations & finance
    om_frac: float                # annual O&M as fraction of capex
    life_yr: float                # hardware life
    discount: float               # real discount rate
    availability: float           # operational availability
    base_degr: float              # non-radiation array degradation per year
    module_m2: float              # structural module area (lost per >10 cm impact)
    # --- terrestrial counterfactual (same year)
    pv_ground_usd_w: float        # utility tracking PV, installed, per W_dc
    batt_ground_usd_kwh: float    # 4-h battery system, installed
    ground_wacc: float = 0.07

    def to_dict(self):
        return asdict(self)

    def with_(self, **kw) -> "Scenario":
        return replace(self, **kw)


TODAY = Scenario(
    name="today", launch_usd_kg=2500.0, isp_ep=2000.0, isp_chem=380.0, ep_tug_overhead=0.25,
    ep_accel=4e-4, hw_usd_kg=1500.0, pv_w_kg=120.0, pv_eff=0.30, pmad_kg_kw=1.0,
    tx_kg_kw=2.0, ant_kg_m2=2.0, laser_kg_kw=5.0, struct_frac=0.20, battery_wh_kg=150.0,
    eta_pmad=0.93, eta_dcrf=0.65, eta_rfdc=0.75, eta_laser=0.40, eta_laser_pv=0.45,
    eta_grid=0.97, eta_batt_rt=0.90, rect_usd_m2=30.0, grid_usd_kw=150.0,
    laser_rx_usd_kw=1000.0, site_fixed_usd=50e6, om_frac=0.02, life_yr=15.0, discount=0.07,
    availability=0.95, base_degr=0.005, module_m2=2500.0, pv_ground_usd_w=0.75,
    batt_ground_usd_kwh=200.0)

STARSHIP = Scenario(
    name="starship", launch_usd_kg=300.0, isp_ep=2500.0, isp_chem=380.0, ep_tug_overhead=0.15,
    ep_accel=6e-4, hw_usd_kg=600.0, pv_w_kg=300.0, pv_eff=0.32, pmad_kg_kw=0.6,
    tx_kg_kw=1.0, ant_kg_m2=1.0, laser_kg_kw=3.0, struct_frac=0.15, battery_wh_kg=250.0,
    eta_pmad=0.95, eta_dcrf=0.75, eta_rfdc=0.80, eta_laser=0.50, eta_laser_pv=0.55,
    eta_grid=0.97, eta_batt_rt=0.92, rect_usd_m2=20.0, grid_usd_kw=120.0,
    laser_rx_usd_kw=600.0, site_fixed_usd=30e6, om_frac=0.015, life_yr=20.0, discount=0.07,
    availability=0.96, base_degr=0.004, module_m2=2500.0, pv_ground_usd_w=0.55,
    batt_ground_usd_kwh=130.0)

ADVANCED = Scenario(
    name="advanced", launch_usd_kg=50.0, isp_ep=3000.0, isp_chem=380.0, ep_tug_overhead=0.10,
    ep_accel=1e-3, hw_usd_kg=200.0, pv_w_kg=1000.0, pv_eff=0.35, pmad_kg_kw=0.3,
    tx_kg_kw=0.5, ant_kg_m2=0.5, laser_kg_kw=1.5, struct_frac=0.12, battery_wh_kg=300.0,
    eta_pmad=0.96, eta_dcrf=0.85, eta_rfdc=0.85, eta_laser=0.60, eta_laser_pv=0.60,
    eta_grid=0.97, eta_batt_rt=0.94, rect_usd_m2=15.0, grid_usd_kw=100.0,
    laser_rx_usd_kw=300.0, site_fixed_usd=20e6, om_frac=0.01, life_yr=30.0, discount=0.07,
    availability=0.97, base_degr=0.003, module_m2=2500.0, pv_ground_usd_w=0.40,
    batt_ground_usd_kwh=90.0)

SCENARIOS = {s.name: s for s in (TODAY, STARSHIP, ADVANCED)}

# Exogenous uncertainty. "ln": log-normal sigma about the central value (multiplicative);
# "logit": normal sigma on the logit of an efficiency; "abs": additive normal sigma.
SIGMA = {
    "launch_usd_kg": ("ln", 0.5), "hw_usd_kg": ("ln", 0.5), "pv_w_kg": ("ln", 0.3),
    "tx_kg_kw": ("ln", 0.4), "ant_kg_m2": ("ln", 0.4), "laser_kg_kw": ("ln", 0.4),
    "pmad_kg_kw": ("ln", 0.3), "rect_usd_m2": ("ln", 0.4), "laser_rx_usd_kw": ("ln", 0.4),
    "site_fixed_usd": ("ln", 0.5), "om_frac": ("ln", 0.4), "ep_tug_overhead": ("ln", 0.5),
    "eta_dcrf": ("logit", 0.35), "eta_rfdc": ("logit", 0.35), "eta_laser": ("logit", 0.3),
    "eta_laser_pv": ("logit", 0.3), "eta_pmad": ("logit", 0.3),
    "life_yr": ("abs", 3.0), "base_degr": ("ln", 0.4),
    # multipliers on physics sub-models (central value 1)
    "rad_mult_p": ("ln", 0.8), "rad_mult_e": ("ln", 0.6), "rad_mult_sep": ("ln", 0.5),
    "atm_mult": ("ln", 0.6), "debris_mult": ("ln", 0.8),
    "pv_ground_usd_w": ("ln", 0.25), "batt_ground_usd_kwh": ("ln", 0.3),
}
MULTIPLIERS = ("rad_mult_p", "rad_mult_e", "rad_mult_sep", "atm_mult", "debris_mult")
