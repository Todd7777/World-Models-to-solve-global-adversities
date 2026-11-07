"""Structural model behaves causally and conserves energy; ground baseline is calibrated."""
import numpy as np
import pytest

from space_solar_world_model import planner
from space_solar_world_model.economics import transport as TR
from space_solar_world_model.economics.ground import firm_supply
from space_solar_world_model.economics.system import shared_availability
from space_solar_world_model.sites import SITES


def test_transport_costs_rise_with_energy():
    m = [TR.chemical_mass_multiplier(dv, 380.0) for dv in (500, 2000, 4200)]
    assert 1.0 < m[0] < m[1] < m[2] < 6.0
    assert TR.ep_mass_multiplier(5900, 2500) < m[2]          # SEP is cheaper in mass to GEO
    assert TR.launch_inclination_multiplier(98.6) > TR.launch_inclination_multiplier(53.0) > 1.0


def test_lcoe_monotone_in_launch_cost_and_degradation(world, geo_plan):
    a = geo_plan.action
    lc = [float(world.outcome(a, dict(world.central_u(), launch_usd_kg=x))["lcoe"])
          for x in (50.0, 300.0, 2000.0)]
    assert lc[0] < lc[1] < lc[2]
    base = float(world.outcome(a)["lcoe"])
    worse = float(world.outcome(a, dict(world.central_u(), rad_mult_e=5.0, rad_mult_sep=5.0))["lcoe"])
    no_rad = float(world.outcome(a, do={"fluence_rate": 0.0, "transfer_fluence": 0.0})["lcoe"])
    assert no_rad < base < worse


def test_geo_design_resembles_reference_systems(geo_plan):
    o = geo_plan.outcome
    d = geo_plan.action.design_dict()
    assert 500 < d["d_t"] < 3000                          # ~1 km class transmitter
    assert 0.5e9 < float(o["P_delivered_bol"]) < 6e9       # GW-class delivery
    assert 10e6 < float(o["A_r_ground"][0]) < 250e6        # multi-km rectenna
    assert 0.5 < float(o["link_efficiency"]) < 0.85        # end-to-end DC->grid
    assert 50 < geo_plan.lcoe < 1000


def test_optimiser_beats_naive_design(world, geo_plan):
    naive = geo_plan.action.with_design(P_sat=4e9, d_t=800.0, d_r=6000.0)
    assert geo_plan.lcoe <= float(world.outcome(naive)["lcoe"]) + 1e-6


def test_shared_network_cannot_deliver_more_than_satellites():
    q = np.full((24, 10), 0.4)                    # MEO-like: every satellite sees every site
    for T in (2.0, 5.0, 20.0):
        served = shared_availability(np.float64(T), q, np).sum()
        assert served <= T * 1.0 + 1e-9
    small = np.full((3, 10), 0.01)                # LEO-like: sparse visibility, no conflict
    assert np.allclose(shared_availability(np.float64(10.0), small, np), 1 - np.exp(-0.1))


def test_beamed_power_never_exceeds_transmitted(world):
    r = planner.plan_beam(world, "meo_20200_55", "Las Vegas", "mw2.45", site_options=(24,))
    o = r.outcome
    T = r.action.design_dict()["T"]
    p_rf = r.action.design_dict()["P_sat"] * world.central_u()["eta_pmad"] * world.central_u()["eta_dcrf"]
    assert float(o["P_delivered_bol"]) <= T * p_rf


def test_ground_baseline_matches_ember_las_vegas(world):
    u = dict(world.central_u(), pv_ground_usd_w=0.75, batt_ground_usd_kwh=200.0)
    r = firm_supply(SITES["Las Vegas"], u)
    assert 80 < r.lcoe < 130                     # Ember (2025): $104/MWh at 97% of 24/365
    assert 4 < r.pv_kw < 9 and 10 < r.batt_kwh < 25
    london = firm_supply(SITES["London"], u)
    assert london.lcoe > 2 * r.lcoe              # winter makes high-latitude firm solar dear


def test_in_space_dawn_dusk_beats_inclined_leo(world):
    dd = planner.in_space(world, "sso18_1400")
    leo = planner.in_space(world, "leo_1000_53")
    assert dd.lcoe < leo.lcoe
    assert float(dd.outcome["M_batt"]) == pytest.approx(0.0, abs=1e-3)   # eclipse-free


def test_transport_is_continuous_across_leo_meo_boundary():
    lo = TR.transfer_options("circular", 1999.0, 0.0)
    hi = TR.transfer_options("circular", 2001.0, 0.0)
    for k in ("chemical", "sep_tug"):
        assert abs(lo[k]["dv"] - hi[k]["dv"]) < 20.0
    # equatorial targets from a 28.5 deg site pay the plane change; 53 deg ones do not
    assert TR.transfer_options("circular", 1000.0, 0.0)["chemical"]["dv"] > 3000.0
    assert TR.transfer_options("circular", 1000.0, 53.0)["chemical"]["dv"] < 500.0
