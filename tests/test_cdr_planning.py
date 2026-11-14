"""Removals as a planning lever (plan/cdr_planning.py)."""
import numpy as np

from carbon_world_model.climate import cdr as CDR
from carbon_world_model.plan import cdr_planning as P
from carbon_world_model.plan.scenario import build_scenario


def test_supply_bins_match_the_method_table():
    price, cap, dac = P.supply_bins(2.4)
    assert np.all(np.diff(price) >= 0)
    np.testing.assert_allclose(cap.sum(), CDR.cdr_ceiling(1e9, 2.4))
    np.testing.assert_allclose(cap[dac].sum(), CDR.cdr_supply(1e9, 2.4)["daccs"])
    # land goes to the cheapest biomass first: total biomass equals the cap
    full = P.supply_bins(1e9)[1].sum()
    np.testing.assert_allclose(full - cap.sum(), sum(
        0.5 * (v[2] + v[3]) for v in CDR.CDR_METHODS.values() if v[4]) - 2.4)


def test_capacity_matches_the_accounting_path():
    years = np.arange(2023, 2101)
    lc = np.linspace(100.0, 400.0, len(years))
    capt = np.asarray(P.capacity(years, lc, P.supply_bins()))
    ref, _ = CDR.energy_limited_cdr_path(years, 1e9, CDR.LAND_CAPS["central"], lc)
    np.testing.assert_allclose(capt.sum(axis=1), ref, rtol=1e-5, atol=1e-6)


def test_spending_fills_the_cheapest_bins_first():
    years = np.arange(2023, 2101)
    bins = P.supply_bins()
    capt = np.asarray(P.capacity(years, np.full(len(years), 300.0), bins))
    k = int(np.where(years == 2060)[0][0])
    tot = capt[k].sum()
    r = np.linspace(0.0, tot, 50)
    s = np.array([float(P.spending_bn(np.full(len(years), x), capt, bins[0])[k]) for x in r])
    assert s[0] == 0.0
    assert np.all(np.diff(s) >= -1e-9)
    assert np.all(np.diff(s, 2) >= -1e-6)                       # convex: rising marginal cost
    np.testing.assert_allclose(s[-1], float((capt[k] * bins[0]).sum()), rtol=1e-6)


def test_planner_respects_the_spending_allowance(panel, cfg, wm, params):
    scen = build_scenario(panel, cfg, end_year=2060, ssp="SSP2")
    off = P.plan_with_removals(wm, params, scen, budget_gt=900.0, spend_cap_tn=0.0, steps=60)
    assert np.all(off.removals == 0.0) and off.spend_tn == 0.0
    cap = P.plan_with_removals(wm, params, scen, budget_gt=900.0, spend_cap_tn=2.0, steps=200)
    assert cap.spend_tn <= 2.0 * 1.05
    assert np.all(cap.removals <= cap.capacity + 1e-6)
