"""Novel CDR path and peak-cumulative accounting."""
import numpy as np

from carbon_world_model.climate import cdr


def test_cdr_path_starts_at_the_measured_level_and_saturates():
    years = np.arange(2023, 2101)
    x = cdr.novel_cdr_path(years)
    np.testing.assert_allclose(x[years == 2025], cdr.CDR_START_GT)
    assert np.all(np.diff(x) > 0)
    assert x[-1] < cdr.CDR_CEILINGS["central"] and x[-1] > 0.99 * cdr.CDR_CEILINGS["central"]
    # early growth is the measured rate
    np.testing.assert_allclose(x[years == 2026] / x[years == 2025], np.exp(cdr.CDR_GROWTH), rtol=1e-3)
    # near the scenario level around 2050, as the source describes
    assert 0.75 * 3.5 < x[years == 2050] < 3.5


def test_ceilings_order_the_paths():
    years = np.arange(2023, 2101)
    lo, mid, hi = (cdr.novel_cdr_path(years, cdr.CDR_CEILINGS[k]) for k in ("low", "central", "high"))
    assert np.all(lo <= mid + 1e-12) and np.all(mid <= hi + 1e-12)


def test_peak_cumulative_and_net_zero_year():
    years = np.arange(2023, 2033)
    net = np.array([5, 4, 3, 2, 1, 0, -1, -2, -3, -4], dtype=float)
    peak, reached = cdr.peak_cumulative(net)
    assert peak == 15.0 and reached
    assert cdr.net_zero_year(years, net) == 2028
    still_rising = np.array([5, 4, 3, 2, 1], dtype=float)
    peak, reached = cdr.peak_cumulative(still_rising)
    assert peak == 15.0 and not reached
    assert cdr.net_zero_year(years[:5], still_rising) is None


def test_supply_respects_price_and_land_caps():
    s0 = cdr.cdr_supply(5.0)
    assert all(v == 0.0 for v in s0.values())               # below every cost range
    full = cdr.cdr_supply(1e9, land_cap=1e9)
    for k, (clo, chi, plo, phi, _) in cdr.CDR_METHODS.items():
        assert abs(full[k] - 0.5 * (plo + phi)) < 1e-12
    s = cdr.cdr_supply(1e9, land_cap=2.4)
    assert abs(s["beccs"] + s["biochar"] - 2.4) < 1e-12     # land caps biomass methods together
    assert s["daccs"] == full["daccs"]                       # DACCS is not land-limited
    caps = [cdr.cdr_ceiling(p) for p in (50, 100, 200, 300, 400)]
    assert all(a <= b for a, b in zip(caps, caps[1:]))
    lands = [cdr.cdr_ceiling(200, l) for l in (1.1, 2.4, 5.3)]
    assert all(a <= b for a, b in zip(lands, lands[1:]))


def test_100_dollar_cap_lands_near_the_scenario_level():
    assert 2.8 < cdr.cdr_ceiling(100.0) < 3.6
    assert cdr.cdr_supply(100.0)["daccs"] == 0.0             # DACCS starts at USD 100/t


def test_spending_and_energy_are_consistent():
    r = cdr.cdr_spending_and_energy(200.0)
    s = r["supply_gt"]
    assert r["daccs_ej"][0] == 4.0 * s["daccs"] and r["daccs_ej"][1] == 10.0 * s["daccs"]
    assert 0 < r["avg_cost_usd_t"] <= 200.0


def test_energy_limit_only_cuts_direct_air_capture():
    years = np.arange(2023, 2101)
    lc = np.full(len(years), 300.0)                     # EJ/yr of low-carbon energy
    free = cdr.novel_cdr_path(years, cdr.cdr_ceiling(200.0))
    big, dac_big = cdr.energy_limited_cdr_path(years, 200.0, 2.4, lc, frac=1e6)
    np.testing.assert_allclose(big, free)               # no binding limit -> unchanged
    zero, dac0 = cdr.energy_limited_cdr_path(years, 200.0, 2.4, lc, frac=0.0)
    s = cdr.cdr_supply(200.0)
    np.testing.assert_allclose(zero, free * (1 - s["daccs"] / sum(s.values())))
    assert np.all(dac0 == 0.0)
    mid, dac = cdr.energy_limited_cdr_path(years, 200.0, 2.4, lc, frac=0.10)
    assert np.all(mid <= free + 1e-12) and np.all(mid >= zero - 1e-12)
    assert np.all(dac * cdr.DACCS_EJ_PER_GT_CENTRAL <= 0.10 * lc + 1e-9)
    same, _ = cdr.energy_limited_cdr_path(years, 100.0, 2.4, lc, frac=0.0)
    np.testing.assert_allclose(same, cdr.novel_cdr_path(years, cdr.cdr_ceiling(100.0)))  # no DACCS at USD 100
