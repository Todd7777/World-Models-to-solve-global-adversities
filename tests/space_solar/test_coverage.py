"""Link-coverage simulator: GEO continuity, LEO daytime bias, Poisson shortcut validity."""
from space_solar_world_model.physics.coverage import (geo_link, heo_link, l1_link,
                                                      shared_constellation_link,
                                                      simulate_walker_availability)
from space_solar_world_model.sites import SITES


def test_geo_link_is_nearly_continuous_with_midnight_eclipses():
    g = geo_link(SITES["Las Vegas"])
    assert 0.985 < g.availability() < 0.995
    prof = g.hourly_profile()
    # equinox months lose power only around local midnight
    assert prof[2, 0] < 0.8 and prof[2, 12] > 0.99 and prof[5].min() > 0.95


def test_l1_only_serves_daytime():
    s = l1_link(SITES["Las Vegas"])
    prof = s.hourly_profile()
    assert prof[:, 0].max() == 0.0 and prof[:, 12].min() > 0.9
    assert 0.25 < s.availability() < 0.45


def test_tundra_covers_high_latitude_site():
    t = heo_link("tundra", SITES["Tromso"])
    assert t.availability() > 0.95
    assert t.D.min() > 40000e3        # but from even farther away than GEO


def test_leo_constellation_poisson_vs_explicit_walker():
    site = SITES["Las Vegas"]
    ls = shared_constellation_link(1000.0, [site], inc_deg=53.0)
    s = ls.sites[0]
    for planes, per in [(10, 20)]:
        brute = simulate_walker_availability(1000.0, 53.0, planes, per, site, days=4)
        pois = s.availability(planes * per, shared=True)
        assert pois <= brute + 0.05 and abs(brute - pois) < 0.15
    # winter nights: satellites overhead at local midnight are in Earth's shadow
    # (in summer the Sun is only ~30 deg below the horizon at 36N, so 1,000 km
    # satellites north of the site stay lit all night)
    prof = s.hourly_profile(500, shared=True)
    assert prof[[11, 0, 1], 0].max() < 0.05 and prof[:, 12].mean() > 0.9
    assert prof[[5, 6], 0].mean() > 0.5
