"""Physics layer reproduces textbook and published reference values."""
import numpy as np
import pytest

from space_solar_world_model.constants import DEG, R_EARTH, R_GEO
from space_solar_world_model.physics import beaming as B
from space_solar_world_model.physics import environment as ENV
from space_solar_world_model.physics import radiation as RAD
from space_solar_world_model.physics.illumination import (circular_illumination, lagrange_illumination,
                                                          numeric_illumination)
from space_solar_world_model.physics.orbits import (Orbit, geo_look_angles, propagate, slant_range,
                                                    sso_inclination)
from space_solar_world_model.physics.shadow import eclipse_fraction_circular, illumination_fraction
from space_solar_world_model.physics.sun import EPOCH_2030, sun_ra_dec, irradiance_at_earth


def test_sun_declination_and_irradiance():
    # June solstice 2030 ~ day 171 after 1 Jan; declination ~ +23.44 deg
    _, dec = sun_ra_dec(EPOCH_2030 + 171.0)
    assert abs(np.degrees(dec) - 23.44) < 0.1
    # perihelion (early Jan) is brightest, aphelion (early Jul) dimmest, +/-3.3%
    jan, jul = irradiance_at_earth(EPOCH_2030 + 3.0), irradiance_at_earth(EPOCH_2030 + 184.0)
    assert 1400 < jan < 1415 and 1312 < jul < 1322


def test_sun_synchronous_inclination():
    assert abs(np.degrees(sso_inclination(R_EARTH + 800e3)) - 98.6) < 0.05
    assert abs(np.degrees(sso_inclination(R_EARTH + 500e3)) - 97.4) < 0.05


def test_eclipse_reference_values():
    # ISS at beta = 0: ~38.7% of a 92.7-min orbit in shadow (~36 min)
    assert abs(eclipse_fraction_circular(R_EARTH + 420e3, 0.0) - 0.3875) < 1e-3
    # GEO at equinox: 69.4 min cylindrical shadow
    assert abs(eclipse_fraction_circular(R_GEO, 0.0) * 1436.07 - 69.4) < 0.1
    # no eclipse above beta*
    assert eclipse_fraction_circular(R_EARTH + 420e3, 75 * DEG) == 0.0


def test_geo_eclipse_season():
    s = circular_illumination(35786.0, 0.0)
    assert 0.989 < s.sunlit_fraction < 0.992              # ~99.1% sunlit
    assert 69.0 < s.max_eclipse_min < 70.0
    assert 85 < s.eclipse_season_days < 95                # two ~45-day seasons


def test_analytic_matches_numeric_conical():
    a = circular_illumination(420.0, 51.6)
    n = numeric_illumination([Orbit.circular(420.0, 51.6, raan_deg=r) for r in np.arange(0, 360, 45)],
                             day_step=5, samples_per_orbit=720)
    assert abs(a.sunlit_fraction - n.sunlit_fraction) < 2e-3
    d_a = circular_illumination(800.0, ltan_hours=18.0)
    d_n = numeric_illumination(Orbit.sun_synchronous(800.0, 18.0), day_step=2, samples_per_orbit=720)
    assert abs(d_a.sunlit_fraction - d_n.sunlit_fraction) < 2e-3


def test_dawn_dusk_becomes_eclipse_free_with_altitude():
    low = circular_illumination(650.0, ltan_hours=18.0)
    high = circular_illumination(1500.0, ltan_hours=18.0)
    assert 0.95 < low.sunlit_fraction < 0.97 and low.max_eclipse_min < 25
    assert high.sunlit_fraction == 1.0 and high.max_eclipse_min == 0.0


def test_conical_shadow_cases():
    sun = np.array([1.496e11, 0.0, 0.0])
    assert illumination_fraction(np.array([0.0, 7e6, 0.0]), sun) == 1.0         # beside Earth
    assert illumination_fraction(np.array([-7e6, 0.0, 0.0]), sun) == 0.0        # behind Earth
    # Sun-Earth L2 point (1.5e9 m anti-sunward): annular eclipse blocks most of the Sun
    l2 = illumination_fraction(np.array([-1.5e9, 0.0, 0.0]), sun)
    assert 0.05 < l2 < 0.3
    assert lagrange_illumination("L1").irradiance_factor > 1.0


def test_critical_inclination_orbits_and_geo_geometry():
    t = EPOCH_2030 + np.linspace(0, 1, 500)
    tundra = Orbit.critical_heo("tundra", 20.0)
    assert abs(tundra.period_s / 3600 - 23.934) < 0.01
    r = np.linalg.norm(propagate(tundra, t), axis=-1)
    assert abs(r.max() / tundra.a - (1 + tundra.e)) < 1e-3
    el, rng = geo_look_angles(36.17)
    assert abs(np.degrees(el) - 48.0) < 0.2 and abs(rng / 1e3 - 37206) < 5
    assert abs(slant_range(1000e3, np.pi / 2) - 1000e3) < 1.0


def test_radiation_anchors_and_structure():
    for row in RAD.anchor_table():
        assert 1.0 / row["band"] < row["ratio"] < row["band"], row["label"]
    eq = {h: RAD.orbit_fluence_rate(h, 0.0)["total"] for h in (500, 3500, 12000, 22000, 35786)}
    assert eq[3500] > 100 * eq[500]                 # inner belt >> low LEO
    assert eq[12000] < eq[3500] and eq[12000] < eq[22000]   # slot between belts
    assert eq[35786] < eq[22000]                    # GEO below outer-belt heart
    # Spectrolab-like triple-junction retention
    assert abs(RAD.remaining_power_factor(1e15) - 0.87) < 0.02
    assert abs(RAD.remaining_power_factor(1e14) - 0.955) < 0.01


def test_spiral_transfer_dose_is_plausible():
    slow = RAD.spiral_transfer_fluence(R_EARTH + 400e3, R_GEO, 28.5, 0.0, 5900 / (200 * 86400), 5900)
    fast = RAD.spiral_transfer_fluence(R_EARTH + 400e3, R_GEO, 28.5, 0.0, 5900 / (30 * 86400), 5900)
    assert abs(slow["days"] - 200) < 2
    r_slow, r_fast = RAD.remaining_power_factor(slow["total"]), RAD.remaining_power_factor(fast["total"])
    # LEO-start electric orbit raising costs ~10-20% for a 200-day spiral (cf. 15-25% reported)
    assert 0.80 < r_slow < 0.90 and r_fast > r_slow


def test_drag_and_debris_shape():
    assert ENV.atmospheric_density(400.0) == pytest.approx(3.725e-12, rel=1e-3)
    c = [ENV.drag_dv_coefficient(h) for h in (400, 600, 800, 1200)]
    assert c[0] > c[1] > c[2] > c[3] > 0
    f = {h: float(ENV.debris_flux(h)) for h in (400, 850, 1200, 20000, 35786)}
    assert f[850] > f[400] and f[850] > f[1200] and f[20000] < 1e-8 and f[35786] < 1e-8


def test_microwave_beam_reference_system():
    # 1979 DOE/NASA reference: 2.45 GHz, 1 km transmitter, ~10 km rectenna from GEO
    tech = B.MICROWAVE_2G45
    at = np.pi * 500.0 ** 2
    ar = np.pi * 5000.0 ** 2
    eta = B.capture_efficiency(tech, at, ar, 36000e3)
    assert eta > 0.95
    # 230 W/m^2 beam-centre cap allows ~5 GW at 2.45 GHz but only ~1 GW at 5.8 GHz
    p245 = B.max_power_for_peak(tech, at, 36000e3)
    p58 = B.max_power_for_peak(B.MICROWAVE_5G8, at, 36000e3)
    assert 4e9 < p245 < 7e9 and 0.7e9 < p58 < 1.3e9
    # Goubau equals the Gaussian-beam peak-intensity relation
    assert B.peak_intensity(1e9, at, tech.wavelength_m, 36000e3) == pytest.approx(
        1e9 * at / (tech.wavelength_m * 36000e3) ** 2)
