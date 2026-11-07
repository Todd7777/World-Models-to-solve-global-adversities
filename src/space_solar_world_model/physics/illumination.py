"""Annual illumination statistics for any placement.

The statistics that matter to a solar power system are:

* mean irradiance factor  <nu(t) * (1 AU / r(t))^2>, where nu is the visible fraction of
  the solar disk (so the mean flux on a Sun-pointing array is ``TSI * factor``);
* eclipse statistics (longest eclipse, events per year, worst per-orbit eclipse
  fraction) that size batteries and drive thermal cycling.

Circular orbits use the closed-form beta-angle model with J2 nodal precession;
eccentric orbits are integrated numerically with the conical shadow. Sun-Earth L1/L2
and lunar sites use documented literature values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..constants import AU, DAY_S, DEG, MU_EARTH, R_EARTH, SE_L1_DISTANCE
from .orbits import Orbit, j2_secular_rates, propagate, sso_inclination
from .shadow import beta_angle, eclipse_fraction_circular, illumination_fraction
from .sun import EPOCH_2030, sun_ecliptic, sun_position_eci, sun_ra_dec


@dataclass
class IlluminationStats:
    sunlit_fraction: float          # time-mean visible fraction of the solar disk
    irradiance_factor: float        # time-mean nu * (1 AU / r)^2
    max_eclipse_min: float          # longest single eclipse [min]
    eclipse_season_days: float      # days per year containing an eclipse
    eclipses_per_year: float        # eclipse events per year (thermal / battery cycles)
    worst_eclipse_fraction: float   # largest per-orbit eclipse fraction during the year
    period_min: float               # orbital period (NaN for non-orbits)
    method: str
    mean_abs_cos_beta: float = 2.0 / 3.1415926   # drives drag on a Sun-tracking array

    def to_dict(self):
        return asdict(self)


def circular_illumination(alt_km: float, inc_deg: float | None = None, ltan_hours: float | None = None,
                          days: int = 365, raan_phases: int = 8,
                          epoch: float = EPOCH_2030) -> IlluminationStats:
    """Annual statistics for a circular orbit (cylindrical shadow, J2 nodal precession).

    Pass ``ltan_hours`` for a sun-synchronous orbit (inclination is then derived);
    otherwise results are averaged over ``raan_phases`` initial node positions so they
    do not depend on an arbitrary launch date.
    """
    a = R_EARTH + alt_km * 1e3
    t = epoch + np.arange(days) + 0.5
    lam, _, eps, r_au = sun_ecliptic(t)
    period = 2.0 * np.pi * np.sqrt(a ** 3 / MU_EARTH)
    if ltan_hours is not None:
        inc = sso_inclination(a)
        ra0, _ = sun_ra_dec(epoch)
        raan0 = np.array([float(ra0) + (ltan_hours - 12.0) * 15.0 * DEG])
        raan_dot = 2.0 * np.pi / (365.24219 * DAY_S)
    else:
        inc = inc_deg * DEG
        raan0 = np.linspace(0.0, 2.0 * np.pi, raan_phases, endpoint=False)
        raan_dot = j2_secular_rates(a, 0.0, inc)[0]
    raan = raan0[:, None] + raan_dot * (t[None, :] - epoch) * DAY_S       # [phases, days]
    beta = beta_angle(inc, raan, lam[None, :], eps[None, :])
    fe = eclipse_fraction_circular(a, beta)                                 # [phases, days]
    sunlit = 1.0 - fe
    irr = sunlit / r_au[None, :] ** 2
    orbits_per_day = DAY_S / period
    return IlluminationStats(
        sunlit_fraction=float(sunlit.mean()),
        irradiance_factor=float(irr.mean()),
        max_eclipse_min=float(fe.max(axis=1).mean() * period / 60.0),
        eclipse_season_days=float((fe > 0).sum(axis=1).mean() * 365.25 / days),
        eclipses_per_year=float((fe > 0).sum(axis=1).mean() * orbits_per_day * 365.25 / days),
        worst_eclipse_fraction=float(fe.max(axis=1).mean()),
        period_min=float(period / 60.0),
        method="analytic-cylindrical",
        mean_abs_cos_beta=float(np.abs(np.cos(beta)).mean()),
    )


def numeric_illumination(orbits: list[Orbit] | Orbit, days: int = 365, day_step: int = 1,
                         samples_per_orbit: int = 360, epoch: float = EPOCH_2030) -> IlluminationStats:
    """Annual statistics by direct integration with the conical (umbra+penumbra) shadow.

    For every sampled day one full orbit is sampled uniformly in time. A list of orbits
    (e.g. several RAAN phases) is averaged.
    """
    orbits = [orbits] if isinstance(orbits, Orbit) else list(orbits)
    day_idx = np.arange(0, days, day_step)
    stats = []
    for orb in orbits:
        P = orb.period_s
        tau = np.linspace(0.0, P, samples_per_orbit, endpoint=False) / DAY_S
        t = epoch + day_idx[:, None] + 0.5 + tau[None, :]                  # [days, samples]
        r_sat = propagate(orb, t)
        r_sun = sun_position_eci(t)
        nu = illumination_fraction(r_sat, r_sun)
        _, _, _, r_au = sun_ecliptic(t)
        fe = 1.0 - nu.mean(axis=1)                                          # per-orbit eclipse frac
        stats.append(dict(
            sunlit=nu.mean(), irr=(nu / r_au ** 2).mean(),
            max_ecl=_longest_run(nu < 0.5, P / samples_per_orbit) / 60.0,
            season=(fe > 1e-4).mean() * 365.25,
            events=(fe > 1e-4).mean() * 365.25 * DAY_S / P,
            worst=fe.max(), P=P / 60.0))
    m = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
    return IlluminationStats(m["sunlit"], m["irr"], m["max_ecl"], m["season"], m["events"],
                             m["worst"], m["P"], "numeric-conical")


def _longest_run(mask: np.ndarray, dt_s: float) -> float:
    """Longest contiguous True run (wrapping within each row) times dt, max over rows."""
    best = 0
    for row in np.atleast_2d(mask):
        if row.all():
            return row.size * dt_s
        if not row.any():
            continue
        r = np.roll(row, -int(np.argmin(row)))      # start at a False to unwrap
        runs, cur = 0, 0
        for v in r:
            cur = cur + 1 if v else 0
            runs = max(runs, cur)
        best = max(best, runs)
    return best * dt_s


def lagrange_illumination(point: str = "L1") -> IlluminationStats:
    """Sun-Earth L1 / L2 (large halo orbit, so Earth's penumbra is avoided)."""
    if point == "L1":
        r = 1.0 - SE_L1_DISTANCE / AU
    elif point == "L2":
        r = 1.0 + SE_L1_DISTANCE / AU
    else:
        raise ValueError(point)
    return IlluminationStats(1.0, 1.0 / r ** 2, 0.0, 0.0, 0.0, 0.0, float("nan"), "lagrange-halo")


# Lunar sites (literature values; see docs). South-pole "peaks of near-eternal light"
# reach ~85-90% illumination a few metres above the surface (Bussey et al. 2010;
# Mazarico et al. 2011) but still have multi-day dark spells; NRHO (the Gateway orbit)
# is designed to avoid lunar eclipses and sees only brief, rare Earth eclipses.
LUNAR_SITES = {
    # worst_eclipse_fraction here is the dark share of the worst lunation (29.5 d) or orbit
    "moon_south_pole_pel": IlluminationStats(0.86, 0.86, 4.0 * 1440.0, 60.0, 8.0, 0.20,
                                             float("nan"), "literature"),
    "moon_equator": IlluminationStats(0.50, 0.50, 14.77 * 1440.0, 365.0, 12.4, 0.5,
                                      float("nan"), "geometry"),
    "lunar_nrho": IlluminationStats(0.998, 0.998, 80.0, 2.0, 2.0, 80.0 / (6.56 * 1440.0),
                                    6.56 * 1440.0, "literature"),
}
