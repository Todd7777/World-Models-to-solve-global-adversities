"""Low-precision solar ephemeris and Earth orientation.

Implements the Astronomical Almanac low-precision solar coordinates (accuracy ~0.01 deg
for 1950-2050), which is far more precise than anything downstream needs, plus GMST for
rotating ground sites into the inertial frame.

Time is measured throughout the package in *days since J2000.0* (2000-01-01 12:00 TT;
the TT/UT distinction is irrelevant at this precision).
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

from ..constants import AU, DEG, TSI_1AU

J2000_JD = 2451545.0
_J2000 = _dt.datetime(2000, 1, 1, 12, 0, 0)

# Default simulation epoch: start of 2030, a representative deployment year.
EPOCH_2030 = (_dt.datetime(2030, 1, 1) - _J2000).total_seconds() / 86400.0


def days_since_j2000(when: _dt.datetime) -> float:
    """Days since J2000.0 for a naive UTC datetime."""
    return (when - _J2000).total_seconds() / 86400.0


def sun_ecliptic(t_days):
    """Return (ecliptic longitude [rad], mean anomaly [rad], obliquity [rad], distance [AU])."""
    n = np.asarray(t_days, dtype=float)
    L = (280.460 + 0.9856474 * n) * DEG
    g = (357.528 + 0.9856003 * n) * DEG
    lam = L + (1.915 * np.sin(g) + 0.020 * np.sin(2.0 * g)) * DEG
    eps = (23.439 - 4.0e-7 * n) * DEG
    r_au = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2.0 * g)
    return lam, g, eps, r_au


def sun_unit_eci(t_days) -> np.ndarray:
    """Unit vector from Earth to Sun in the (mean-of-date) equatorial inertial frame."""
    lam, _, eps, _ = sun_ecliptic(t_days)
    return np.stack([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)], axis=-1)


def sun_position_eci(t_days) -> np.ndarray:
    """Earth-to-Sun position vector [m]."""
    _, _, _, r_au = sun_ecliptic(t_days)
    return sun_unit_eci(t_days) * (np.asarray(r_au) * AU)[..., None]


def sun_ra_dec(t_days):
    """Right ascension and declination of the Sun [rad]."""
    s = sun_unit_eci(t_days)
    return np.arctan2(s[..., 1], s[..., 0]), np.arcsin(np.clip(s[..., 2], -1.0, 1.0))


def irradiance_at_earth(t_days):
    """Solar irradiance at Earth's distance [W m^-2] (varies +/-3.3% over the year)."""
    _, _, _, r_au = sun_ecliptic(t_days)
    return TSI_1AU / np.asarray(r_au) ** 2


def gmst_rad(t_days):
    """Greenwich mean sidereal time [rad] (IAU 1982, linear term; ample for this use)."""
    n = np.asarray(t_days, dtype=float)
    return np.mod((280.46061837 + 360.98564736629 * n) * DEG, 2.0 * np.pi)


def sun_elevation_at_site(lat_deg, lon_deg, t_days):
    """Elevation of the Sun above the horizon at a ground site [rad] (spherical Earth)."""
    ra, dec = sun_ra_dec(t_days)
    lst = gmst_rad(t_days) + lon_deg * DEG
    ha = lst - ra
    lat = lat_deg * DEG
    sin_el = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)
    return np.arcsin(np.clip(sin_el, -1.0, 1.0))
