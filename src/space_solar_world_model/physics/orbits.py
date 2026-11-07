"""Orbit definitions and propagation (Keplerian motion + secular J2 drift).

Secular J2 rates (Vallado, *Fundamentals of Astrodynamics*, Eq. 9-41):

    dRAAN/dt = -3/2 n J2 (R/p)^2 cos i
    dargp/dt =  3/4 n J2 (R/p)^2 (5 cos^2 i - 1)
    dM/dt    =  n [1 + 3/4 J2 (R/p)^2 sqrt(1-e^2) (3 cos^2 i - 1)]

Station-kept orbits whose ground track must stay fixed (GEO, Tundra, Molniya) are
propagated with their drift removed and mean motion locked to Earth's rotation,
which is what their operators enforce with station-keeping.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import (DAY_S, DEG, J2_EARTH, MU_EARTH, OMEGA_EARTH, R_EARTH, R_GEO,
                         SUN_SYNC_RATE)
from .sun import EPOCH_2030, gmst_rad, sun_ra_dec

SIDEREAL_DAY_S = 2.0 * np.pi / OMEGA_EARTH


def j2_secular_rates(a, e, inc):
    """Secular rates (dRAAN/dt, dargp/dt, dM/dt - n) [rad/s] due to J2."""
    a = np.asarray(a, float)
    n = np.sqrt(MU_EARTH / a ** 3)
    p = a * (1.0 - np.asarray(e) ** 2)
    k = n * J2_EARTH * (R_EARTH / p) ** 2
    ci = np.cos(inc)
    raan_dot = -1.5 * k * ci
    argp_dot = 0.75 * k * (5.0 * ci ** 2 - 1.0)
    m_dot = 0.75 * k * np.sqrt(1.0 - np.asarray(e) ** 2) * (3.0 * ci ** 2 - 1.0)
    return raan_dot, argp_dot, m_dot


def sso_inclination(a, e=0.0):
    """Inclination [rad] that makes the node precess with the mean Sun (sun-synchronous).

    Raises ValueError if no sun-synchronous solution exists (a above ~12,350 km).
    """
    a = float(a)
    n = np.sqrt(MU_EARTH / a ** 3)
    p = a * (1.0 - e ** 2)
    cos_i = -SUN_SYNC_RATE / (1.5 * n * J2_EARTH * (R_EARTH / p) ** 2)
    if abs(cos_i) > 1.0:
        raise ValueError(f"no sun-synchronous inclination for a={a/1e3:.0f} km")
    return float(np.arccos(cos_i))


def solve_kepler(M, e, iters: int = 20):
    """Eccentric anomaly from mean anomaly (vectorised Newton iteration)."""
    M = np.mod(np.asarray(M, float), 2.0 * np.pi)
    E = np.where(e < 0.8, M, np.pi * np.ones_like(M))
    for _ in range(iters):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


@dataclass(frozen=True)
class Orbit:
    """Classical orbital elements at ``epoch`` (days since J2000) plus a drift model."""

    a: float                      # semi-major axis [m]
    e: float = 0.0
    inc: float = 0.0              # [rad]
    raan: float = 0.0             # [rad]
    argp: float = 0.0             # [rad]
    m0: float = 0.0               # mean anomaly at epoch [rad]
    epoch: float = EPOCH_2030
    j2: bool = True               # apply secular J2 drift
    fixed_ground_track: bool = False   # station-kept repeat track: no drift, locked mean motion
    revs_per_sidereal_day: float | None = None

    # ---- derived quantities -------------------------------------------------
    @property
    def mean_motion(self) -> float:
        if self.revs_per_sidereal_day is not None:
            return self.revs_per_sidereal_day * OMEGA_EARTH
        return float(np.sqrt(MU_EARTH / self.a ** 3))

    @property
    def period_s(self) -> float:
        return 2.0 * np.pi / self.mean_motion

    @property
    def perigee_alt_km(self) -> float:
        return (self.a * (1.0 - self.e) - R_EARTH) / 1e3

    @property
    def apogee_alt_km(self) -> float:
        return (self.a * (1.0 + self.e) - R_EARTH) / 1e3

    # ---- constructors -------------------------------------------------------
    @staticmethod
    def circular(alt_km: float, inc_deg: float, raan_deg: float = 0.0, u0_deg: float = 0.0,
                 epoch: float = EPOCH_2030) -> "Orbit":
        return Orbit(a=R_EARTH + alt_km * 1e3, e=0.0, inc=inc_deg * DEG, raan=raan_deg * DEG,
                     argp=0.0, m0=u0_deg * DEG, epoch=epoch)

    @staticmethod
    def sun_synchronous(alt_km: float, ltan_hours: float, u0_deg: float = 0.0,
                        epoch: float = EPOCH_2030) -> "Orbit":
        """Circular sun-synchronous orbit with the given local time of ascending node.

        LTAN 06:00 or 18:00 is the *dawn-dusk* orbit that rides the terminator.
        """
        a = R_EARTH + alt_km * 1e3
        inc = sso_inclination(a)
        ra_sun, _ = sun_ra_dec(epoch)
        raan = float(ra_sun) + (ltan_hours - 12.0) * 15.0 * DEG
        return Orbit(a=a, e=0.0, inc=inc, raan=raan, argp=0.0, m0=u0_deg * DEG, epoch=epoch)

    @staticmethod
    def geostationary(lon_deg: float, epoch: float = EPOCH_2030) -> "Orbit":
        """Station-kept geostationary satellite fixed above longitude ``lon_deg``."""
        m0 = float(gmst_rad(epoch)) + lon_deg * DEG
        return Orbit(a=R_GEO, e=0.0, inc=0.0, raan=0.0, argp=0.0, m0=m0, epoch=epoch,
                     j2=False, fixed_ground_track=True, revs_per_sidereal_day=1.0)

    @staticmethod
    def critical_heo(kind: str, apogee_lon_deg: float, index: int = 0, n_sats: int = 1,
                     epoch: float = EPOCH_2030) -> "Orbit":
        """Tundra (1 rev/sidereal day) or Molniya (2 rev/day) orbit at 63.4 deg inclination.

        Apogee dwells over the northern hemisphere at ``apogee_lon_deg``; ``index`` of
        ``n_sats`` spaces the satellites evenly in time along one shared ground track.
        """
        if kind == "tundra":
            k, e = 1.0, 0.2668
        elif kind == "molniya":
            k, e = 2.0, 0.74
        else:
            raise ValueError(kind)
        n = k * OMEGA_EARTH
        a = (MU_EARTH / n ** 2) ** (1.0 / 3.0)
        t_apo = epoch + index * (SIDEREAL_DAY_S / n_sats) / DAY_S
        raan = float(gmst_rad(t_apo)) + apogee_lon_deg * DEG - 90.0 * DEG
        m0 = np.pi - 2.0 * np.pi * k * index / n_sats
        return Orbit(a=a, e=e, inc=63.4349 * DEG, raan=raan, argp=270.0 * DEG, m0=m0,
                     epoch=epoch, j2=False, fixed_ground_track=True, revs_per_sidereal_day=k)


def propagate(orbit: Orbit, t_days) -> np.ndarray:
    """Inertial (ECI) position [m] at times ``t_days`` (any shape) -> shape (..., 3)."""
    t = np.asarray(t_days, float)
    dt = (t - orbit.epoch) * DAY_S
    n = orbit.mean_motion
    if orbit.j2 and not orbit.fixed_ground_track:
        rd, wd, md = j2_secular_rates(orbit.a, orbit.e, orbit.inc)
    else:
        rd = wd = md = 0.0
    raan = orbit.raan + rd * dt
    argp = orbit.argp + wd * dt
    M = orbit.m0 + (n + md) * dt
    e = orbit.e
    if e == 0.0:
        nu = np.mod(M, 2.0 * np.pi)
        r = np.full_like(nu, orbit.a)
    else:
        E = solve_kepler(M, e)
        nu = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0), np.sqrt(1.0 - e) * np.cos(E / 2.0))
        r = orbit.a * (1.0 - e * np.cos(E))
    u = argp + nu
    ci, si = np.cos(orbit.inc), np.sin(orbit.inc)
    cO, sO = np.cos(raan), np.sin(raan)
    cu, su = np.cos(u), np.sin(u)
    x = r * (cO * cu - sO * su * ci)
    y = r * (sO * cu + cO * su * ci)
    z = r * (su * si)
    return np.stack([x, y, z], axis=-1)


def orbit_normal(orbit: Orbit, t_days) -> np.ndarray:
    """Unit angular-momentum vector of the orbit plane at ``t_days``."""
    t = np.asarray(t_days, float)
    dt = (t - orbit.epoch) * DAY_S
    rd = 0.0
    if orbit.j2 and not orbit.fixed_ground_track:
        rd = j2_secular_rates(orbit.a, orbit.e, orbit.inc)[0]
    raan = orbit.raan + rd * dt
    si, ci = np.sin(orbit.inc), np.cos(orbit.inc)
    return np.stack([np.sin(raan) * si, -np.cos(raan) * si, np.full_like(raan, ci)], axis=-1)


def site_eci(lat_deg: float, lon_deg: float, t_days) -> np.ndarray:
    """ECI position [m] of a ground site on a spherical Earth."""
    th = gmst_rad(t_days) + lon_deg * DEG
    lat = lat_deg * DEG
    return R_EARTH * np.stack([np.cos(lat) * np.cos(th), np.cos(lat) * np.sin(th),
                               np.sin(lat) * np.ones_like(th)], axis=-1)


def look_angles(r_sat: np.ndarray, r_site: np.ndarray):
    """(elevation [rad], slant range [m]) of satellites as seen from a site (broadcasting)."""
    rho = r_sat - r_site
    rng = np.linalg.norm(rho, axis=-1)
    up = r_site / np.linalg.norm(r_site, axis=-1, keepdims=True)
    sin_el = np.sum(rho * up, axis=-1) / rng
    return np.arcsin(np.clip(sin_el, -1.0, 1.0)), rng


def geo_look_angles(site_lat_deg: float, dlon_deg: float = 0.0):
    """Elevation [rad] and slant range [m] from a site to a GEO satellite ``dlon`` away."""
    lat, dl = site_lat_deg * DEG, dlon_deg * DEG
    cos_psi = np.cos(lat) * np.cos(dl)              # Earth-central angle
    rng = np.sqrt(R_GEO ** 2 + R_EARTH ** 2 - 2.0 * R_GEO * R_EARTH * cos_psi)
    sin_el = (R_GEO * cos_psi - R_EARTH) / rng
    return float(np.arcsin(np.clip(sin_el, -1.0, 1.0))), float(rng)


def slant_range(alt_m, el_rad):
    """Slant range [m] to a satellite at altitude ``alt_m`` seen at elevation ``el_rad``."""
    r = R_EARTH + np.asarray(alt_m)
    s = np.sin(el_rad)
    return np.sqrt(r ** 2 - (R_EARTH * np.cos(el_rad)) ** 2) - R_EARTH * s
