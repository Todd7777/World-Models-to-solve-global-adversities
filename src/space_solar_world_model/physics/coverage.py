"""Link coverage: when a receiving site can be served, from what range and elevation.

Two system types:

* **dedicated** -- a fixed set of satellites serves one site (GEO above the site's
  longitude; a Tundra or Molniya trio dwelling over it; Sun-Earth L1). Coverage is
  simulated directly over a full year.
* **shared** -- a Walker-type LEO/MEO constellation serves a global network of sites.
  Because Walker planes are uniformly spaced in right ascension, the probability ``q``
  that one satellite is simultaneously above a site's elevation mask *and* sunlit
  depends only on the site latitude, its local solar time and the solar declination.
  We tabulate q on that grid by integrating over a dense lattice of virtual satellites
  (deterministic, no Monte-Carlo noise) and interpolate it onto the year. With T
  satellites the site is served with probability 1 - exp(-T q) (Poisson; validated
  against an explicit Walker simulation in the tests).

Key physical effect captured: a LEO satellite above a site at local night is usually
inside Earth's shadow, so LEO delivery is concentrated in daytime and twilight --
precisely when ground solar already produces.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import AU, DAY_S, DEG, MU_EARTH, R_EARTH, SE_L1_DISTANCE
from ..sites import Site
from .orbits import (Orbit, geo_look_angles, j2_secular_rates, look_angles, propagate,
                     site_eci, sso_inclination)
from .shadow import illumination_fraction
from .sun import EPOCH_2030, gmst_rad, sun_ecliptic, sun_position_eci, sun_ra_dec, sun_elevation_at_site

K_SAMPLES = 256


@dataclass
class SiteLink:
    """Deliverability time series for one site plus the (range, elevation) distribution."""

    site: Site
    q: np.ndarray            # deliverable weight per time sample (per satellite if shared)
    month: np.ndarray        # month index 0..11 of each sample
    hour: np.ndarray         # local apparent solar hour 0..23 of each sample
    D: np.ndarray            # slant-range samples [m] while deliverable
    el: np.ndarray           # elevation samples [rad] while deliverable
    w: np.ndarray            # sample weights (sum to 1)

    def availability(self, n_sats: float = 1.0, shared: bool = False) -> float:
        if shared:
            return float(np.mean(1.0 - np.exp(-n_sats * self.q)))
        return float(np.mean(self.q))

    def hourly_profile(self, n_sats: float = 1.0, shared: bool = False) -> np.ndarray:
        """[12, 24] mean deliverable fraction by month and local solar hour."""
        a = 1.0 - np.exp(-n_sats * self.q) if shared else self.q
        out = np.zeros((12, 24))
        cnt = np.zeros((12, 24))
        np.add.at(out, (self.month, self.hour), a)
        np.add.at(cnt, (self.month, self.hour), 1.0)
        return out / np.maximum(cnt, 1.0)


@dataclass
class LinkStats:
    placement: str
    mode: str                     # "dedicated" or "shared"
    n_sats: int                   # satellites per dedicated system (ignored if shared)
    el_min_deg: float
    sites: list = field(default_factory=list)

    def site(self, name: str) -> SiteLink:
        for s in self.sites:
            if s.site.name == name:
                return s
        raise KeyError(name)


# --------------------------------------------------------------------------- helpers
def _year_times(step_min: float, epoch: float = EPOCH_2030) -> np.ndarray:
    return epoch + np.arange(0.0, 365.0 * 1440.0, step_min) / 1440.0


def _time_indices(t_days: np.ndarray, lon_deg: float):
    ra, _ = sun_ra_dec(t_days)
    ha = np.mod(gmst_rad(t_days) + lon_deg * DEG - ra, 2 * np.pi)       # solar hour angle
    hour = np.floor(np.mod(ha / (2 * np.pi) * 24.0 + 12.0, 24.0)).astype(np.int8)
    doy = np.floor(t_days - t_days[0]).astype(int)
    month = np.minimum((doy / 30.4375).astype(int), 11).astype(np.int8)
    return month, hour


def _subsample(D, el, wt, k: int = K_SAMPLES, seed: int = 0):
    D, el, wt = np.ravel(D), np.ravel(el), np.ravel(wt)
    m = wt > 0
    if not m.any():
        return np.array([1.0]), np.array([np.pi / 2]), np.array([1.0])
    D, el, wt = D[m], el[m], wt[m]
    if D.size <= k:
        return D, el, wt / wt.sum()
    rng = np.random.default_rng(seed)
    idx = rng.choice(D.size, size=k, replace=True, p=wt / wt.sum())
    return D[idx], el[idx], np.full(k, 1.0 / k)


# --------------------------------------------------------------------- dedicated
def geo_link(site: Site, el_min_deg: float = 10.0, step_min: float = 2.0,
             sat_lon_deg: float | None = None, epoch: float = EPOCH_2030) -> SiteLink:
    lon = site.lon if sat_lon_deg is None else sat_lon_deg
    el, D = geo_look_angles(site.lat, site.lon - lon)
    t = _year_times(step_min, epoch)
    nu = illumination_fraction(propagate(Orbit.geostationary(lon, epoch), t), sun_position_eci(t))
    _, _, _, r_au = sun_ecliptic(t)
    q = nu / r_au ** 2 * (el >= el_min_deg * DEG)
    month, hour = _time_indices(t, site.lon)
    return SiteLink(site, q.astype(np.float32), month, hour, np.array([D]), np.array([el]),
                    np.array([1.0]))


def heo_link(kind: str, site: Site, n_sats: int = 3, el_min_deg: float = 20.0,
             step_min: float = 4.0, epoch: float = EPOCH_2030) -> SiteLink:
    """Tundra/Molniya constellation with apogee over the site's longitude; best satellite wins."""
    t = _year_times(step_min, epoch)
    r_site = site_eci(site.lat, site.lon, t)
    r_sun = sun_position_eci(t)
    _, _, _, r_au = sun_ecliptic(t)
    best = np.full(t.shape, -np.inf)
    D_b = np.zeros_like(t)
    nu_b = np.zeros_like(t)
    for j in range(n_sats):
        r = propagate(Orbit.critical_heo(kind, site.lon, j, n_sats, epoch), t)
        el, D = look_angles(r, r_site)
        nu = illumination_fraction(r, r_sun)
        ok = (el >= el_min_deg * DEG) & (nu > 0)
        score = np.where(ok, el, -np.inf)
        better = score > best
        best = np.where(better, score, best)
        D_b = np.where(better, D, D_b)
        nu_b = np.where(better, nu, nu_b)
    q = np.where(np.isfinite(best), nu_b / r_au ** 2, 0.0)
    el_b = np.where(np.isfinite(best), best, 0.0)
    month, hour = _time_indices(t, site.lon)
    Ds, els, ws = _subsample(D_b, el_b, q)
    return SiteLink(site, q.astype(np.float32), month, hour, Ds, els, ws)


def l1_link(site: Site, el_min_deg: float = 20.0, step_min: float = 10.0,
            epoch: float = EPOCH_2030) -> SiteLink:
    """Sun-Earth L1 sits on the Sun line, so a site sees it only in daytime."""
    t = _year_times(step_min, epoch)
    el = sun_elevation_at_site(site.lat, site.lon, t)
    _, _, _, r_au = sun_ecliptic(t)
    q = (el >= el_min_deg * DEG) / (r_au * (1.0 - SE_L1_DISTANCE / AU)) ** 2
    month, hour = _time_indices(t, site.lon)
    Ds, els, ws = _subsample(np.full_like(t, SE_L1_DISTANCE), el, q)
    return SiteLink(site, q.astype(np.float32), month, hour, Ds, els, ws)


# ------------------------------------------------------------------------ shared
def _virtual_constellation(a_m: float, inc: float, raans: np.ndarray, n_u: int):
    u = np.linspace(0.0, 2 * np.pi, n_u, endpoint=False)
    O, U = np.meshgrid(raans, u, indexing="ij")
    O, U = O.ravel(), U.ravel()
    ci, si = np.cos(inc), np.sin(inc)
    x = a_m * (np.cos(O) * np.cos(U) - np.sin(O) * np.sin(U) * ci)
    y = a_m * (np.sin(O) * np.cos(U) + np.cos(O) * np.sin(U) * ci)
    z = a_m * (np.sin(U) * si)
    return np.stack([x, y, z], axis=-1)            # [N, 3]


def shared_q_table(alt_km: float, lat_deg: float, inc_deg: float | None = None,
                   ltan_hours: float | None = None, el_min_deg: float = 20.0,
                   n_ha: int = 96, n_dec: int = 13, n_raan: int = 72, n_u: int = 360):
    """Per-satellite deliverable probability q[hour angle, declination] for one latitude.

    The Sun is fixed at right ascension 0 in this frame. Returns (ha grid, dec grid, q,
    sample arrays D, el, weight).
    """
    a = R_EARTH + alt_km * 1e3
    if ltan_hours is not None:
        inc = sso_inclination(a)
        raans = np.array([(ltan_hours - 12.0) * 15.0 * DEG])
        n_u = max(n_u, 720)
    else:
        inc = inc_deg * DEG
        raans = np.linspace(0.0, 2 * np.pi, n_raan, endpoint=False)
    sats = _virtual_constellation(a, inc, raans, n_u)                     # [N,3]
    ha = np.linspace(0.0, 2 * np.pi, n_ha, endpoint=False)
    dec = np.linspace(-23.44, 23.44, n_dec) * DEG
    lat = lat_deg * DEG
    q = np.zeros((n_ha, n_dec))
    Ds, els, ws = [], [], []
    for j, d in enumerate(dec):
        r_sun = AU * np.array([np.cos(d), 0.0, np.sin(d)])
        nu = illumination_fraction(sats, r_sun[None, :])                   # [N]
        lit = nu > 0
        for i, h in enumerate(ha):
            g = R_EARTH * np.array([np.cos(lat) * np.cos(h), np.cos(lat) * np.sin(h), np.sin(lat)])
            rho = sats[lit] - g
            rng = np.linalg.norm(rho, axis=-1)
            sin_el = rho @ (g / R_EARTH) / rng
            vis = sin_el >= np.sin(el_min_deg * DEG)
            wv = nu[lit][vis]
            q[i, j] = wv.sum() / sats.shape[0]
            if wv.size:
                Ds.append(rng[vis])
                els.append(np.arcsin(np.clip(sin_el[vis], -1, 1)))
                ws.append(wv)
    if Ds:
        D_all, el_all, w_all = np.concatenate(Ds), np.concatenate(els), np.concatenate(ws)
    else:
        D_all = el_all = w_all = np.zeros(0)
    return ha, dec, q, D_all, el_all, w_all


def shared_constellation_link(alt_km: float, sites: list[Site], inc_deg: float | None = None,
                              ltan_hours: float | None = None, el_min_deg: float = 20.0,
                              step_min: float = 15.0, placement: str = "",
                              epoch: float = EPOCH_2030, **grid) -> LinkStats:
    t = _year_times(step_min, epoch)
    ra, dec_t = sun_ra_dec(t)
    _, _, _, r_au = sun_ecliptic(t)
    out = []
    cache = {}
    for s in sites:
        key = round(s.lat, 3)
        if key not in cache:
            cache[key] = shared_q_table(alt_km, s.lat, inc_deg, ltan_hours, el_min_deg, **grid)
        ha_g, dec_g, qt, D, el, w = cache[key]
        ha_t = np.mod(gmst_rad(t) + s.lon * DEG - ra, 2 * np.pi)
        # bilinear interpolation on the (periodic hour angle, declination) grid
        fi = ha_t / (2 * np.pi) * len(ha_g)
        i0 = np.floor(fi).astype(int) % len(ha_g)
        i1 = (i0 + 1) % len(ha_g)
        ti = fi - np.floor(fi)
        fj = np.clip((dec_t - dec_g[0]) / (dec_g[1] - dec_g[0]), 0, len(dec_g) - 1 - 1e-9)
        j0 = np.floor(fj).astype(int)
        tj = fj - j0
        q = ((1 - ti) * (1 - tj) * qt[i0, j0] + ti * (1 - tj) * qt[i1, j0]
             + (1 - ti) * tj * qt[i0, j0 + 1] + ti * tj * qt[i1, j0 + 1]) / r_au ** 2
        month, hour = _time_indices(t, s.lon)
        Ds, els, ws = _subsample(D, el, w, seed=int(abs(s.lat) * 100))
        out.append(SiteLink(s, q.astype(np.float32), month, hour, Ds, els, ws))
    return LinkStats(placement, "shared", 0, el_min_deg, out)


def simulate_walker_availability(alt_km: float, inc_deg: float, n_planes: int, per_plane: int,
                                 site: Site, el_min_deg: float = 20.0, days: int = 6,
                                 step_s: float = 60.0, phasing: int = 1,
                                 epoch: float = EPOCH_2030) -> float:
    """Brute-force fraction of time an explicit Walker constellation serves ``site``.

    Used to validate the Poisson / symmetric-grid shortcut.
    """
    a = R_EARTH + alt_km * 1e3
    inc = inc_deg * DEG
    T = n_planes * per_plane
    raan = np.repeat(np.arange(n_planes) * 2 * np.pi / n_planes, per_plane)
    u0 = (np.tile(np.arange(per_plane), n_planes) * 2 * np.pi / per_plane
          + np.repeat(np.arange(n_planes), per_plane) * 2 * np.pi * phasing / T)
    rd, wd, md = j2_secular_rates(a, 0.0, inc)
    n = np.sqrt(MU_EARTH / a ** 3) + md + wd
    served = []
    for d in np.linspace(0, 365, days, endpoint=False):
        t = epoch + d + np.arange(0, DAY_S, step_s) / DAY_S
        dt = (t - epoch)[:, None] * DAY_S
        O = raan[None, :] + rd * dt
        U = u0[None, :] + n * dt
        ci, si = np.cos(inc), np.sin(inc)
        r = a * np.stack([np.cos(O) * np.cos(U) - np.sin(O) * np.sin(U) * ci,
                          np.sin(O) * np.cos(U) + np.cos(O) * np.sin(U) * ci,
                          np.sin(U) * si * np.ones_like(O)], axis=-1)
        nu = illumination_fraction(r, sun_position_eci(t)[:, None, :])
        el, _ = look_angles(r, site_eci(site.lat, site.lon, t)[:, None, :])
        ok = (el >= el_min_deg * DEG) & (nu > 0.5)
        served.append(ok.any(axis=1).mean())
    return float(np.mean(served))
