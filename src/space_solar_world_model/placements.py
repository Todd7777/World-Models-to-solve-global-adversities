"""Candidate places to put solar arrays, with their precomputed physics.

A ``Placement`` bundles every deterministic, placement-specific quantity the structural
model needs (illumination, radiation, drag, debris, station-keeping, transport), so that
uncertainty, design and economics can be layered on top cheaply and vectorised.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import C_LIGHT, MOON_DISTANCE, R_EARTH, SE_L1_DISTANCE
from .economics import transport as TR
from .physics import environment as ENV
from .physics import radiation as RAD
from .physics.illumination import (LUNAR_SITES, IlluminationStats, circular_illumination,
                                   lagrange_illumination, numeric_illumination)
from .physics.orbits import Orbit, sso_inclination

REF_ACCEL = 1.0e-3          # m/s^2, reference SEP acceleration for stored transfer doses
STOW_SHIELD = 0.35          # dose fraction reaching stowed (rolled/stacked) arrays in transit


@dataclass
class Placement:
    key: str
    label: str
    family: str                      # leo | sso | meo | geo | heo | lagrange | lunar
    alt_km: float | None
    inc_deg: float | None
    illum: IlluminationStats
    fluence: dict                    # p, e, sep [1-MeV e-eq cm^-2 yr^-1]
    drag_coeff: float                # m/s/yr per (m^2/kg), Sun-tracking attitude included
    srp_coeff: float                 # m/s/yr per (m^2/kg)
    sk_dv: float                     # other station-keeping [m/s/yr]
    disposal_dv: float               # [m/s]
    debris_10cm: float               # [m^-2 yr^-1]
    debris_1cm: float
    transport: dict                  # option -> {mass_mult_chem | dv, ep, incl_mult, days_ref, phi_ref}
    latency_ms: float
    ltan_hours: float | None = None
    e: float = 0.0
    irr_min: float = 1.0 / 1.0167 ** 2   # worst-month irradiance factor (aphelion)
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        d = dict(key=self.key, label=self.label, family=self.family, alt_km=self.alt_km,
                 inc_deg=self.inc_deg, latency_ms=self.latency_ms,
                 sunlit_fraction=self.illum.sunlit_fraction,
                 irradiance_factor=self.illum.irradiance_factor,
                 max_eclipse_min=self.illum.max_eclipse_min,
                 eclipses_per_year=self.illum.eclipses_per_year,
                 fluence_total=self.fluence["total"], drag_coeff=self.drag_coeff,
                 debris_10cm=self.debris_10cm)
        return d


def _transport(kind, alt_km=None, inc_deg=None, e=0.0, a_m=None, isp_chem=380.0):
    opts = TR.transfer_options(kind, alt_km, inc_deg, e, a_m)
    out = {}
    for name, o in opts.items():
        if not o["ep"]:
            stages = o.get("stages", [(o["dv"], o["eps"])])
            out[name] = dict(ep=False, dv=o["dv"], incl_mult=o["incl_mult"],
                             mass_mult_chem=TR.staged_chemical_multiplier(stages, isp_chem),
                             days_ref=3.0, phi_ref=dict(p=0.0, e=0.0, sep=0.0))
        else:
            f = RAD.spiral_transfer_fluence(o["r0"], min(o["r1"], 5.0e7), o["inc0"], o["inc1"],
                                            REF_ACCEL, dv_total_ms=o["dv"])
            days = o["dv"] / REF_ACCEL / 86400.0
            out[name] = dict(ep=True, dv=o["dv"], incl_mult=o["incl_mult"], mass_mult_chem=1.0,
                             days_ref=days,
                             phi_ref={k: STOW_SHIELD * f[k] for k in ("p", "e", "sep")})
    return out


def _sk_dv(family: str) -> float:
    return {"leo": 10.0, "sso": 10.0, "meo": 5.0, "geo": 52.0, "heo": 20.0,
            "lagrange": 5.0, "lunar": 5.0}[family]


def circular(alt_km: float, inc_deg: float | None = None, ltan_hours: float | None = None,
             key: str | None = None, label: str | None = None) -> Placement:
    if ltan_hours is not None:
        inc_deg = float(np.degrees(sso_inclination(R_EARTH + alt_km * 1e3)))
        family = "sso"
    else:
        family = "geo" if abs(alt_km - 35786.0) < 1.0 and inc_deg == 0.0 else (
            "leo" if alt_km <= 2000.0 else "meo")
    ill = circular_illumination(alt_km, inc_deg if ltan_hours is None else None, ltan_hours)
    flu = RAD.orbit_fluence_rate(alt_km, inc_deg)
    att = ENV.sun_tracking_attitude_factor(ill.mean_abs_cos_beta, ill.sunlit_fraction)
    drag = ENV.drag_dv_coefficient(alt_km, attitude_factor=att) if alt_km < 3000 else 0.0
    srp = ENV.srp_dv_coefficient() if alt_km > 2000 else 0.0
    if key is None:
        key = (f"sso{int(ltan_hours):02d}_{int(alt_km)}" if ltan_hours is not None
               else f"{family}_{int(alt_km)}_{int(round(inc_deg))}")
    if label is None:
        label = (f"{'Dawn-dusk' if ltan_hours in (6.0, 18.0) else 'SSO'} SSO {alt_km:,.0f} km"
                 if ltan_hours is not None else
                 ("GEO" if family == "geo" else f"{family.upper()} {alt_km:,.0f} km, {inc_deg:.0f} deg"))
    return Placement(
        key=key, label=label, family=family, alt_km=alt_km, inc_deg=inc_deg, illum=ill,
        fluence=flu, drag_coeff=drag, srp_coeff=srp, sk_dv=_sk_dv(family),
        disposal_dv=ENV.disposal_dv(alt_km), debris_10cm=float(ENV.debris_flux(alt_km, "10cm")),
        debris_1cm=float(ENV.debris_flux(alt_km, "1cm")),
        transport=_transport("circular", alt_km, inc_deg), latency_ms=alt_km / C_LIGHT * 1e6,
        ltan_hours=ltan_hours)


def geo() -> Placement:
    return circular(35786.0, 0.0, key="geo", label="GEO (geostationary)")


def heo(kind: str) -> Placement:
    orb = Orbit.critical_heo(kind, 0.0)
    ill = numeric_illumination(orb, day_step=2, samples_per_orbit=720)
    flu = RAD.orbit_fluence_rate(a_m=orb.a, e=orb.e, inc_deg=63.4349, argp_deg=270.0,
                                 n_anom=180, n_rot=48)
    drag = ENV.drag_dv_coefficient(a_m=orb.a, e=orb.e, attitude_factor=2 / np.pi)
    return Placement(
        key=kind, label=f"{kind.capitalize()} (HEO, 63.4 deg)", family="heo",
        alt_km=orb.apogee_alt_km, inc_deg=63.4349, illum=ill, fluence=flu, drag_coeff=drag,
        srp_coeff=ENV.srp_dv_coefficient(), sk_dv=_sk_dv("heo"), disposal_dv=30.0,
        debris_10cm=float(ENV.debris_flux(orb.perigee_alt_km, "10cm")) * 0.02,
        debris_1cm=float(ENV.debris_flux(orb.perigee_alt_km, "1cm")) * 0.02,
        transport=_transport("heo", e=orb.e, a_m=orb.a), latency_ms=orb.apogee_alt_km / C_LIGHT * 1e6,
        e=orb.e, meta=dict(perigee_km=orb.perigee_alt_km, apogee_km=orb.apogee_alt_km))


def lagrange(point: str) -> Placement:
    ill = lagrange_illumination(point)
    return Placement(
        key=point.lower(), label=f"Sun-Earth {point} (halo)", family="lagrange", alt_km=None,
        inc_deg=None, illum=ill, fluence=RAD.deep_space_fluence_rate(), drag_coeff=0.0,
        srp_coeff=0.0, sk_dv=_sk_dv("lagrange"), disposal_dv=20.0, debris_10cm=0.0, debris_1cm=0.0,
        transport=_transport(point), latency_ms=SE_L1_DISTANCE / C_LIGHT * 1e3,
        irr_min=ill.irradiance_factor / 1.0167 ** 2)


def lunar(site: str) -> Placement:
    ill = LUNAR_SITES[site]
    surface = site.startswith("moon_")
    return Placement(
        key=site, label={"moon_south_pole_pel": "Lunar south pole (peak of light)",
                         "moon_equator": "Lunar equator", "lunar_nrho": "Lunar NRHO"}[site],
        family="lunar", alt_km=None, inc_deg=None, illum=ill, fluence=RAD.deep_space_fluence_rate(),
        drag_coeff=0.0, srp_coeff=0.0 if surface else ENV.srp_dv_coefficient() * 0.25,
        sk_dv=0.0 if surface else _sk_dv("lunar"), disposal_dv=0.0 if surface else 20.0,
        debris_10cm=0.0, debris_1cm=0.0, transport=_transport(site),
        latency_ms=MOON_DISTANCE / C_LIGHT * 1e3)


def standard_catalog() -> dict[str, Placement]:
    """Representative placements spanning every orbital regime."""
    P = {}
    for alt in (500, 650, 800, 1000, 1200, 1500):
        for inc in (0.0, 53.0, 90.0):
            p = circular(float(alt), inc)
            P[p.key] = p
    for alt in (500, 650, 800, 1000, 1200, 1400, 1600, 2000):
        p = circular(float(alt), ltan_hours=18.0)
        P[p.key] = p
    p = circular(800.0, ltan_hours=12.0)
    P[p.key] = p
    for alt in (3000, 5000, 8000, 12000, 20200, 25000):
        for inc in (0.0, 55.0):
            p = circular(float(alt), inc)
            P[p.key] = p
    g = geo()
    P[g.key] = g
    for k in ("tundra", "molniya"):
        P[k] = heo(k)
    for pt in ("L1", "L2"):
        p = lagrange(pt)
        P[p.key] = p
    for s in ("lunar_nrho", "moon_south_pole_pel"):
        P[s] = lunar(s)
    return P
