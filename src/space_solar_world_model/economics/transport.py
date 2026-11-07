"""Getting mass to its final orbit: launch, chemical stages and solar-electric tugs.

Delta-v models
--------------
* impulsive: Hohmann transfer with the plane change combined into the apoapsis burn;
* low thrust: Edelbaum, dv = sqrt(v0^2 + v1^2 - 2 v0 v1 cos(pi/2 * di)).

Launch: price is quoted to a reference 400 km LEO from a 28.5 deg site; reaching higher
inclinations forfeits Earth-rotation assist v_eq (cos(lat) - cos(i)), which reduces
reusable-launcher payload roughly as exp(-dv / 3 km/s).

The structural coefficient eps of in-space chemical stages is 0.12 (tanks, engines).
"""
from __future__ import annotations

import numpy as np

from ..constants import DEG, G0, MU_EARTH, OMEGA_EARTH, R_EARTH

REF_ALT_KM = 400.0
REF_INC_DEG = 28.5
V_EQ = OMEGA_EARTH * R_EARTH            # 465 m/s
LV_DV_SENSITIVITY = 3000.0               # m/s, payload e-folding for the launcher
STAGE_EPS = 0.12


def circ_v(r):
    return np.sqrt(MU_EARTH / r)


def hohmann_dv(r1: float, r2: float, di_rad: float = 0.0) -> float:
    if r2 < r1:
        r1, r2 = r2, r1
    at = 0.5 * (r1 + r2)
    v1, v2 = circ_v(r1), circ_v(r2)
    vp = np.sqrt(MU_EARTH * (2 / r1 - 1 / at))
    va = np.sqrt(MU_EARTH * (2 / r2 - 1 / at))
    dv2 = np.sqrt(va ** 2 + v2 ** 2 - 2 * va * v2 * np.cos(di_rad))
    return float((vp - v1) + dv2)


def edelbaum_dv(r1: float, r2: float, di_rad: float = 0.0) -> float:
    v1, v2 = circ_v(r1), circ_v(r2)
    return float(np.sqrt(v1 ** 2 + v2 ** 2 - 2 * v1 * v2 * np.cos(0.5 * np.pi * di_rad)))


def launch_inclination_multiplier(inc_deg: float, site_lat_deg: float = REF_INC_DEG) -> float:
    """Launch cost multiplier for inserting into ``inc_deg`` instead of due east."""
    i = max(inc_deg, site_lat_deg)
    dv = V_EQ * (np.cos(site_lat_deg * DEG) - np.cos(i * DEG))
    return float(np.exp(max(dv, 0.0) / LV_DV_SENSITIVITY))


def chemical_mass_multiplier(dv: float, isp: float, eps: float = STAGE_EPS) -> float:
    """Mass that must be launched to LEO per kg delivered after a chemical burn of ``dv``."""
    if dv <= 0:
        return 1.0
    R = np.exp(dv / (isp * G0))
    lam = 1.0 / R - eps / (1.0 - eps) * (1.0 - 1.0 / R)
    return float(1.0 / lam) if lam > 0.02 else float("inf")


def staged_chemical_multiplier(stages, isp: float) -> float:
    m = 1.0
    for dv, eps in stages:
        m *= chemical_mass_multiplier(dv, isp, eps)
    return m


def ep_mass_multiplier(dv: float, isp: float) -> float:
    return float(np.exp(dv / (isp * G0)))


# Mission delta-v from the reference LEO for non-circular-Earth-orbit destinations
# (chemical, low-thrust) [m/s]. Sources: standard mission-design values
# (TLI ~3.1-3.2 km/s; Sun-Earth L1/L2 insertion ~0.2-0.4 km/s; NRHO ~0.45 km/s;
# lunar landing ~2.0 km/s incl. margins; low-thrust escape ~ v_LEO).
SPECIAL_DV = {
    "L1": (3400.0, 7300.0),
    "L2": (3400.0, 7300.0),
    "lunar_nrho": (3650.0, 7900.0),
    "moon_south_pole_pel": (5950.0, None),       # landing needs a chemical lander
    "moon_equator": (5950.0, None),
}
LANDER_EPS = 0.25                                 # landers carry legs, engines, avionics
LUNAR_STAGES = [(4000.0, STAGE_EPS), (1950.0, LANDER_EPS)]   # LEO->LLO, LLO->surface


def transfer_options(kind: str, alt_km: float | None = None, inc_deg: float | None = None,
                     e: float = 0.0, a_m: float | None = None) -> dict:
    """Delta-v and launch-inclination factor for each way of reaching a destination.

    Returns {option: dict(dv, eps, ep (bool), incl_mult, r0, r1, inc0, inc1)}.
    """
    r_ref = R_EARTH + REF_ALT_KM * 1e3
    out = {}
    if kind in SPECIAL_DV:
        dv_c, dv_l = SPECIAL_DV[kind]
        stages = LUNAR_STAGES if kind.startswith("moon_") else [(dv_c, STAGE_EPS)]
        out["chemical"] = dict(dv=dv_c, eps=STAGE_EPS, ep=False, incl_mult=1.0, stages=stages)
        if dv_l is not None:
            out["sep_tug"] = dict(dv=dv_l, eps=0.0, ep=True, incl_mult=1.0, r0=r_ref,
                                  r1=3.0e8, inc0=REF_INC_DEG, inc1=REF_INC_DEG)
        return out
    if kind == "heo":
        # launch into 63.4 deg LEO, raise apogee at perigee, then raise perigee at apogee
        rp, ra = a_m * (1 - e), a_m * (1 + e)
        incl = launch_inclination_multiplier(63.4)
        at = 0.5 * (r_ref + ra)
        dv1 = np.sqrt(MU_EARTH * (2 / r_ref - 1 / at)) - circ_v(r_ref)
        va1 = np.sqrt(MU_EARTH * (2 / ra - 1 / at))
        va2 = np.sqrt(MU_EARTH * (2 / ra - 1 / a_m))
        dv = float(dv1 + abs(va2 - va1))
        out["chemical"] = dict(dv=dv, eps=STAGE_EPS, ep=False, incl_mult=incl)
        out["sep_tug"] = dict(dv=1.25 * dv, eps=0.0, ep=True, incl_mult=incl, r0=r_ref,
                              r1=float(np.sqrt(rp * ra)), inc0=63.4, inc1=63.4)
        return out
    # circular Earth orbit. Launch from a 28.5 deg site: inclinations at or above it are
    # reached by direct insertion (paying the lost rotation assist); lower ones need a
    # plane change, done at the highest point of an impulsive transfer or spread along a
    # low-thrust spiral (Edelbaum).
    r1 = R_EARTH + alt_km * 1e3
    if inc_deg >= REF_INC_DEG:
        inc0, di = inc_deg, 0.0
        incl = launch_inclination_multiplier(inc_deg)
    else:
        inc0, di = REF_INC_DEG, (REF_INC_DEG - inc_deg) * DEG
        incl = 1.0
    if r1 >= r_ref:
        dv_c = hohmann_dv(r_ref, r1, di)
    else:
        dv_c = hohmann_dv(r1, r_ref, di)
    out["chemical"] = dict(dv=dv_c, eps=STAGE_EPS, ep=False, incl_mult=incl)
    if r1 > r_ref or di > 0:
        out["sep_tug"] = dict(dv=edelbaum_dv(r_ref, r1, di), eps=0.0, ep=True, incl_mult=incl,
                              r0=min(r_ref, r1), r1=max(r_ref, r1), inc0=inc0, inc1=inc_deg)
    return out
