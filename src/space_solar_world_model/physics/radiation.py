"""Trapped-radiation and solar-particle damage to solar arrays (calibrated surrogate).

This is *not* AE9/AP9. It is a compact, physically structured surrogate whose free
constants are fitted to published design-fluence anchors (see ``ANCHORS``), so that the
world model gets the first-order structure right -- benign low LEO, a lethal inner
proton belt at ~1,500-6,000 km, a softer slot, the outer electron belt, moderate GEO,
and solar-particle-only deep space -- while exposing its uncertainty to the causal
model through multipliers (``rad_mult_p/e/sep``). Replace with AE9/AP9 + SPENVIS
outputs for engineering decisions.

Structure
---------
Geomagnetic field: eccentric tilted dipole (axis through the IGRF geomagnetic pole,
centre displaced ~570 km towards the western Pacific -- the origin of the South Atlantic
Anomaly). At each point we compute McIlwain L, the field ratio B/B_eq and the lowest true
altitude reached by the local drift shell, ``h_eff``. Damage-equivalent fluence rate
(1-MeV-electron-equivalent per cm^2 per year, for triple-junction cells behind ~150 um of
coverglass) is

    F = A_p logN(L; L_p, s_p) (B/B_eq)^-k_p S(h_eff)        inner-belt protons
      + A_e logN(L; L_e, s_e) (B/B_eq)^-k_e S(h_eff)        outer-belt electrons
      + A_s sigmoid((L - L_c)/0.4)                           solar particles (cut-off)

averaged over the orbit and over Earth's rotation. Cell power follows the standard
semi-log characteristic  P/P0 = 1 - C log10(1 + Phi/Phi_x)  (fitted to Spectrolab
triple-junction data: P/P0 ~ 0.955 at 1e14 and ~0.87 at 1e15 e/cm^2).
"""
from __future__ import annotations

import numpy as np

from ..constants import DEG, MU_EARTH, R_EARTH
from .orbits import solve_kepler

# --- geomagnetic geometry (IGRF-13/14 era) -----------------------------------------
POLE_LAT_DEG, POLE_LON_DEG = 80.7, -72.7          # geomagnetic north pole
OFFSET_KM, OFFSET_LAT_DEG, OFFSET_LON_DEG = 570.0, 22.0, 141.0


def _unit(lat_deg, lon_deg):
    la, lo = lat_deg * DEG, lon_deg * DEG
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


M_HAT = _unit(POLE_LAT_DEG, POLE_LON_DEG)
D_VEC = OFFSET_KM * 1e3 * _unit(OFFSET_LAT_DEG, OFFSET_LON_DEG)
D_PAR = float(D_VEC @ M_HAT)
D_PERP = float(np.linalg.norm(D_VEC - D_PAR * M_HAT))

# --- fitted surrogate constants (see scripts/space_solar/fit_radiation.py) ---------
RAD_PARAMS = dict(
    A_p=1.681e16, L_p=1.583, s_p=0.178, k_p=1.168,    # inner proton belt
    A_e=1.000e15, L_e=4.515, s_e=0.154, k_e=1.003,    # outer electron belt
    h_c=1637.5, w_h=259.9,                            # atmospheric cut-off [km]
    A_s=3.0e13, L_c=4.0,                              # solar energetic particles
)

# Degradation law constants (triple junction, ~150 um coverglass).
CELL_C = 0.104
CELL_PHI_X = 5.8e13

# Published / derived anchors: (label, alt_km, inc_deg, target fluence per year,
# plausible factor of uncertainty, source). Values marked "estimate" are the authors'
# order-of-magnitude reading of AP8/AE8-era design practice and carry wide bands.
ANCHORS = [
    ("ISS-like LEO", 420.0, 51.6, 7.0e12, 2.5, "Spectrolab LEO column 1e14 per 10-15 yr"),
    ("800 km SSO", 800.0, 98.6, 5.0e13, 3.0, "LEO design range 1e14/10 yr to 5e14/5 yr"),
    ("1,400 km LEO", 1400.0, 52.0, 2.0e14, 4.0, "estimate (inner-belt flank)"),
    ("inner belt core", 3500.0, 0.0, 2.0e16, 5.0, "estimate (cells lose ~25-30%/yr; 1960s data)"),
    ("slot / O3b-like", 8000.0, 0.0, 2.0e15, 5.0, "estimate"),
    ("GPS-like MEO", 20200.0, 55.0, 3.0e14, 3.0, "estimate (outer electron belt)"),
    ("GEO", 35786.0, 0.0, 6.7e13, 1.6, "1e15 e/cm^2 per 15-yr GEO mission (industry design value)"),
]


def magnetic_coordinates(r_ecef: np.ndarray):
    """(L, B/B_eq, h_eff [km]) at Earth-fixed positions [m] in the eccentric dipole."""
    rp = r_ecef - D_VEC
    rr = np.linalg.norm(rp, axis=-1)
    z = rp @ M_HAT
    sin_l = z / rr
    cos2 = np.clip(1.0 - sin_l ** 2, 1e-9, None)
    L = rr / (R_EARTH * cos2)
    b_ratio = np.sqrt(1.0 + 3.0 * sin_l ** 2) / cos2 ** 3
    rho = rr * np.sqrt(cos2)
    # lowest true radius reached around the drift ring (both mirror hemispheres)
    r_min = np.minimum(np.hypot(rho - D_PERP, z + D_PAR), np.hypot(rho - D_PERP, -z + D_PAR))
    h_eff = (r_min - R_EARTH) / 1e3
    return L, b_ratio, h_eff


def _lognormal_shell(L, L0, s):
    return np.exp(-np.log(np.maximum(L, 1e-6) / L0) ** 2 / (2.0 * s ** 2))


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def point_fluence_rate(r_ecef: np.ndarray, params: dict | None = None):
    """Damage-equivalent fluence rate components (p, e, sep) [e/cm^2/yr] at positions."""
    q = RAD_PARAMS if params is None else params
    L, b, h = magnetic_coordinates(r_ecef)
    cut = _sigmoid((h - q["h_c"]) / q["w_h"])
    fp = q["A_p"] * _lognormal_shell(L, q["L_p"], q["s_p"]) * b ** (-q["k_p"]) * cut
    fe = q["A_e"] * _lognormal_shell(L, q["L_e"], q["s_e"]) * b ** (-q["k_e"]) * cut
    fs = q["A_s"] * _sigmoid((L - q["L_c"]) / 0.4)
    return fp, fe, fs


def _orbit_points_ecef(a_m, e, inc_rad, argp_rad=0.0, n_anom: int = 96, n_rot: int = 48):
    """Time-uniform samples of an orbit, spread over Earth's rotation, in Earth-fixed axes."""
    M = np.linspace(0.0, 2 * np.pi, n_anom, endpoint=False)
    if e == 0.0:
        nu, r = M, np.full_like(M, a_m)
    else:
        E = solve_kepler(M, e)
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
        r = a_m * (1 - e * np.cos(E))
    u = argp_rad + nu
    # node at x-axis; rotate about z by phi to sample Earth's rotation underneath
    x0 = r * np.cos(u)
    y0 = r * np.sin(u) * np.cos(inc_rad)
    z0 = r * np.sin(u) * np.sin(inc_rad)
    phi = np.linspace(0.0, 2 * np.pi, n_rot, endpoint=False)[:, None]
    x = x0[None, :] * np.cos(phi) - y0[None, :] * np.sin(phi)
    y = x0[None, :] * np.sin(phi) + y0[None, :] * np.cos(phi)
    z = np.broadcast_to(z0[None, :], x.shape)
    return np.stack([x, y, z], axis=-1)


def orbit_fluence_rate(alt_km: float | None = None, inc_deg: float = 0.0, a_m: float | None = None,
                       e: float = 0.0, argp_deg: float = 0.0, params: dict | None = None,
                       n_anom: int = 96, n_rot: int = 48) -> dict:
    """Orbit-averaged fluence-rate components [1-MeV e-eq / cm^2 / yr]."""
    a = a_m if a_m is not None else R_EARTH + alt_km * 1e3
    pts = _orbit_points_ecef(a, e, inc_deg * DEG, argp_deg * DEG, n_anom, n_rot)
    fp, fe, fs = point_fluence_rate(pts, params)
    out = dict(p=float(fp.mean()), e=float(fe.mean()), sep=float(fs.mean()))
    out["total"] = out["p"] + out["e"] + out["sep"]
    return out


def deep_space_fluence_rate(params: dict | None = None) -> dict:
    """Outside the magnetosphere (Sun-Earth L1/L2, lunar vicinity): solar particles only."""
    q = RAD_PARAMS if params is None else params
    return dict(p=0.0, e=0.0, sep=float(q["A_s"]), total=float(q["A_s"]))


def remaining_power_factor(fluence, C=CELL_C, phi_x=CELL_PHI_X):
    """Maximum-power remaining factor P/P0 after a 1-MeV-equivalent fluence [e/cm^2]."""
    return np.clip(1.0 - C * np.log10(1.0 + np.asarray(fluence) / phi_x), 0.05, 1.0)


def spiral_transfer_fluence(r0_m: float, r1_m: float, inc0_deg: float, inc1_deg: float,
                            accel_ms2: float, dv_total_ms: float | None = None, n: int = 120,
                            params: dict | None = None) -> dict:
    """Fluence components accumulated on a constant-acceleration low-thrust spiral.

    Time spent near radius r is |dv/dr| / a with v = sqrt(mu/r). If the total (Edelbaum)
    delta-v exceeds the tangential speed change, the surplus is the plane change, which
    optimal steering performs where orbital speed is lowest; we distribute it in
    proportion to 1/v. Inclination follows the cumulative plane-change budget.
    """
    if r1_m <= r0_m:
        return dict(p=0.0, e=0.0, sep=0.0, total=0.0, days=0.0)
    r = np.geomspace(r0_m, r1_m, n + 1)
    v = np.sqrt(MU_EARTH / r)
    dv_tan = -np.diff(v)                                   # tangential speed change
    rm = np.sqrt(r[1:] * r[:-1])
    vm = np.sqrt(MU_EARTH / rm)
    extra = 0.0 if dv_total_ms is None else max(dv_total_ms - dv_tan.sum(), 0.0)
    w = 1.0 / vm
    dv_pc = extra * w / w.sum()                            # plane-change share per segment
    dv = dv_tan + dv_pc
    if extra > 0:
        frac = (np.cumsum(dv_pc) - 0.5 * dv_pc) / extra
    else:
        frac = (np.cumsum(dv) - 0.5 * dv) / dv.sum()
    inc = inc0_deg + (inc1_deg - inc0_deg) * frac
    dt_yr = dv / accel_ms2 / (365.25 * 86400.0)
    comp = dict(p=0.0, e=0.0, sep=0.0)
    for k in range(n):
        f = orbit_fluence_rate(a_m=rm[k], inc_deg=float(inc[k]), params=params, n_anom=48, n_rot=24)
        for c in comp:
            comp[c] += f[c] * dt_yr[k]
    comp["total"] = comp["p"] + comp["e"] + comp["sep"]
    comp["days"] = float(dt_yr.sum() * 365.25)
    return comp


def anchor_table(params: dict | None = None):
    """Model vs anchor fluence for every calibration point (for tests and docs)."""
    rows = []
    for label, alt, inc, target, band, src in ANCHORS:
        f = orbit_fluence_rate(alt, inc, params=params)["total"]
        rows.append(dict(label=label, alt_km=alt, inc_deg=inc, target=target, model=f,
                         ratio=f / target, band=band, source=src))
    return rows
