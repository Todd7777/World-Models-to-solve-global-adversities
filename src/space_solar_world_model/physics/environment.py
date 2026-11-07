"""Space environment besides radiation: atmospheric drag, station-keeping and debris.

* Atmosphere: piecewise-exponential density (Vallado, *Fundamentals of Astrodynamics*,
  Table 8-4; CIRA-72, moderate solar activity). Solar-cycle variability (x~0.3-5 at
  400-800 km) enters the causal model as the exogenous multiplier ``atm_mult``.
* Drag on a Sun-tracking array: the projected area against the flow is
  A |cos(angle between array normal and velocity)|; averaged over an orbit this is
  (2/pi) A |cos beta|. Dawn-dusk orbits (beta near 90 deg) are therefore edge-on to the
  flow most of the time -- one reason they are favoured for large arrays.
* Debris: a smooth parametric fit to the shape of ORDEM 3.x / MASTER-8 flux-vs-altitude
  curves (main peak ~800-1,000 km, secondary ~1,400-1,500 km, sparse MEO, low-velocity
  GEO ring). Absolute levels are uncertain by ~x3 and enter the causal model through the
  exogenous multiplier ``debris_mult``. A catastrophic (>10 cm) impact is assumed to
  destroy one structural module.
"""
from __future__ import annotations

import numpy as np

from ..constants import C_LIGHT, JULIAN_YEAR_S, MU_EARTH, R_EARTH, SUN_SYNC_RATE, TSI_1AU

# (base altitude km, density kg/m^3, scale height km) -- Vallado Table 8-4.
_ATM = np.array([
    [0, 1.225, 7.249], [25, 3.899e-2, 6.349], [30, 1.774e-2, 6.682], [40, 3.972e-3, 7.554],
    [50, 1.057e-3, 8.382], [60, 3.206e-4, 7.714], [70, 8.770e-5, 6.549], [80, 1.905e-5, 5.799],
    [90, 3.396e-6, 5.382], [100, 5.297e-7, 5.877], [110, 9.661e-8, 7.263],
    [120, 2.438e-8, 9.473], [130, 8.484e-9, 12.636], [140, 3.845e-9, 16.149],
    [150, 2.070e-9, 22.523], [180, 5.464e-10, 29.740], [200, 2.789e-10, 37.105],
    [250, 7.248e-11, 45.546], [300, 2.418e-11, 53.628], [350, 9.518e-12, 53.298],
    [400, 3.725e-12, 58.515], [450, 1.585e-12, 60.828], [500, 6.967e-13, 63.822],
    [600, 1.454e-13, 71.835], [700, 3.614e-14, 88.667], [800, 1.170e-14, 124.64],
    [900, 5.245e-15, 181.05], [1000, 3.019e-15, 268.00],
])


def atmospheric_density(alt_km):
    """Neutral density [kg/m^3] (moderate solar activity); exponential above 1,000 km."""
    h = np.asarray(alt_km, float)
    idx = np.clip(np.searchsorted(_ATM[:, 0], h, side="right") - 1, 0, len(_ATM) - 1)
    h0, rho0, H = _ATM[idx, 0], _ATM[idx, 1], _ATM[idx, 2]
    return rho0 * np.exp(-(h - h0) / H)


def drag_dv_coefficient(alt_km=None, a_m=None, e: float = 0.0, attitude_factor: float = 1.0,
                        cd: float = 2.2, n: int = 360) -> float:
    """Annual station-keeping delta-v per unit area-to-mass [m/s/yr per m^2/kg].

    Multiply by the array's area-to-mass ratio (and by ``atm_mult``) to get m/s/yr.
    Eccentric orbits are time-averaged over one revolution.
    """
    a = a_m if a_m is not None else R_EARTH + alt_km * 1e3
    if e == 0.0:
        v2 = MU_EARTH / a
        rho = atmospheric_density((a - R_EARTH) / 1e3)
        mean = rho * v2
    else:
        from .orbits import solve_kepler
        M = np.linspace(0, 2 * np.pi, n, endpoint=False)
        E = solve_kepler(M, e)
        r = a * (1 - e * np.cos(E))
        v2 = MU_EARTH * (2.0 / r - 1.0 / a)
        mean = float(np.mean(atmospheric_density((r - R_EARTH) / 1e3) * v2))
    return float(0.5 * cd * mean * attitude_factor * JULIAN_YEAR_S)


def sun_tracking_attitude_factor(mean_abs_cos_beta: float, sunlit_fraction: float) -> float:
    """Orbit-mean projected-area fraction of a Sun-tracking flat array (edge-on in eclipse)."""
    return float((2.0 / np.pi) * mean_abs_cos_beta * sunlit_fraction + 0.05 * (1.0 - sunlit_fraction))


def srp_dv_coefficient(reflectivity: float = 0.3) -> float:
    """Eccentricity-control delta-v from solar radiation pressure [m/s/yr per m^2/kg].

    For a high orbit the SRP-forced eccentricity circle has radius 3 a_srp / (2 n_sun v);
    holding it costs ~pi v e_srp per year = 1.5 pi a_srp / n_sun.
    """
    a_srp_per_am = TSI_1AU / C_LIGHT * (1.0 + reflectivity)
    return float(1.5 * np.pi * a_srp_per_am / SUN_SYNC_RATE)


def debris_flux(alt_km, size: str = "10cm"):
    """Damage-relevant debris flux [impacts m^-2 yr^-1] for objects larger than ``size``.

    GEO and MEO values are scaled by (v_rel / 10 km/s)^2 so that they are equivalent in
    impact energy to LEO hypervelocity impacts.
    """
    h = np.asarray(alt_km, float)
    f1 = (2.0e-4 * np.exp(-((h - 850.0) / 250.0) ** 2)        # 700-1,000 km debris belt
          + 5.0e-5 * np.exp(-((h - 1450.0) / 150.0) ** 2)     # ~1,400-1,500 km shell
          + 6.0e-6 * np.exp(-(h - 300.0) / 500.0)              # low LEO tail
          + 1.0e-8 / (1.0 + np.exp(-(h - 2000.0) / 300.0))    # sparse MEO
          + 3.0e-8 * np.exp(-((h - 35786.0) / 800.0) ** 2))   # GEO ring (energy-equivalent)
    if size == "1cm":
        return f1
    if size == "10cm":
        return f1 / 25.0
    if size == "1mm":
        return f1 * 100.0
    raise ValueError(size)


def disposal_dv(alt_km: float, e: float = 0.0, a_m: float | None = None) -> float:
    """End-of-life disposal delta-v [m/s]: LEO re-entry (perigee to 50 km) or graveyard."""
    a = a_m if a_m is not None else R_EARTH + alt_km * 1e3
    if e > 0.0:
        return 30.0
    if alt_km < 2000.0:
        r1, r2 = a, R_EARTH + 50e3
        at = 0.5 * (r1 + r2)
        v1 = np.sqrt(MU_EARTH / r1)
        v_apo = np.sqrt(MU_EARTH * (2.0 / r1 - 1.0 / at))
        return float(v1 - v_apo)
    if abs(alt_km - 35786.0) < 2000.0:
        return 11.0          # +300 km graveyard
    return 20.0
