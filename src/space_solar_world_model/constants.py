"""Physical constants (SI units) for the Space Solar World Model.

Values are IAU/IERS/WGS-84 standards unless noted. Keeping them in one module makes
the provenance of every number auditable.
"""
from __future__ import annotations

import math

# --- Earth ------------------------------------------------------------------
MU_EARTH = 3.986004418e14          # m^3 s^-2, gravitational parameter (WGS-84)
R_EARTH = 6.378137e6               # m, equatorial radius (WGS-84)
J2_EARTH = 1.08262668e-3           # dynamic form factor (EGM-96)
OMEGA_EARTH = 7.2921150e-5         # rad s^-1, sidereal rotation rate
R_GEO = 42164.17e3                 # m, geostationary orbit radius
ALT_GEO_KM = (R_GEO - R_EARTH) / 1e3

# --- Sun --------------------------------------------------------------------
AU = 1.495978707e11                # m
R_SUN = 6.957e8                    # m, nominal solar radius (IAU 2015)
TSI_1AU = 1361.0                   # W m^-2, total solar irradiance at 1 AU
                                   # (Kopp & Lean 2011: 1360.8 +/- 0.5)
OBLIQUITY_J2000_DEG = 23.4393      # deg, mean obliquity of the ecliptic

# --- Moon / Lagrange points -------------------------------------------------
MOON_DISTANCE = 3.844e8            # m, mean Earth-Moon distance
R_MOON = 1.7374e6                  # m
SE_L1_DISTANCE = 1.4916e9          # m, Sun-Earth L1 distance from Earth's centre

# --- Misc -------------------------------------------------------------------
G0 = 9.80665                       # m s^-2, standard gravity (rocket equation)
C_LIGHT = 299_792_458.0            # m s^-1
DAY_S = 86400.0
TROPICAL_YEAR_DAYS = 365.24219
JULIAN_YEAR_S = 365.25 * DAY_S
HOURS_PER_YEAR = 8766.0            # Julian year in hours (used for annual energy)
# Mean motion of the Sun about the Earth: the RAAN rate a sun-synchronous orbit needs.
SUN_SYNC_RATE = 2.0 * math.pi / (TROPICAL_YEAR_DAYS * DAY_S)   # rad s^-1

DEG = math.pi / 180.0
