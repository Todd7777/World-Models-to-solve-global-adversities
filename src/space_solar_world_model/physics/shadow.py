"""Earth-shadow geometry.

Two models:

* ``illumination_fraction`` -- exact visible fraction of the solar disk for spherical Sun
  and occulting body (umbra, penumbra and annular cases; Montenbruck & Gill, *Satellite
  Orbits*, Sec. 3.4.2). Works for any position, including the Sun-Earth L2 point where
  Earth's disk is smaller than the Sun's.
* ``eclipse_fraction_circular`` -- closed-form cylindrical-shadow eclipse fraction of a
  circular orbit as a function of the solar beta angle (used for fast annual statistics;
  validated against the conical model in the tests).
"""
from __future__ import annotations

import numpy as np

from ..constants import R_EARTH, R_SUN


def illumination_fraction(r_sat, r_sun, body_radius: float = R_EARTH, r_body=None):
    """Fraction (0..1) of the solar disk visible from ``r_sat``.

    ``r_sat`` and ``r_sun`` are positions [m] relative to the occulting body's centre
    unless ``r_body`` (the body's position in the same frame) is given.
    """
    r_sat = np.asarray(r_sat, float)
    r_sun = np.asarray(r_sun, float)
    if r_body is not None:
        r_sat = r_sat - r_body
        r_sun = r_sun - r_body
    s_to_sun = r_sun - r_sat
    d_sun = np.linalg.norm(s_to_sun, axis=-1)
    d_body = np.linalg.norm(r_sat, axis=-1)
    a = np.arcsin(np.clip(R_SUN / d_sun, 0.0, 1.0))            # apparent solar radius
    b = np.arcsin(np.clip(body_radius / d_body, 0.0, 1.0))     # apparent body radius
    cos_c = np.sum(-r_sat * s_to_sun, axis=-1) / (d_body * d_sun)
    c = np.arccos(np.clip(cos_c, -1.0, 1.0))                   # apparent separation

    # Partial-overlap area of two disks (radii a, b; centre separation c).
    c_safe = np.maximum(c, 1e-12)
    x = (c_safe ** 2 + a ** 2 - b ** 2) / (2.0 * c_safe)
    y = np.sqrt(np.clip(a ** 2 - x ** 2, 0.0, None))
    area = (a ** 2 * np.arccos(np.clip(x / a, -1.0, 1.0))
            + b ** 2 * np.arccos(np.clip((c_safe - x) / b, -1.0, 1.0)) - c_safe * y)
    nu = 1.0 - area / (np.pi * a ** 2)

    nu = np.where(c >= a + b, 1.0, nu)                         # no occultation
    nu = np.where(c <= b - a, 0.0, nu)                         # total (umbra)
    nu = np.where((c <= a - b) & (a > b), 1.0 - (b / a) ** 2, nu)   # annular
    return np.clip(nu, 0.0, 1.0)


def beta_angle(inc, raan, sun_lambda, eps):
    """Solar beta angle [rad]: angle between the orbit plane and the Sun vector."""
    sb = (np.cos(sun_lambda) * np.sin(raan) * np.sin(inc)
          - np.sin(sun_lambda) * np.cos(eps) * np.cos(raan) * np.sin(inc)
          + np.sin(sun_lambda) * np.sin(eps) * np.cos(inc))
    return np.arcsin(np.clip(sb, -1.0, 1.0))


def eclipse_fraction_circular(radius_m, beta_rad, body_radius: float = R_EARTH):
    """Fraction of a circular orbit spent in the cylindrical shadow (0 if |beta| > beta*)."""
    r = np.asarray(radius_m, float)
    beta = np.asarray(beta_rad, float)
    cb = np.maximum(np.cos(beta), 1e-12)
    arg = np.sqrt(np.clip(r ** 2 - body_radius ** 2, 0.0, None)) / (r * cb)
    frac = np.arccos(np.clip(arg, -1.0, 1.0)) / np.pi
    beta_star = np.arcsin(np.clip(body_radius / r, -1.0, 1.0))
    return np.where(np.abs(beta) < beta_star, frac, 0.0)
