"""Wireless power transmission from orbit to the ground.

Microwave links use the Goubau relation for an optimally tapered aperture pair,

    tau = sqrt(A_t A_r) / (lambda D),   eta_beam = 1 - exp(-tau^2),

which is identical to a Gaussian beam whose waist is a_t/sqrt(2) truncated at the
transmitter rim. The same Gaussian picture gives the on-axis intensity

    I_0 = P_tx A_t / (lambda D)^2,

which is capped by ionospheric-heating and safety limits (~230 W/m^2 at beam centre,
<=10 W/m^2 at the rectenna edge in the DOE/NASA reference system). The transmitter
itself can only shed so much heat, which caps radiated power per aperture area. Together
the two caps bound the power one transmitter can deliver: P <= sqrt(I_max q_max) lambda D. Seen from a site at
elevation ``el`` a ground rectenna of area A presents A sin(el) to the beam.

Laser links capture almost everything with small receivers but are blocked by cloud;
their beam spread has a floor set by pointing jitter and turbulence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import C_LIGHT


@dataclass(frozen=True)
class LinkTech:
    name: str
    kind: str                         # "microwave" or "laser"
    wavelength_m: float
    zenith_loss: float                # clear-air fractional loss at zenith
    max_peak_intensity: float | None  # W/m^2, beam centre (None = not limiting)
    max_edge_intensity: float | None  # W/m^2 at receiver edge
    weather_blocked: bool             # True if cloud interrupts the link
    divergence_floor_rad: float = 0.0
    max_tx_flux: float | None = None  # W/m^2 radiated per transmitter area (thermal limit)
    max_rx_flux: float | None = None  # W/m^2 average on the receiver (laser PV cooling limit)


# Transmitter thermal limit: the 1979 DOE/NASA reference phased array peaked at ~21 kW/m^2
# at its centre with a tapered illumination; we cap the aperture-average at 10 kW/m^2.
MICROWAVE_2G45 = LinkTech("microwave 2.45 GHz", "microwave", C_LIGHT / 2.45e9, 0.010, 230.0, 10.0,
                          False, max_tx_flux=1.0e4)
MICROWAVE_5G8 = LinkTech("microwave 5.8 GHz", "microwave", C_LIGHT / 5.8e9, 0.020, 230.0, 10.0,
                         False, max_tx_flux=1.0e4)
# Lasers: ~2 MW per m^2 through the beam director (high-energy-laser practice is ~1 MW/m^2)
# and ~100 kW/m^2 (100 suns) on actively cooled laser-power-converter receivers.
LASER_1064 = LinkTech("laser 1064 nm", "laser", 1.064e-6, 0.10, None, None, True, 2.0e-6,
                      max_tx_flux=2.0e6, max_rx_flux=1.0e5)
LINKS = {"mw2.45": MICROWAVE_2G45, "mw5.8": MICROWAVE_5G8, "laser": LASER_1064}


def airmass(el_rad, xp=np):
    """Relative optical air mass (Kasten & Young 1989); finite at the horizon."""
    el_deg = xp.maximum(el_rad * 180.0 / np.pi, 0.0)
    return 1.0 / (xp.sin(xp.maximum(el_rad, 0.0)) + 0.50572 * (el_deg + 6.07995) ** -1.6364)


def atmospheric_transmission(el_rad, zenith_loss: float, xp=np):
    return xp.exp(np.log(1.0 - zenith_loss) * airmass(el_rad, xp))


def capture_efficiency(tech: LinkTech, a_t_m2, a_r_perp_m2, distance_m, xp=np):
    """Fraction of transmitted power intercepted by the receiver."""
    lam = tech.wavelength_m
    if tech.kind == "microwave":
        tau2 = a_t_m2 * a_r_perp_m2 / (lam * distance_m) ** 2
        return 1.0 - xp.exp(-tau2)
    w0 = xp.sqrt(a_t_m2 / (2.0 * np.pi))                     # waist = a_t / sqrt(2)
    theta = xp.sqrt((lam / (np.pi * w0)) ** 2 + tech.divergence_floor_rad ** 2)
    w_d = distance_m * theta
    return 1.0 - xp.exp(-2.0 * (a_r_perp_m2 / np.pi) / w_d ** 2)


def peak_intensity(p_tx_w, a_t_m2, wavelength_m, distance_m):
    """On-axis intensity normal to the beam [W/m^2] (far field)."""
    return p_tx_w * a_t_m2 / (wavelength_m * distance_m) ** 2


def max_power_for_peak(tech: LinkTech, a_t_m2, distance_m):
    """Largest transmit power that keeps the beam centre at the intensity cap."""
    if tech.max_peak_intensity is None:
        return np.inf
    return tech.max_peak_intensity * (tech.wavelength_m * distance_m) ** 2 / a_t_m2


def edge_intensity(p_tx_w, a_t_m2, a_r_perp_m2, wavelength_m, distance_m, xp=np):
    """Intensity at the edge of a circular receiver: I_0 exp(-tau^2)."""
    tau2 = a_t_m2 * a_r_perp_m2 / (wavelength_m * distance_m) ** 2
    return peak_intensity(p_tx_w, a_t_m2, wavelength_m, distance_m) * xp.exp(-tau2)
