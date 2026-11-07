"""Mechanistic emissions layer (Module A).

CO2 from combustion is, to very good approximation, a linear function of the energy
burned by fuel:  CO2_energy = sum_f  EF_f * energy_f,  with EF_f a physical emission
factor (MtCO2 per TWh of primary energy). Non-combustion CO2 (cement, flaring, other
industry) is carried as an additive process term.

This layer is deliberately *not* learned from scratch: the emission factors are
pinned to the data by non-negative least squares and then held fixed, so the world
model cannot "explain away" emissions by distorting physics. It is fully
differentiable (plain multiply-add), so it composes with the learned dynamics and the
planner.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
from scipy.optimize import nnls

from ..config import FUELS, FOSSIL_FUELS

# Indices of the fossil fuels within the FUELS tuple.
FOSSIL_IDX = tuple(FUELS.index(f) for f in FOSSIL_FUELS)


@dataclass(frozen=True)
class EmissionFactors:
    """Emission factors in MtCO2 per TWh, one per fuel (0 for low-carbon)."""
    ef: np.ndarray  # [F]

    @property
    def fossil(self) -> dict[str, float]:
        return {f: float(self.ef[FUELS.index(f)]) for f in FOSSIL_FUELS}

    def as_full(self) -> np.ndarray:
        return self.ef


def estimate_emission_factors(panel, method: str = "nnls", upto_idx: int | None = None) -> EmissionFactors:
    """Fit per-fuel emission factors from the panel.

    Target is the per-fuel combustion CO2 (coal+oil+gas sources). We fit the three
    fossil EFs jointly by NNLS against fossil energy use, pooling all valid
    country-years. Low-carbon fuels are assigned EF=0 by construction.
    """
    energy = panel.energy  # [C, T, F]
    co2_src = panel.co2_by_source  # [C, T, 3] coal/oil/gas
    mask = panel.mask.astype(bool).copy()  # [C, T]
    if upto_idx is not None:          # leak-free: only years up to the fit cutoff
        mask[:, upto_idx + 1:] = False

    # Build pooled design matrix over valid country-years.
    A = energy[:, :, FOSSIL_IDX][mask]           # [N, 3] fossil energy (TWh)
    b = co2_src[mask].sum(axis=1)                 # [N]   combustion CO2 (Mt)
    keep = np.isfinite(A).all(1) & np.isfinite(b)
    A, b = A[keep], b[keep]

    if method == "nnls":
        coef, _ = nnls(A, b)
    elif method == "ratio":
        # Robust per-fuel median of source/consumption.
        coef = np.array([
            np.median(co2_src[mask][:, i] / np.clip(A[:, i], 1e-6, None))
            for i in range(3)
        ])
    else:
        raise ValueError(method)

    ef = np.zeros(len(FUELS))
    for j, idx in enumerate(FOSSIL_IDX):
        ef[idx] = coef[j]
    return EmissionFactors(ef=ef)


def energy_to_co2(energy, ef, process_co2=None):
    """Differentiable forward: energy mix -> production CO2 (Mt).

    Parameters
    ----------
    energy : array [..., F]  primary-energy consumption by fuel (TWh)
    ef     : array [F]       emission factors (MtCO2/TWh)
    process_co2 : array [...] or None   additive non-energy CO2 (Mt)
    """
    ef = jnp.asarray(ef)
    co2 = jnp.tensordot(jnp.asarray(energy), ef, axes=([-1], [0]))
    if process_co2 is not None:
        co2 = co2 + jnp.asarray(process_co2)
    return co2


def reconstruction_report(panel, ef: EmissionFactors) -> dict:
    """Validate the mechanistic layer: reconstruct reported CO2 from energy mix."""
    mask = panel.mask.astype(bool)
    pred = np.asarray(energy_to_co2(panel.energy, ef.as_full(), panel.process_co2))
    y = panel.co2
    p, t = pred[mask], y[mask]
    resid = p - t
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    mape = float(np.mean(np.abs(resid) / np.clip(np.abs(t), 1e-6, None)))
    # Per-country R2 of the energy-only part vs reported energy CO2.
    return {
        "r2": r2,
        "mape": mape,
        "rmse_mt": float(np.sqrt(np.mean(resid ** 2))),
        "median_abs_pct_err": float(np.median(np.abs(resid) / np.clip(np.abs(t), 1e-6, None))),
        "emission_factors": ef.fossil,
        "n": int(mask.sum()),
    }
