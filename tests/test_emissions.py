"""Mechanistic emissions layer: physics and differentiability."""
import jax
import jax.numpy as jnp
import numpy as np

from carbon_world_model.config import FUELS, FOSSIL_FUELS
from carbon_world_model.model.emissions import (
    estimate_emission_factors, energy_to_co2, reconstruction_report, FOSSIL_IDX,
)


def test_emission_factor_ordering(panel):
    ef = estimate_emission_factors(panel, "nnls")
    f = ef.fossil
    assert f["coal"] > f["oil"] > f["gas"] > 0  # physical ordering
    assert f["coal"] < 0.5 and f["gas"] > 0.1   # plausible magnitudes (MtCO2/TWh)


def test_low_carbon_zero_ef(panel):
    ef = estimate_emission_factors(panel, "nnls").as_full()
    for fuel in ("nuclear", "hydro", "wind", "solar"):
        assert ef[FUELS.index(fuel)] == 0.0


def test_reconstruction_quality(panel):
    ef = estimate_emission_factors(panel, "nnls")
    rep = reconstruction_report(panel, ef)
    assert rep["r2"] > 0.99          # physics reconstructs reported CO2
    assert rep["median_abs_pct_err"] < 0.10


def test_gradient_equals_ef():
    ef = jnp.array([0.3, 0.2, 0.19] + [0.0] * (len(FUELS) - 3))
    e = jnp.ones((1, len(FUELS))) * 100.0
    g = jax.grad(lambda x: energy_to_co2(x, ef).sum())(e)
    assert np.allclose(np.asarray(g[0]), np.asarray(ef), atol=1e-6)


def test_clean_swap_reduces_co2():
    ef = jnp.array([0.3, 0.2, 0.19] + [0.0] * (len(FUELS) - 3))
    e = jnp.ones((1, len(FUELS))) * 100.0
    e2 = e.at[0, 0].add(-50).at[0, FUELS.index("solar")].add(50)  # coal -> solar
    assert float(energy_to_co2(e2, ef).sum()) < float(energy_to_co2(e, ef).sum())
