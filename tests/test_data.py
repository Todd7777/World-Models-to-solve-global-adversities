"""Panel data invariants."""
import numpy as np


def test_shapes(panel):
    C, T, F = panel.energy.shape
    assert panel.n_countries == C and panel.n_years == T and panel.n_fuels == F
    assert panel.co2.shape == (C, T)
    assert panel.static.shape[0] == C


def test_shares_on_simplex(panel):
    sh = panel.shares()
    s = sh.sum(axis=2)
    assert np.allclose(s[panel.mask.astype(bool)], 1.0, atol=1e-5)
    assert (sh >= -1e-9).all()


def test_no_nans_and_nonneg(panel):
    for a in (panel.energy, panel.co2, panel.gdp, panel.population, panel.primary_energy):
        assert np.isfinite(a).all()
        assert (a >= -1e-6).all()


def test_primary_energy_is_fuel_sum(panel):
    assert np.allclose(panel.primary_energy, panel.energy.sum(axis=2), atol=1e-6)


def test_co2_reported_matches_sources(panel):
    # Reported CO2 should be close to combustion sources + process term.
    recon = panel.co2_by_source.sum(axis=2) + panel.process_co2
    m = panel.mask.astype(bool)
    rel = np.abs(recon - panel.co2)[m] / np.clip(panel.co2[m], 1e-6, None)
    assert np.median(rel) < 0.05
