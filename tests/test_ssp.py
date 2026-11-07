"""SSP driver pipeline: interpolation, harmonisation, coverage, scenario logic."""
import numpy as np
import pytest

from carbon_world_model.data.ssp import (
    SSPS, LEGACY_ISO, harmonise, load_processed, loglinear_interp, ssp_driver_paths,
)


def test_loglinear_interp_nodes_and_midpoint():
    yrs = np.array([2010, 2020])
    vals = np.array([100.0, 400.0])
    out = loglinear_interp(yrs, vals, np.array([2010, 2015, 2020, 2025]))
    assert np.isclose(out[0], 100) and np.isclose(out[2], 400)
    assert np.isclose(out[1], 200.0)          # geometric midpoint
    assert np.isnan(out[3])                  # no extrapolation


def test_harmonise_ratio_method():
    years = np.arange(2020, 2031)
    series = np.vstack([np.linspace(10, 20, 11), np.linspace(5, 5, 11)])
    out = harmonise(np.array([100.0, 7.0]), series, years, 2020, np.array([2020, 2030]))
    assert np.allclose(out[:, 0], [100.0, 7.0])      # base == observed
    assert np.isclose(out[0, 1], 200.0)              # growth factor 2 preserved
    assert np.isclose(out[1, 1], 7.0)


@pytest.mark.parametrize("vintage", ["v9", "v3.2"])
def test_full_coverage(vintage, panel):
    s = load_processed(vintage)
    assert s.avail.shape == (len(SSPS), panel.n_countries)
    assert s.avail.all(), "every (scenario, country) pair should have GDP and population"


def test_no_gaps_in_used_windows(panel):
    v9, v3 = load_processed("v9"), load_processed("v3.2")
    g9, _ = ssp_driver_paths(panel, v9, "SSP2", 2015, np.arange(2015, 2023))
    g3, p3 = ssp_driver_paths(panel, v3, "SSP2", 2022, np.arange(2022, 2101))
    assert np.isfinite(g9).all() and np.isfinite(g3).all() and np.isfinite(p3).all()


def test_base_year_matches_observed(panel):
    g, p = ssp_driver_paths(panel, load_processed("v3.2"), "SSP3", 2022, np.arange(2022, 2051))
    assert np.allclose(g[:, 0], panel.gdp[:, -1]) and np.allclose(p[:, 0], panel.population[:, -1])


def test_ssp_narrative_ordering(panel):
    """SSP5 (fossil-fuelled development) out-grows SSP3 (regional rivalry) in GDP by 2050."""
    v3 = load_processed("v3.2")
    yrs = np.arange(2022, 2051)
    g5, _ = ssp_driver_paths(panel, v3, "SSP5", 2022, yrs)
    g3, _ = ssp_driver_paths(panel, v3, "SSP3", 2022, yrs)
    assert np.mean(g5[:, -1] > g3[:, -1]) > 0.9


def test_legacy_iso_alias_present():
    assert LEGACY_ISO["ROM"] == "ROU"
