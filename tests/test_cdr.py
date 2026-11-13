"""Novel CDR path and peak-cumulative accounting."""
import numpy as np

from carbon_world_model.climate import cdr


def test_cdr_path_starts_at_the_measured_level_and_saturates():
    years = np.arange(2023, 2101)
    x = cdr.novel_cdr_path(years)
    np.testing.assert_allclose(x[years == 2025], cdr.CDR_START_GT)
    assert np.all(np.diff(x) > 0)
    assert x[-1] < cdr.CDR_CEILINGS["central"] and x[-1] > 0.99 * cdr.CDR_CEILINGS["central"]
    # early growth is the measured rate
    np.testing.assert_allclose(x[years == 2026] / x[years == 2025], np.exp(cdr.CDR_GROWTH), rtol=1e-3)
    # near the scenario level around 2050, as the source describes
    assert 0.75 * 3.5 < x[years == 2050] < 3.5


def test_ceilings_order_the_paths():
    years = np.arange(2023, 2101)
    lo, mid, hi = (cdr.novel_cdr_path(years, cdr.CDR_CEILINGS[k]) for k in ("low", "central", "high"))
    assert np.all(lo <= mid + 1e-12) and np.all(mid <= hi + 1e-12)


def test_peak_cumulative_and_net_zero_year():
    years = np.arange(2023, 2033)
    net = np.array([5, 4, 3, 2, 1, 0, -1, -2, -3, -4], dtype=float)
    peak, reached = cdr.peak_cumulative(net)
    assert peak == 15.0 and reached
    assert cdr.net_zero_year(years, net) == 2028
    still_rising = np.array([5, 4, 3, 2, 1], dtype=float)
    peak, reached = cdr.peak_cumulative(still_rising)
    assert peak == 15.0 and not reached
    assert cdr.net_zero_year(years[:5], still_rising) is None
