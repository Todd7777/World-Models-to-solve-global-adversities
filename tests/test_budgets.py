"""Published remaining carbon budgets and budget accounting."""
import numpy as np

from carbon_world_model.climate import budgets as B


def test_table_values_and_rebasing():
    assert B.budget_gt(1.5, 0.5, base_year=2025) == 130.0
    assert B.budget_gt(2.0, 0.67, base_year=2025) == 870.0
    np.testing.assert_allclose(B.budget_gt(1.5, 0.5, base_year=2023), 130 + 40.6 + 41.6)
    np.testing.assert_allclose(B.budget_gt(2.0, 0.5, non_co2_shift=-200.0), 1050 + 82.2 - 200)


def test_likelihood_inverts_the_table_and_clips():
    for t, row in B.RCB_TABLE.items():
        for q, b in zip(B.LIKELIHOODS, row):
            np.testing.assert_allclose(B.likelihood_below(t, b, base_year=2025), q)
    assert B.likelihood_below(1.5, 10_000.0) == 0.17   # far over budget: "17% or less"
    assert B.likelihood_below(2.0, 0.0) == 0.83        # far under: "83% or more"
    x = np.linspace(0, 3000, 50)
    assert np.all(np.diff(B.likelihood_below(2.0, x)) <= 0)


def test_tcre_only_budget_reproduces_the_old_numbers():
    assert round(B.tcre_only_budget(1.5)) == 578
    assert round(B.tcre_only_budget(2.0)) == 1689


def test_accounting_terms():
    assert B.land_use_co2(mode="zero") == 0.0
    np.testing.assert_allclose(B.land_use_co2(mode="constant"), 4.2 * 28)
    np.testing.assert_allclose(B.land_use_co2(mode="linear_to_zero"), 4.2 * 28 / 2)
    np.testing.assert_allclose(B.tail_after(20.0, 2070), 20.0 * 19 / 2)   # 2051..2069 > 0
    assert B.tail_after(20.0, None) == 0.0
    np.testing.assert_allclose(B.total_until_net_zero(700.0, 20.0, "zero", None), 700.0)
