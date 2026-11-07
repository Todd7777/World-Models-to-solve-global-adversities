"""Robust planning across stacked SSP futures."""
import numpy as np

from carbon_world_model.plan import optimize as O
from carbon_world_model.plan.scenario import ssp_ensemble


def test_zero_controls_match_bau(panel, cfg, wm, params):
    scens = ssp_ensemble(panel, cfg, end_year=2035)
    ev = O.evaluate_controls(wm, params, scens)
    for s, cum in zip(scens, ev["cum_gt"]):
        assert np.isclose(cum, O.business_as_usual(wm, params, s).cum_gt, rtol=1e-4)


def test_ssp_ordering_of_bau(panel, cfg, wm, params):
    """Higher-growth SSP5 should emit more than SSP1 under business-as-usual."""
    scens = ssp_ensemble(panel, cfg, end_year=2040)
    cum = O.evaluate_controls(wm, params, scens)["cum_gt"]
    assert cum[4] > cum[0]


def test_robust_plan_meets_budget_everywhere(panel, cfg, wm, params):
    scens = ssp_ensemble(panel, cfg, end_year=2035)
    ev0 = O.evaluate_controls(wm, params, scens)
    mx = [O.max_ambition(wm, params, s).cum_gt for s in scens]
    budget = 0.5 * (max(mx) + min(ev0["cum_gt"]))  # feasible in every scenario
    r = O.robust_pathway(wm, params, scens, budget_gt=budget, penalty=20.0, steps=400)
    assert max(r["cum_gt"]) <= budget * 1.01
    assert 0 < r["cost"]
