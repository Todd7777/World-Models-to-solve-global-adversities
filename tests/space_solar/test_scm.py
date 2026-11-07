"""Causal engine: interventions, abduction, counterfactuals, decision analysis."""
import numpy as np

from space_solar_world_model import CausalWorld, planner


def test_intervention_on_mechanism(world, geo_plan):
    cw = CausalWorld(world)
    base = cw.intervene(geo_plan.action, n=500, seed=1)["summary"]["median"]
    no_belts = cw.intervene(geo_plan.action, n=500, seed=1,
                            do={"fluence_rate": 0.0, "transfer_fluence": 0.0})["summary"]["median"]
    assert no_belts < base


def test_uninformative_evidence_leaves_prior(world, geo_plan):
    cw = CausalWorld(world)
    ev = {"weak": (lambda o, u: o["link_efficiency"], 0.7, 1e6, "abs")}
    post = cw.abduct(geo_plan.action, ev, n=2000, seed=2)
    assert post.ess > 1990


def test_evidence_moves_posterior_in_right_direction(world, geo_plan):
    cw = CausalWorld(world)
    # a pilot that observed a poor rectenna -> posterior RF->DC efficiency must fall
    low = {"rx": (lambda o, u: u["eta_rfdc"], 0.65, 0.02, "abs")}
    post = cw.abduct(geo_plan.action, low, n=4000, seed=3)
    prior_mean = np.mean(post.u["eta_rfdc"])
    post_mean = np.sum(post.w * post.u["eta_rfdc"])
    assert post_mean < prior_mean - 0.05


def test_counterfactual_is_paired_and_consistent(world, geo_plan):
    cw = CausalWorld(world)
    leo = planner.plan_beam(world, "leo_1000_53", "Las Vegas", "mw2.45", site_options=(12,))
    ev = {"eff": (lambda o, u: o["link_efficiency"], 0.6, 0.05, "log")}
    cf = cw.counterfactual(geo_plan.action, ev, {"LEO": leo.action}, n=3000, seed=4)
    assert cf["ess"] > 50
    assert cf["LEO"]["p_better_than_factual"] < 0.05     # LEO is dominated by GEO here
    assert cf["LEO"]["diff_posterior"]["median"] > 0


def test_decision_analysis_probabilities(world, geo_plan):
    cw = CausalWorld(world)
    da = cw.decision_analysis({"a": geo_plan.action, "b": geo_plan.action.with_design(d_t=600.0)},
                              n=400, seed=5)
    assert abs(sum(v["p_best"] for v in da.values()) - 1.0) < 1e-9
    assert all(v["expected_regret"] >= 0 for v in da.values())
