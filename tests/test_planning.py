"""Planning behaves causally: more effort -> fewer emissions, within feasibility."""
import numpy as np

from carbon_world_model.plan.scenario import build_scenario, feasibility_bounds
from carbon_world_model.plan import optimize as O
from carbon_world_model.climate import tcre


def test_feasibility_bounds_positive(panel):
    g, h = feasibility_bounds(panel)
    assert 0 < g <= 0.025 and 0 < h <= 0.035


def test_scenario_build(panel, cfg):
    scen = build_scenario(panel, cfg, end_year=2050)
    assert scen.years[0] == int(panel.years[-1]) and scen.years[-1] == 2050
    assert np.isfinite(scen.gdp).all() and (scen.gdp > 0).all()
    # control shift vector is mass-conserving (reallocation, not creation)
    shift = scen.clean_weights[None, :] - scen.fossil_weights
    assert np.allclose(shift.sum(axis=1), 0.0, atol=1e-6)


def test_max_ambition_below_bau(panel, cfg, wm, params):
    scen = build_scenario(panel, cfg, end_year=2050)
    bau = O.business_as_usual(wm, params, scen)
    mx = O.max_ambition(wm, params, scen)
    assert mx.global_co2[-1] < bau.global_co2[-1]
    assert mx.cum_gt < bau.cum_gt


def test_effort_monotonic(panel, cfg, wm, params):
    scen = build_scenario(panel, cfg, end_year=2050)
    finals = [float(O.effort_pathway(wm, params, scen, frac).global_co2[-1])
              for frac in (0.0, 0.25, 0.75, 1.0)]
    # more effort -> monotonically lower 2050 emissions
    assert all(finals[i] > finals[i + 1] for i in range(len(finals) - 1))


def _rates(run, scen):
    """Each world's annual fall in log intensity and gain in low-carbon share."""
    li = np.log(np.asarray(run["D"], dtype=float)) - np.log(scen.gdp)[None]
    lowc = np.asarray(run["shares"], dtype=float)[..., O.LOWC_IDX].sum(axis=-1)
    return li[..., :-1] - li[..., 1:], np.diff(lowc, axis=-1)


def _fossil_left(run):
    """[W, C, Tf-1] steps after which the country still burns fossil fuel, so the plan's
    low-carbon gain is not capped by running out of it."""
    return np.asarray(run["shares"], dtype=float)[..., 1:, :][..., O.FOSSIL_IDX].sum(-1) > 1e-6


def test_zero_plan_is_business_as_usual(panel, cfg, wm, params):
    scen = build_scenario(panel, cfg, end_year=2050)
    W = O.make_worlds(wm, O._one(params), [scen])
    zero = np.zeros((scen.C, scen.Tf))
    run, bau = O.apply_plan(W, zero, zero), O.apply_plan(W)
    for k in ("co2", "D", "shares"):
        np.testing.assert_allclose(run[k], bau[k], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(bau["co2"][0], O.business_as_usual(wm, params, scen).co2)


def test_plans_never_exceed_the_feasibility_limits(panel, cfg, wm, params):
    """A plan adds min(plan, room below the limit) to business as usual's rates, so it
    never takes a country beyond the limits, and maximum effort reaches them: each year's
    intensity fall is max(BAU fall, h_max), and the low-carbon gain max(BAU gain, g_max)
    until the country has no fossil fuel left to displace."""
    scen = build_scenario(panel, cfg, end_year=2050)
    W = O.make_worlds(wm, O._one(params), [scen])
    fall_b, gain_b = _rates(O.apply_plan(W), scen)
    rng = np.random.default_rng(0)
    rand = rng.uniform(0, 2, (2, scen.C, scen.Tf)) * np.array([scen.g_max, scen.h_max])[:, None, None]
    for name, (g, h) in (("random", (rand[0], rand[1])), ("maximum", O.max_plan(W))):
        g, h = np.asarray(g), np.asarray(h)
        run = O.apply_plan(W, g, h)
        fall, gain = _rates(run, scen)
        ok = _fossil_left(run)
        assert (np.asarray(run["shares"]) >= 0).all(), name
        np.testing.assert_allclose(np.asarray(run["shares"]).sum(-1), 1.0, atol=1e-5, err_msg=name)
        added_h = np.minimum(h, np.asarray(W.gap_h))[..., :-1]
        added_g = np.minimum(g, np.asarray(W.gap_g))[..., :-1]
        np.testing.assert_allclose(fall, fall_b + added_h, atol=1e-5, err_msg=name)
        np.testing.assert_allclose(gain[ok], (gain_b + added_g)[ok], atol=1e-5, err_msg=name)
        assert (fall <= np.maximum(fall_b, scen.h_max) + 1e-5).all(), name
        assert (gain[ok] <= np.maximum(gain_b, scen.g_max)[ok] + 1e-5).all(), name
        assert ok.mean() > 0.75, (name, ok.mean())
    np.testing.assert_allclose(fall, np.maximum(fall_b, scen.h_max), atol=1e-5)
    np.testing.assert_allclose(gain[ok], np.maximum(gain_b, scen.g_max)[ok], atol=1e-5)
    assert (~ok).any()  # some countries run out of fossil fuel, so that case is covered too
    mx = O.max_ambition(wm, params, scen)
    np.testing.assert_allclose(mx.g, np.asarray(W.gap_g[0]))
    np.testing.assert_allclose(mx.cost, float(O._cost(mx.g, mx.h, scen)), rtol=1e-6)


def test_tcre_translation(cfg):
    clim = cfg.climate
    w = tcre.warming_from_cumulative(1000.0, clim)  # 1000 Gt = 1 Tt
    assert abs(w["central"] - clim.tcre_degC_per_ttco2) < 1e-9
    assert w["low"] < w["central"] < w["high"]
    assert tcre.remaining_budget_gtco2(1.5, clim) > 0
