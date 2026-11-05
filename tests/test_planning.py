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
    import jax.numpy as jnp
    finals = []
    for raw in (-10.0, -1.0, 2.0, 10.0):
        rg = jnp.full((scen.C, scen.Tf), raw)
        ctrl, g, h = O._controls(rg, rg, scen)
        tr = O._rollout(wm, params, scen, ctrl)
        finals.append(float(jnp.sum(tr["co2"][:, -1])))
    # more effort -> monotonically lower 2050 emissions
    assert all(finals[i] > finals[i + 1] for i in range(len(finals) - 1))


def test_tcre_translation(cfg):
    clim = cfg.climate
    w = tcre.warming_from_cumulative(1000.0, clim)  # 1000 Gt = 1 Tt
    assert abs(w["central"] - clim.tcre_degC_per_ttco2) < 1e-9
    assert w["low"] < w["central"] < w["high"]
    assert tcre.remaining_budget_gtco2(1.5, clim) > 0
