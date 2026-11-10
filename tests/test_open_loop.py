"""Open-loop counterfactual and control-retention diagnostic: exact properties."""
import jax.numpy as jnp
import numpy as np

from carbon_world_model.plan import optimize as O
from carbon_world_model.plan.diagnostics import control_retention
from carbon_world_model.plan.scenario import build_scenario


def _scen(panel, cfg):
    return build_scenario(panel, cfg, end_year=2035)


def test_natural_override_with_own_bau_changes_reproduces_bau(panel, cfg, wm, params):
    scen = _scen(panel, cfg)
    tr = O._rollout(wm, params, scen, None)
    tr2 = O._rollout(wm, params, scen, None, {"dI_nat": tr["dI_nat"], "ds_nat": tr["ds_nat"]})
    for k in ("co2", "D", "shares"):
        np.testing.assert_allclose(np.asarray(tr2[k]), np.asarray(tr[k]), rtol=1e-6, atol=1e-6)


def test_open_loop_intensity_gap_is_exactly_the_push(panel, cfg, wm, params):
    scen = _scen(panel, cfg)
    bau = O.business_as_usual(wm, params, scen)
    ol = O.max_ambition_open_loop(wm, params, scen)
    gdp = np.asarray(scen.gdp)[:, -1]
    gap = np.log(bau.D[:, -1] / gdp) - np.log(ol.D[:, -1] / gdp)
    np.testing.assert_allclose(gap, ol.h[:, :-1].sum(1), rtol=1e-4)


def test_open_loop_max_ambition_is_below_bau(panel, cfg, wm, params):
    scen = _scen(panel, cfg)
    bau, ol = O.business_as_usual(wm, params, scen), O.max_ambition_open_loop(wm, params, scen)
    assert ol.cum_gt < bau.cum_gt
    assert ol.global_co2[-1] < bau.global_co2[-1]


def test_intensity_retention_is_one_when_dynamics_ignore_the_state(panel, cfg, wm, params):
    # Zero the output-head weights: the learned intensity change no longer depends on the
    # state, so the fitted and open-loop runs must agree on intensity.
    p = dict(params)
    for head in ("mix_head", "int_head"):
        p[head] = {"w": jnp.zeros_like(params[head]["w"]), "b": params[head]["b"]}
    scen = _scen(panel, cfg)
    bau, mx = O.business_as_usual(wm, p, scen), O.max_ambition(wm, p, scen)
    ol = O.max_ambition_open_loop(wm, p, scen)
    r = control_retention(bau, mx, ol, scen)
    assert abs(r["intensity_retention"] - 1.0) < 1e-4
