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


def test_clean_push_removes_fossil_at_its_bound_even_after_a_fuel_runs_out(panel, cfg, wm, params):
    """With natural changes switched off, the max-effort push must cut the fossil share by
    g_max every year until fossil is gone. The pre-2026-11-13 push was split by FIXED fossil
    weights; aimed entirely at coal here, it stalls once coal runs out while oil and gas
    remain. The current push follows the current fossil mix and does not."""
    import dataclasses
    import numpy as np
    import jax.numpy as jnp
    from carbon_world_model.config import FUELS
    scen = build_scenario(panel, cfg, end_year=2100)
    coal_only = np.zeros_like(scen.fossil_weights)
    coal_only[:, FUELS.index("coal")] = 1.0
    scen = dataclasses.replace(scen, fossil_weights=coal_only)   # only the old form reads it
    zero = {"dI_nat": jnp.zeros((scen.C, scen.Tf)), "ds_nat": jnp.zeros((scen.C, scen.Tf, len(FUELS)))}
    big = jnp.full((scen.C, scen.Tf), 8.0)
    ctrl, g, _ = O._controls(big, big, scen)
    gm = float(np.asarray(g)[0, 0])

    def fossil(controls):
        sh = np.asarray(O._rollout(wm, params, scen, controls, zero)["shares"])
        return sh[:, :, :3].sum(-1)

    new = fossil(ctrl)
    step = new[:, :-1] - new[:, 1:]
    active = new[:, 1:] > gm
    np.testing.assert_allclose(step[active], gm, rtol=1e-3)
    assert np.all(new[:, -1] < 1e-6)
    shift = scen.clean_weights[None, None, :] - scen.fossil_weights[:, None, :]
    old = fossil({"u_int": jnp.zeros((scen.C, scen.Tf)),
                  "u_deploy": jnp.full((scen.C, scen.Tf, 1), gm) * shift})
    oil_gas = panel.shares()[:, -1, 1:3].sum(-1) > 0.05
    assert np.all(old[oil_gas, -1] > 0.05)   # the old form leaves oil and gas in place


def test_process_lever_bound_and_effect(panel, cfg, wm, params):
    """Max effort cuts process CO2 by exactly p_max per year (compounding) from the year
    after it starts, and the lever only ever lowers emissions."""
    import numpy as np
    import jax.numpy as jnp
    from carbon_world_model.plan.scenario import process_bound
    scen = build_scenario(panel, cfg, end_year=2050)
    assert 0.0 < scen.p_max == process_bound(panel) <= 0.035
    big = jnp.full((scen.C, scen.Tf), 8.0)
    ctrl, _, _ = O._controls(big, big, scen)
    proc = np.asarray(O._abated_process(jnp.asarray(scen.process), ctrl))
    p = float(np.asarray(ctrl["u_proc"])[0, 0])
    np.testing.assert_allclose(proc[:, 0], scen.process[:, 0])
    np.testing.assert_allclose(proc[:, -1], scen.process[:, -1] * np.exp(-p * (scen.Tf - 1)), rtol=1e-5)
    off = dict(ctrl, u_proc=jnp.zeros_like(ctrl["u_proc"]))
    co2_on = np.asarray(O._rollout(wm, params, scen, ctrl)["co2"]).sum(0)
    co2_off = np.asarray(O._rollout(wm, params, scen, off)["co2"]).sum(0)
    assert np.all(co2_on[1:] < co2_off[1:]) and co2_on[0] == co2_off[0]
