"""Relaxing the frontier's assumptions: rebound and crowding out in plan/optimize.py, and
scripts/plan_sensitivity.py.

Rebound and crowding out must take back exactly their share of what a plan achieves,
charge the same effort, and only ever raise emissions. The committed table must come
from the committed ensemble and, under the stated assumptions, agree with plan.json.
"""
import importlib.util
import json

import numpy as np
import pytest

from carbon_world_model.config import default_config
from carbon_world_model.plan import optimize as O
from carbon_world_model.plan.ensemble import ENSEMBLE_DIR, load_ensemble, stack_members
from carbon_world_model.plan.scenario import build_scenario


def _rates(run, scen):
    """Each world's annual fall in log intensity and gain in low-carbon share."""
    li = np.log(np.asarray(run["D"], dtype=float)) - np.log(scen.gdp)[None]
    lowc = np.asarray(run["shares"], dtype=float)[..., O.LOWC_IDX].sum(axis=-1)
    return li[..., :-1] - li[..., 1:], np.diff(lowc, axis=-1)


def _script():
    spec = importlib.util.spec_from_file_location("plan_sensitivity", "scripts/plan_sensitivity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rebound_and_crowding_out_take_back_their_share(panel, cfg, wm, params):
    scen = build_scenario(panel, cfg, end_year=2050)
    full = O.make_worlds(wm, O._one(params), [scen])
    part = O.make_worlds(wm, O._one(params), [scen], rebound=0.3, crowd_out=0.4)
    g, h = (np.asarray(x) for x in O.max_plan(full))
    fall_b, gain_b = _rates(O.apply_plan(full), scen)
    run_f, run_p = O.apply_plan(full, g, h), O.apply_plan(part, g, h)
    (fall_f, gain_f), (fall_p, gain_p) = _rates(run_f, scen), _rates(run_p, scen)
    np.testing.assert_allclose(fall_p - fall_b, 0.7 * (fall_f - fall_b), atol=1e-5)
    # Where the full plan leaves fossil fuel to displace, so does the partial one.
    left = np.asarray(run_f["shares"])[..., 1:, :][..., O.FOSSIL_IDX].sum(-1) > 1e-6
    assert left.mean() > 0.75
    np.testing.assert_allclose((gain_p - gain_b)[left], 0.6 * (gain_f - gain_b)[left], atol=1e-5)

    co2_b, co2_f, co2_p = (np.asarray(r["co2"]) for r in (O.apply_plan(full), run_f, run_p))
    assert (co2_p >= co2_f - 1e-3).all() and (co2_p <= co2_b + 1e-3).all()
    assert co2_f[..., -1].sum() < co2_p[..., -1].sum() < co2_b[..., -1].sum()

    # The same plan costs the same effort; it achieves less.
    rf, rp = O._result(full, g, h, scen), O._result(part, g, h, scen)
    assert rp.cost == pytest.approx(rf.cost, rel=1e-9) and rf.cost > 0
    np.testing.assert_allclose(rp.g, 0.6 * rf.g)
    np.testing.assert_allclose(rp.h, 0.7 * rf.h)


def test_shares_outside_zero_to_one_are_refused(panel, cfg, wm, params):
    scen = build_scenario(panel, cfg, end_year=2030)
    for kw in ({"rebound": 1.5}, {"crowd_out": -0.1}):
        with pytest.raises(ValueError, match="share in"):
            O.make_worlds(wm, O._one(params), [scen], **kw)


def test_committed_plan_sensitivity_matches_the_ensemble_and_plan_json(panel, wm):
    """results/tables/plan_sensitivity.json comes from the committed ensemble (rerun
    scripts/plan_sensitivity.py whenever scripts/02b_train_ensemble.py or 04_plan.py is
    rerun), and its stated-assumptions row is what 04_plan.py planned with."""
    with open("results/tables/plan_sensitivity.json") as fh:
        committed = json.load(fh)
    cfg = default_config()
    seeds, members = load_ensemble(ENSEMBLE_DIR, cfg)
    assert committed["seeds"] == seeds
    mod = _script()
    assert list(committed["settings"]) == list(mod.SETTINGS)
    stacked = stack_members(members)
    scen = build_scenario(panel, cfg, ssp=None)
    for name in ("as stated", "all three moderate"):
        fresh = mod.summarise(mod.max_effort(wm, stacked, scen, *mod.SETTINGS[name]), cfg.climate)
        old = committed["settings"][name]["scenarios"][scen.name]
        np.testing.assert_allclose(fresh["by_seed"]["cut_2050_pct"], old["by_seed"]["cut_2050_pct"],
                                   atol=0.05)
        np.testing.assert_allclose(fresh["global_cum_max_gt"], old["global_cum_max_gt"], rtol=1e-4)

    with open("results/tables/plan.json") as fh:
        plan = json.load(fh)
    for sname in (scen.name, "SSP2 (v3.2)"):
        stated, planned = committed["settings"]["as stated"]["scenarios"][sname], plan["scenarios"][sname]
        np.testing.assert_allclose(stated["cut_2050_pct"], planned["cut_2050_pct"], atol=0.05)
        np.testing.assert_allclose(stated["by_seed"]["max_2050_mt"],
                                   planned["by_seed"]["max_2050_mt"], rtol=1e-4)
