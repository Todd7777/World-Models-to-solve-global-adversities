"""The diagnostic of plan/response.py, and the committed table.

The open loop (the old planner's controls added to business as usual's own changes)
must be business as usual without controls, keep every unit of control, and agree with
the closed loop until the network first sees a controlled state. The frontier is the
planner's maximum effort, checked here against a separate, step-by-step implementation.
"""
import json

import numpy as np
import pytest

from carbon_world_model.config import default_config
from carbon_world_model.plan.ensemble import (
    ENSEMBLE_DIR, load_ensemble, max_controls, rollout_members, stack_members,
)
from carbon_world_model.plan.optimize import FOSSIL_IDX, LOWC_IDX
from carbon_world_model.plan.response import (
    LOOPS, PLANS, closed_loop_max, closed_loop_members, control_response, effective_ef,
    lowc_share, open_loop,
)
from carbon_world_model.plan.scenario import build_scenario


@pytest.fixture(scope="module")
def scen(panel, cfg):
    return build_scenario(panel, cfg, end_year=2030)


@pytest.fixture(scope="module")
def stacked(wm):
    return stack_members([wm.init_params(seed=0), wm.init_params(seed=1)])


@pytest.fixture(scope="module")
def bau(wm, scen, stacked):
    return closed_loop_members(wm, stacked, scen)


def test_open_loop_without_controls_is_business_as_usual(wm, scen, stacked, bau):
    z = np.zeros((scen.C, scen.Tf))
    ol = open_loop(wm, stacked, scen, bau, z, z)
    for k in ("co2", "D", "shares"):
        np.testing.assert_allclose(ol[k], bau[k], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(rollout_members(wm, stacked, scen)[k], bau[k], rtol=1e-5)


def test_open_loop_keeps_every_control(wm, scen, stacked, bau):
    g, h = closed_loop_max(scen)
    z = np.zeros_like(g)
    eff = open_loop(wm, stacked, scen, bau, z, h)
    cum = np.concatenate([np.zeros((scen.C, 1)), np.cumsum(h, axis=1)[:, :-1]], axis=1)
    np.testing.assert_allclose(np.log(bau["D"] / eff["D"]),
                               np.broadcast_to(cum, eff["D"].shape), atol=1e-5)
    clean = open_loop(wm, stacked, scen, bau, g, z)
    assert np.all(lowc_share(clean)[:, 1:] > lowc_share(bau)[:, 1:])

    # The controls of the first year move the state of the second, which the network has
    # not yet responded to: both loops agree on it, and on the first year's emissions.
    closed = closed_loop_members(wm, stacked, scen, g, h)
    opened = open_loop(wm, stacked, scen, bau, g, h)
    np.testing.assert_allclose(closed["D"][:, :, :2], opened["D"][:, :, :2], rtol=1e-4)
    np.testing.assert_allclose(closed["co2"][:, :, 0], opened["co2"][:, :, 0], rtol=1e-4)


def test_planner_max_effort_matches_a_separate_frontier(wm, scen, stacked, bau):
    """Maximum effort recomputed year by year from business as usual: intensity falls at
    max(BAU rate, h_max), and the planned low-carbon share grows by the room below g_max,
    displacing BAU's fossil mix of the year, until no fossil fuel is left."""
    K, C, Tf, _ = bau["shares"].shape
    li = np.log(bau["D"].astype(float)) - np.log(scen.gdp)[None]
    gap_h = np.maximum(0.0, scen.h_max - (li[..., :-1] - li[..., 1:]))
    s = bau["shares"].astype(float)
    gap_g = np.maximum(0.0, scen.g_max - np.diff(s[..., LOWC_IDX].sum(-1), axis=-1))
    D = bau["D"] * np.exp(-np.concatenate([np.zeros((K, C, 1)), np.cumsum(gap_h, -1)], -1))
    fossil = np.zeros_like(s)
    fossil[..., FOSSIL_IDX] = s[..., FOSSIL_IDX]
    F = fossil.sum(-1)
    X = np.zeros((K, C, Tf))
    for t in range(Tf - 1):
        X[..., t + 1] = np.minimum(X[..., t] + gap_g[..., t], F[..., t + 1])
    shares = s - fossil * (X / np.maximum(F, 1e-9))[..., None] + X[..., None] * scen.clean_weights
    co2 = (effective_ef(wm, stacked)[:, :, None, :] * D[..., None] * shares).sum(-1)
    mx = rollout_members(wm, stacked, scen, *max_controls(scen))
    np.testing.assert_allclose(mx["D"], D, rtol=1e-4)
    np.testing.assert_allclose(mx["shares"], shares, atol=1e-5)
    np.testing.assert_allclose(mx["co2"], co2 + scen.process[None], rtol=1e-4)
    assert (mx["co2"] < bau["co2"] + 1e-3).all() and (mx["co2"][:, :, -1] < bau["co2"][:, :, -1]).all()


def test_committed_control_response_matches_the_ensemble(panel, wm):
    """results/tables/control_response.json comes from the committed ensemble: rerun
    scripts/control_response.py whenever scripts/02b_train_ensemble.py is rerun."""
    with open("results/tables/control_response.json") as fh:
        committed = json.load(fh)
    cfg = default_config()
    seeds, members = load_ensemble(ENSEMBLE_DIR, cfg)
    assert committed["seeds"] == seeds
    scen = build_scenario(panel, cfg, ssp=None)
    fresh = control_response(wm, stack_members(members), scen)
    old = committed["scenarios"][scen.name]
    np.testing.assert_allclose(fresh["bau_2050_mt"], old["bau_2050_mt"], rtol=1e-4)
    for loop in LOOPS:
        for plan in PLANS:
            np.testing.assert_allclose(fresh["cut_2050_pct"][loop][plan],
                                       old["cut_2050_pct"][loop][plan], atol=0.05)
    for channel in ("efficiency", "clean"):
        np.testing.assert_allclose(fresh["retention"][channel], old["retention"][channel],
                                   atol=1e-3)
