"""The counterfactuals of plan/response.py, and the committed diagnostic.

With the natural changes held at their business-as-usual values, no control must mean
business as usual, every unit of the efficiency control must persist, the closed loop
must agree with the open loop until the network first sees a controlled state, and the
frontier controls must lift the total rates of change exactly to the bounds.
"""
import json

import numpy as np
import pytest

from carbon_world_model.config import default_config
from carbon_world_model.plan.ensemble import (
    ENSEMBLE_DIR, load_ensemble, max_controls, rollout_members, stack_members,
)
from carbon_world_model.plan.response import (
    LOOPS, PLANS, control_response, frontier_controls, log_intensity, lowc_share, lowc_shares,
    open_loop,
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
    return rollout_members(wm, stacked, scen)


def test_open_loop_without_controls_is_business_as_usual(wm, scen, stacked, bau):
    z = np.zeros((scen.C, scen.Tf))
    ol = open_loop(wm, stacked, scen, bau, z, z)
    for k in ("co2", "D", "shares"):
        np.testing.assert_allclose(ol[k], bau[k], rtol=1e-4, atol=1e-6)


def test_open_loop_keeps_every_control(wm, scen, stacked, bau):
    g, h = max_controls(scen)
    z = np.zeros_like(g)
    eff = open_loop(wm, stacked, scen, bau, z, h)
    cum = np.concatenate([np.zeros((scen.C, 1)), np.cumsum(h, axis=1)[:, :-1]], axis=1)
    np.testing.assert_allclose(np.log(bau["D"] / eff["D"]),
                               np.broadcast_to(cum, eff["D"].shape), atol=1e-5)
    clean = open_loop(wm, stacked, scen, bau, g, z)
    assert np.all(lowc_share(clean)[:, 1:] > lowc_share(bau)[:, 1:])

    # The controls of the first year move the state of the second, which the network has
    # not yet responded to: both loops agree on it, and on the first year's emissions.
    closed, opened = rollout_members(wm, stacked, scen, g, h), open_loop(wm, stacked, scen, bau, g, h)
    np.testing.assert_allclose(closed["D"][:, :, :2], opened["D"][:, :, :2], rtol=1e-4)
    np.testing.assert_allclose(closed["co2"][:, :, 0], opened["co2"][:, :, 0], rtol=1e-4)


def test_frontier_controls_lift_the_total_rates_to_the_bounds(wm, scen, stacked, bau):
    g, h = frontier_controls(scen, bau)
    assert g.shape == h.shape == bau["D"].shape and g.min() >= 0 and h.min() >= 0
    run = open_loop(wm, stacked, scen, bau, g, h)
    d_int, d_bau = np.diff(log_intensity(run, scen), axis=2), np.diff(log_intensity(bau, scen), axis=2)
    np.testing.assert_allclose(d_int, np.minimum(d_bau, -scen.h_max), atol=1e-6)
    # Where no share was clipped at zero, the low-carbon share gains exactly the larger
    # of its BAU gain and the bound; clipping only stops a country running out of fossil.
    gain, gain_bau = np.diff(lowc_shares(run), axis=2), np.diff(lowc_shares(bau), axis=2)
    unclipped = (run["shares"][:, :, 1:] > 0).all(axis=-1)
    assert unclipped.mean() > 0.9
    np.testing.assert_allclose(gain[unclipped], np.maximum(gain_bau, scen.g_max)[unclipped],
                               atol=1e-6)


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
