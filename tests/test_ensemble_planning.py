"""Planning with a seed ensemble (plan/ensemble.py).

The vmapped rollouts must agree with the single-model planner member by member, an
ensemble of one member must plan exactly like the single-model planners, and an ensemble
on disk must be checked before it is used. Untrained members and a short horizon keep
these fast; the committed ensemble is checked separately.
"""
import os

import numpy as np
import pytest

from carbon_world_model.config import default_config
from carbon_world_model.eval.configs import ProvenanceError, make_config
from carbon_world_model.plan import optimize as O
from carbon_world_model.plan.ensemble import (
    ensemble_pathway, evaluate_members, load_ensemble, max_controls, member_path,
    rollout_members, save_ensemble, stack_members,
)
from carbon_world_model.plan.scenario import build_scenario, ssp_ensemble
from carbon_world_model.train.fit import save_params


@pytest.fixture(scope="module")
def members(wm):
    return [wm.init_params(seed=0), wm.init_params(seed=1)]


@pytest.fixture(scope="module")
def scen(panel, cfg):
    return build_scenario(panel, cfg, end_year=2030)


def test_member_rollouts_match_the_single_model_planner(wm, scen, members):
    stacked = stack_members(members)
    g, h = max_controls(scen)
    bau = rollout_members(wm, stacked, scen)
    mx = rollout_members(wm, stacked, scen, g, h)
    assert bau["co2"].shape == (2, scen.C, scen.Tf)
    assert mx["shares"].shape[:3] == (2, scen.C, scen.Tf)
    for k, p in enumerate(members):
        np.testing.assert_allclose(bau["co2"][k], O.business_as_usual(wm, p, scen).co2, rtol=1e-4)
        np.testing.assert_allclose(mx["co2"][k], O.max_ambition(wm, p, scen).co2, rtol=1e-4)
    ev = evaluate_members(wm, stacked, [scen], g, h)
    assert ev["cum_gt"].shape == (2, 1) and ev["global_mt"].shape == (2, 1, scen.Tf)
    np.testing.assert_allclose(ev["cum_gt"][:, 0],
                               [O.max_ambition(wm, p, scen).cum_gt for p in members], rtol=1e-4)


def test_one_member_ensemble_plans_like_the_single_model_planners(panel, cfg, wm, scen, members):
    p = members[0]
    one = stack_members([p])
    budget = 0.9 * O.business_as_usual(wm, p, scen).cum_gt
    ens = ensemble_pathway(wm, one, [scen], budget_gt=budget, mode="mean", steps=20)
    single = O.optimize_pathway(wm, p, scen, budget_gt=budget, penalty=2.0, steps=20)
    np.testing.assert_allclose(ens["g"], single.g, rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(ens["cost"], single.cost, rtol=1e-3)

    scens = ssp_ensemble(panel, cfg, end_year=2030)
    ens = ensemble_pathway(wm, one, scens, budget_gt=budget, mode="all", steps=10)
    rob = O.robust_pathway(wm, p, scens, budget_gt=budget, penalty=2.0, mode="all", steps=10)
    np.testing.assert_allclose(ens["h"], rob["h"], rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(ens["cum_gt"][0], rob["cum_gt"], rtol=1e-4)
    with pytest.raises(ValueError):
        ensemble_pathway(wm, one, [scen], budget_gt=budget, mode="worst", steps=1)


def test_ensemble_on_disk_is_checked(tmp_path, members):
    root = str(tmp_path / "ensemble")
    save_ensemble(root, default_config(), {0: members[0], 1: members[1]}, {1: {"note": "x"}})
    seeds, loaded = load_ensemble(root)
    assert seeds == [0, 1] and len(loaded) == 2

    ckpt = str(tmp_path / "params_full.pkl")
    save_params(members[0], ckpt)
    load_ensemble(root, checkpoint=ckpt)  # its seed-0 member is the checkpoint
    save_params(members[1], ckpt)         # the checkpoint was retrained afterwards
    with pytest.raises(ProvenanceError, match="seed-0 member"):
        load_ensemble(root, checkpoint=ckpt)

    other = str(tmp_path / "other")
    save_ensemble(other, make_config("orig"), {0: members[0]})
    with pytest.raises(ProvenanceError, match=r"train\.multi_shoot: ensemble False, default True"):
        load_ensemble(other, cfg=make_config("ms"))

    os.remove(member_path(root, 1))
    with pytest.raises(ProvenanceError, match="missing"):
        load_ensemble(root)
    with pytest.raises(ProvenanceError, match="not found"):
        load_ensemble(str(tmp_path / "nothing"))
