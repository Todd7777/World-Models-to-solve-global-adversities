"""The committed checkpoint must match the selected configuration and model structure.

A checkpoint trained before a dynamics change silently degrades every downstream
result (this happened once: see CLAUDE.md). These tests fail loudly instead.
"""
import os

import jax
import numpy as np
import pytest

from carbon_world_model.eval.configs import SELECTED, make_config
from carbon_world_model.model.world_model import WorldModel
from carbon_world_model.train.fit import (
    DEFAULT_CHECKPOINT, checkpoint_config_name, load_params, load_trained, save_checkpoint,
)

needs_ckpt = pytest.mark.skipif(not os.path.exists(DEFAULT_CHECKPOINT),
                                reason="no committed checkpoint")


def _shapes(tree):
    return jax.tree_util.tree_map(lambda x: tuple(np.shape(x)), tree)


@needs_ckpt
def test_checkpoint_is_the_selected_config():
    assert checkpoint_config_name() == SELECTED, (
        f"checkpoint was trained as {checkpoint_config_name()!r} but SELECTED={SELECTED!r}; "
        f"run scripts/02_train.py")


@needs_ckpt
def test_checkpoint_matches_model_structure(panel):
    cfg, wm, params = load_trained(panel)
    ref = wm.init_params(seed=0)
    assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(ref)
    assert _shapes(params) == _shapes(ref)


def test_save_load_roundtrip_keeps_config(tmp_path, panel):
    cfg = make_config("ms_percap_damped")
    params = WorldModel.build(panel, cfg).init_params(seed=3)
    path = str(tmp_path / "p.pkl")
    save_checkpoint(params, path, "ms_percap_damped", {"seed": 3})
    assert checkpoint_config_name(path) == "ms_percap_damped"
    back = load_params(path)
    for a, b in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(back)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_legacy_checkpoint_defaults_to_orig(tmp_path):
    assert checkpoint_config_name(str(tmp_path / "missing.pkl")) == "orig"
