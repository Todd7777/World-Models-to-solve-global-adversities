"""Shared pytest fixtures."""
import os

import numpy as np
import pytest

from carbon_world_model.config import default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.model.world_model import WorldModel
from carbon_world_model.train.fit import fit, load_params

# Keep JAX on CPU and quiet during tests.
os.environ.setdefault("JAX_PLATFORMS", "cpu")


@pytest.fixture(scope="session")
def cfg():
    return default_config()


@pytest.fixture(scope="session")
def panel(cfg):
    if not os.path.exists(cfg.data.processed_npz):
        pytest.skip("processed panel not built; run scripts/01_build_dataset.py")
    return Panel.load(cfg)


@pytest.fixture(scope="session")
def wm(panel, cfg):
    return WorldModel.build(panel, cfg)


@pytest.fixture(scope="session")
def params(panel, cfg, wm):
    """Trained parameters: load the committed checkpoint, else a quick fit."""
    ckpt = "results/model/params_full.pkl"
    if os.path.exists(ckpt):
        try:
            return load_params(ckpt)
        except Exception:
            pass
    import dataclasses
    fast = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, steps=300))
    return fit(panel, fast, wm=wm, verbose=False).params


@pytest.fixture(scope="session")
def init_params(wm):
    return wm.init_params(seed=0)
