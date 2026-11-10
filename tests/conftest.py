"""Shared pytest fixtures."""
import os

import pytest

from carbon_world_model.config import default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.eval.configs import make_config
from carbon_world_model.model.world_model import WorldModel
from carbon_world_model.train.fit import (
    DEFAULT_CHECKPOINT, checkpoint_config_name, fit, load_params,
)

# Keep JAX on CPU and quiet during tests.
os.environ.setdefault("JAX_PLATFORMS", "cpu")


@pytest.fixture(scope="session")
def cfg():
    """The configuration the committed checkpoint was trained with (see train/fit.py)."""
    return make_config(checkpoint_config_name())


@pytest.fixture(scope="session")
def panel():
    base = default_config()
    if not os.path.exists(base.data.processed_npz):
        pytest.skip("processed panel not built; run scripts/01_build_dataset.py")
    return Panel.load(base)


@pytest.fixture(scope="session")
def wm(panel, cfg):
    return WorldModel.build(panel, cfg)


@pytest.fixture(scope="session")
def params(panel, cfg, wm):
    """Trained parameters: load the committed checkpoint, else a quick fit."""
    ckpt = DEFAULT_CHECKPOINT
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
