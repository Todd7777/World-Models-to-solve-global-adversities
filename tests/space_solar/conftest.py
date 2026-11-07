import os

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")


@pytest.fixture(scope="session")
def world():
    from space_solar_world_model import SpaceSolarWorld
    return SpaceSolarWorld("starship")


@pytest.fixture(scope="session")
def geo_plan(world):
    from space_solar_world_model import planner
    return planner.plan_beam(world, "geo", "Las Vegas", "mw2.45")
