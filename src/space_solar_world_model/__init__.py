"""Space Solar World Model (SSWM).

A structural causal world model of space-based solar power: where in space to put solar
arrays, how to build them and what to do with the power, evaluated against the
terrestrial counterfactual of keeping the panels on the ground.

Layers
------
physics/      Sun, orbits (J2), Earth shadow, radiation belts, drag, debris, beaming, coverage
economics/    in-space transport, structural cost/energy model (JAX), ground PV + batteries
placements    catalogue of candidate placements with precomputed physics
world         actions and the world object agents plan in
planner       design optimisation by gradient descent through the model
scm           interventions, abduction-action-prediction counterfactuals, decision analysis

Typical use::

    from space_solar_world_model import SpaceSolarWorld, Action, CausalWorld, planner
    world = SpaceSolarWorld("starship")
    geo = planner.plan_beam(world, "geo", "Las Vegas", "mw2.45")
    print(geo.lcoe)                                   # $/MWh delivered to the grid
    cw = CausalWorld(world)
    cw.intervene(geo.action, do={"fluence_rate": 0.0})   # a world without radiation damage
"""
from .scenario import ADVANCED, SCENARIOS, STARSHIP, TODAY, Scenario
from .world import Action, SpaceSolarWorld
from .scm import CausalWorld
from . import planner

__all__ = ["SpaceSolarWorld", "Action", "CausalWorld", "planner", "Scenario", "SCENARIOS",
           "TODAY", "STARSHIP", "ADVANCED"]
