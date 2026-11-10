"""Does the learned dynamics honour the planner's controls?

Controls enter the dynamics additively (dynamics.rollout): an efficiency push h lowers
log energy intensity by h, and a clean push g moves g of share from fossil to
low-carbon fuels. The learned "natural" change, however, depends on the current state.
Once a plan has moved a country into a state it never visited in training, the network
may respond with changes that partly undo the push, or amplify it.

Retention measures this against an exact reference, the open-loop counterfactual
(optimize.max_ambition_open_loop): the same controls applied on top of the
business-as-usual natural changes, so the network cannot react to them.

  retention = (plan - BAU gap with the fitted dynamics) / (gap in the open loop)

at the horizon, energy-weighted across countries, separately for the low-carbon share
and for log energy intensity. 1 means the push passes through, below 1 means the fitted
dynamics push back, above 1 means they amplify it. This is a property of the fitted
model off its training distribution, not a measurement of the world: no data constrain
how real energy systems would respond to such pushes.
"""
from __future__ import annotations

import numpy as np

from ..config import LOW_CARBON_FUELS, FUELS
from .optimize import PlanResult
from .scenario import Scenario

_CLEAN = [FUELS.index(f) for f in LOW_CARBON_FUELS]


def _clean_share(r: PlanResult) -> np.ndarray:
    return r.shares[:, -1, _CLEAN].sum(-1)


def _log_intensity(r: PlanResult, scen: Scenario) -> np.ndarray:
    return np.log(r.D[:, -1] / np.asarray(scen.gdp)[:, -1])


def control_retention(bau: PlanResult, plan: PlanResult, open_loop: PlanResult,
                      scen: Scenario) -> dict:
    """Energy-weighted retention of the clean-mix and efficiency pushes at the horizon."""
    w = bau.D[:, -1] / bau.D[:, -1].sum()  # weight countries by BAU energy use
    cb, cp, co = _clean_share(bau), _clean_share(plan), _clean_share(open_loop)
    ib, ip, io = (_log_intensity(r, scen) for r in (bau, plan, open_loop))
    gap_s, ref_s = np.sum(w * (cp - cb)), np.sum(w * (co - cb))
    gap_i, ref_i = np.sum(w * (ib - ip)), np.sum(w * (ib - io))
    return {
        "clean_share_gap_pts": float(100 * gap_s),
        "clean_share_open_loop_gap_pts": float(100 * ref_s),
        "clean_retention": float(gap_s / ref_s),
        "intensity_gap_log": float(gap_i),
        "intensity_open_loop_gap_log": float(ref_i),
        "intensity_retention": float(gap_i / ref_i),
    }
