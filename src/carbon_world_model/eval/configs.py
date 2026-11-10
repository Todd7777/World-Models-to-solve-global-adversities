"""Named model/training configurations for ablation and model selection."""
from __future__ import annotations

import dataclasses

from ..config import Config, default_config

# name -> (model overrides, train overrides). 'orig' reproduces the original model.
REGISTRY: dict[str, tuple[dict, dict]] = {
    "orig": ({}, {}),
    "notime": ({"time_features": False}, {}),
    "percap": ({"level_mode": "per_capita"}, {}),
    "ema": ({"momentum": "ema"}, {}),
    "damped": ({"momentum": "ema", "damped_trend": True}, {}),
    "ms": ({}, {"multi_shoot": True}),
    "ms_notime": ({"time_features": False}, {"multi_shoot": True}),
    "ms_notime_percap": ({"time_features": False, "level_mode": "per_capita"}, {"multi_shoot": True}),
    "ms_notime_percap_ema": ({"time_features": False, "level_mode": "per_capita",
                              "momentum": "ema"}, {"multi_shoot": True}),
    "ms_full": ({"time_features": False, "level_mode": "per_capita", "momentum": "ema",
                 "damped_trend": True}, {"multi_shoot": True}),
    "ms_full_shootonly": ({"time_features": False, "level_mode": "per_capita",
                           "momentum": "ema", "damped_trend": True},
                          {"multi_shoot": True, "w_full": 0.0}),
    "notime_percap_damped": ({"time_features": False, "level_mode": "per_capita",
                              "momentum": "ema", "damped_trend": True}, {}),
    # Second round (added after round-1 validation showed time features help): multiple
    # shooting combined with the other switches, time features kept ON.
    "ms_ema": ({"momentum": "ema"}, {"multi_shoot": True}),
    "ms_damped": ({"momentum": "ema", "damped_trend": True}, {"multi_shoot": True}),
    "ms_percap": ({"level_mode": "per_capita"}, {"multi_shoot": True}),
    "ms_percap_damped": ({"level_mode": "per_capita", "momentum": "ema", "damped_trend": True},
                         {"multi_shoot": True}),
}

# Third round: the stability term (TrainConfig.w_stab, train/fit.stability_loss) on the
# three configurations tied on validation. Decision rule, fixed before these were run:
# a stabilised variant is preferred over the current selection if its validation score
# is within the tie tolerance (scripts/07_structural.TIE_TOL = 0.5) of the best
# validation score; among such variants, the lowest score wins.
STAB_WEIGHTS = (10.0, 100.0)
for _base in ("ms_percap_damped", "ms", "ms_full_shootonly"):
    for _w in STAB_WEIGHTS:
        _m, _t = REGISTRY[_base]
        REGISTRY[f"{_base}_stab{int(_w)}"] = (dict(_m), {**_t, "w_stab": _w})

# Selected on the VALIDATION folds only (results/sweep_val/report.json, lowest score =
# country MAPE% + mean |aggregate error|%, 2-seed ensemble), by the round-3 rule above:
# ms_full_shootonly_stab100 scores 9.85, against 9.96 for ms_percap_damped_stab10, 11.05
# for the previous selection (ms_percap_damped) and 21.20 for 'orig'. The trend-Kaya
# baseline also scores 9.85 on these folds. 22 configurations were compared on 2 seeds,
# so the best validation score is optimistic; the test folds are the check.
SELECTED = "ms_full_shootonly_stab100"


def make_config(name: str, base: Config | None = None, steps: int | None = None) -> Config:
    base = default_config() if base is None else base
    m, t = REGISTRY[name]
    cfg = dataclasses.replace(base, model=dataclasses.replace(base.model, **m),
                              train=dataclasses.replace(base.train, **t))
    if steps is not None:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, steps=steps))
    return cfg
