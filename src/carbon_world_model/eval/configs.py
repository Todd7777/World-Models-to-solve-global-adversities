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

# Selected on the VALIDATION folds only (results/sweep_val/report.json, lowest score =
# country MAPE% + mean |aggregate error|%, 2-seed ensemble): 11.05, against 11.37 for
# 'ms' and 21.20 for 'orig'. Single seeds score 11.5 and 13.3, so the margin over 'ms'
# is within seed noise. The trend-Kaya baseline scores 9.85 on the same folds.
SELECTED = "ms_percap_damped"


def make_config(name: str, base: Config | None = None, steps: int | None = None) -> Config:
    base = default_config() if base is None else base
    m, t = REGISTRY[name]
    cfg = dataclasses.replace(base, model=dataclasses.replace(base.model, **m),
                              train=dataclasses.replace(base.train, **t))
    if steps is not None:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, steps=steps))
    return cfg
