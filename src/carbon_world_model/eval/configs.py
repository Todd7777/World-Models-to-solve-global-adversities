"""Named model/training configurations for ablation and model selection, and the checks
that results were produced by the configuration they are used as."""
from __future__ import annotations

import dataclasses
import json

from ..config import Config, default_config

# Config sections that determine a fitted model; `climate` only feeds planning.
FIT_SECTIONS = ("data", "model", "train")


class ProvenanceError(ValueError):
    """Results cannot be used or combined as asked because of how they were trained."""

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


def make_config(name: str, base: Config | None = None, steps: int | None = None) -> Config:
    base = default_config() if base is None else base
    m, t = REGISTRY[name]
    cfg = dataclasses.replace(base, model=dataclasses.replace(base.model, **m),
                              train=dataclasses.replace(base.train, **t))
    if steps is not None:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, steps=steps))
    return cfg


def config_dict(cfg: Config) -> dict:
    """`cfg` as plain JSON data, the form config.json and job outputs record."""
    return json.loads(json.dumps(dataclasses.asdict(cfg)))


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()


def config_diff(a: dict, b: dict, sections=None) -> list[tuple[str, object, object]]:
    """(dotted field, value in a, value in b) for each field where two config dicts differ,
    optionally only within the given top-level sections."""
    out = []

    def walk(x, y, path):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in list(x) + [k for k in y if k not in x]:
                walk(x.get(k, _MISSING), y.get(k, _MISSING), f"{path}.{k}" if path else k)
        elif x != y:
            out.append((path, x, y))

    for s in sections if sections is not None else list(a) + [k for k in b if k not in a]:
        walk(a.get(s, _MISSING), b.get(s, _MISSING), s)
    return out
