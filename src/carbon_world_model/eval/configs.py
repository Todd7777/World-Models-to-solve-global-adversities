"""Named model/training configurations for ablation and model selection, and the checks
that results were produced by the configuration they are used as."""
from __future__ import annotations

import dataclasses
import json
import os

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


def _checkpoint_record(path: str) -> dict:
    """The config.json that scripts/02_train.py writes alongside the checkpoint."""
    if not os.path.exists(path):
        raise ProvenanceError(
            f"{path} not found, so the configuration the checkpoint was trained with is "
            "unknown. scripts/02_train.py writes it with the checkpoint; retrain to restore it.")
    with open(path) as fh:
        return json.load(fh)


def _not_trained_with(what: str, against: str, label: str, path: str, diff) -> str:
    return (f"the checkpoint was not trained with {what}: {path}, written by "
            f"scripts/02_train.py with the checkpoint, differs from {against} in\n"
            + "\n".join(f"  {k}: checkpoint {x!r}, {label} {y!r}" for k, x, y in diff))


def check_checkpoint_config(name: str, path: str) -> Config:
    """Return make_config(name) after checking that the checkpoint was trained with it.

    `path` is the config.json that scripts/02_train.py writes alongside the checkpoint. Raises
    ProvenanceError if it is missing or differs from make_config(name) in a section that
    determines the fit (FIT_SECTIONS).
    """
    cfg = make_config(name)
    recorded = _checkpoint_record(path)
    diff = config_diff(recorded, config_dict(cfg), FIT_SECTIONS)
    if not diff:
        return cfg
    same = [n for n in REGISTRY
            if not config_diff(recorded, config_dict(make_config(n)), FIT_SECTIONS)]
    raise ProvenanceError(
        _not_trained_with(f"configuration {name!r}", f"make_config({name!r})", name, path, diff)
        + "\nThe checkpoint matches "
        + (", ".join(repr(n) for n in same) if same else "no named configuration")
        + f". To evaluate {name!r}, make it the default in config.py and retrain with "
        "scripts/02_train.py.")


def check_default_checkpoint(path: str) -> Config:
    """Return default_config() after checking that the checkpoint was trained with it.

    For the scripts that run the checkpoint under the default configuration (03_backtest.py,
    04_plan.py, 05_make_figures.py): once the default in config.py changes, the checkpoint
    was trained with something else until scripts/02_train.py is rerun. Raises
    ProvenanceError as check_checkpoint_config() does.
    """
    cfg = default_config()
    diff = config_diff(_checkpoint_record(path), config_dict(cfg), FIT_SECTIONS)
    if diff:
        raise ProvenanceError(
            _not_trained_with("the default configuration", "default_config()", "default",
                              path, diff)
            + "\nRetrain with scripts/02_train.py, which trains default_config() and rewrites "
            "config.json, or restore the default the checkpoint was trained with.")
    return cfg
