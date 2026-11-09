"""Seed sensitivity of the default configuration: retrain the full-data model with
several seeds and compare training stability, in-sample fit and planning behaviour.

Seed 0 reproduces the committed checkpoint (scripts/02_train.py). For each seed this
records the training-loss history, the in-sample report, business-as-usual and
maximum-effort pathways under the heuristic and SSP2 drivers, how many times the
projected BAU global total changes direction from one year to the next (the observed
1990-2022 total does so 10 times out of a possible 31; a two-year sawtooth does so
almost every year), and whether the assertions of the three checkpoint-dependent
tests hold. Writes results/tables/seed_sensitivity.json.

Usage:  python scripts/seed_sensitivity.py [--seeds 0,1,2,3]
"""
import argparse
import json
import os

import jax.numpy as jnp
import numpy as np

from carbon_world_model.config import default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.model.world_model import WorldModel
from carbon_world_model.train.fit import fit
from carbon_world_model.eval.backtest import insample_report
from carbon_world_model.plan.scenario import build_scenario
from carbon_world_model.plan import optimize as O

OUT = "results/tables/seed_sensitivity.json"


def direction_changes(x) -> int:
    d = np.diff(np.asarray(x, dtype=float))
    return int(np.sum(np.diff(np.sign(d)) != 0))


def effort_finals(wm, params, scen) -> list[float]:
    """2050 total CO2 for increasing uniform effort, as in tests/test_planning.py."""
    finals = []
    for raw in (-10.0, -1.0, 2.0, 10.0):
        rg = jnp.full((scen.C, scen.Tf), raw)
        ctrl, _, _ = O._controls(rg, rg, scen)
        finals.append(float(jnp.sum(O._rollout(wm, params, scen, ctrl)["co2"][:, -1])))
    return finals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3")
    a = ap.parse_args()
    cfg = default_config()
    panel = Panel.load(cfg)
    wm = WorldModel.build(panel, cfg)
    scens = {"heuristic": build_scenario(panel, cfg, end_year=2050, ssp=None),
             "SSP2": build_scenario(panel, cfg, end_year=2050)}  # the tests' scenario
    obs = (panel.co2 * panel.mask).sum(axis=0)
    out = {"config": "default (orig)", "train_steps": cfg.train.steps,
           "observed_direction_changes_1990_2022": direction_changes(obs), "seeds": {}}

    for seed in [int(s) for s in a.seeds.split(",")]:
        r = fit(panel, cfg, wm=wm, seed=seed, verbose=False)
        ins = insample_report(panel, cfg, r.params, wm)
        rec = {"loss_history": {h["step"]: h["loss"] for h in r.history}, "insample": ins}
        for name, scen in scens.items():
            bau = O.business_as_usual(wm, r.params, scen)
            mx = O.max_ambition(wm, r.params, scen)
            rec[name] = {"bau_2050_mt": float(bau.global_co2[-1]),
                         "max_2050_mt": float(mx.global_co2[-1]),
                         "bau_cum_gt": bau.cum_gt, "max_cum_gt": mx.cum_gt,
                         "bau_direction_changes": direction_changes(bau.global_co2),
                         "yearly_steps": len(bau.global_co2) - 1}
        finals = effort_finals(wm, r.params, scens["SSP2"])
        s2 = rec["SSP2"]
        rec["effort_finals_mt"] = finals
        rec["tests_hold"] = {
            "test_trained_insample_accuracy": bool(ins["co2_r2"] > 0.98),
            "test_max_ambition_below_bau": bool(s2["max_2050_mt"] < s2["bau_2050_mt"]
                                                and s2["max_cum_gt"] < s2["bau_cum_gt"]),
            "test_effort_monotonic": bool(all(finals[i] > finals[i + 1] for i in range(3))),
        }
        out["seeds"][seed] = rec
        h = rec["heuristic"]
        print(f"seed {seed}: final loss {r.history[-1]['loss']:.4f} "
              f"(peak {max(rec['loss_history'][k] for k in list(rec['loss_history'])[1:]):.4f}) | "
              f"in-sample CO2 R2 {ins['co2_r2']:.4f} MAPE {100 * ins['co2_mape']:.1f}% | heuristic "
              f"BAU {h['bau_2050_mt'] / 1e3:.1f} max {h['max_2050_mt'] / 1e3:.1f} Gt, cum "
              f"{h['bau_cum_gt']:.0f}/{h['max_cum_gt']:.0f} Gt, direction changes "
              f"{h['bau_direction_changes']}/{h['yearly_steps'] - 1} | tests "
              f"{sum(rec['tests_hold'].values())}/3")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print("[ok] wrote", OUT)


if __name__ == "__main__":
    main()
