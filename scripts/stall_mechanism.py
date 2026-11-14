"""Why models trained up to 2005-2009 stall mature economies' efficiency gains.

The selected configuration's annual change in log energy intensity is
    softclip(phi * momentum + network correction),
and the momentum is an exponential moving average of the model's own past changes,
initialised from the observed history. For 12 mature economies, models trained on data up
to 2005-2009 forecast intensity falling far more slowly than it did (README, Limitations).
This script refits the selected configuration at the validation cutoffs, as the sweep fits
it (each refit must reproduce the sweep's cached forecast bit for bit), and re-implements
its forecast step by step from the production feature and network code (checked against the
production forecast). For each cutoff and seed it records:

1. Per forecast step, the mature economies' mean momentum, damped-momentum part, network
   correction and resulting change.
2. Which network inputs drive each step's change in that correction: every input group is
   set back to its value at the previous step in turn, with the fuel-share momenta split by
   fuel. Share momenta enter the network divided by each fuel's typical training share
   change, floored at 1e-3 (`compute_norm`), so near-empty fuels are scaled up most.
3. A counterfactual forecast in which the network sees the share momenta frozen at their
   observed values in the cutoff year, while the dynamics evolve as usual.

Validation folds only: cutoffs 2005, 2007, 2009 with 6-year horizons, so nothing after 2015
is read. Needs the validation sweep's outputs for the selected configuration to check the
refits. About 5 minutes with 4 workers. Writes results/tables/stall_mechanism.json.

Usage:  python scripts/stall_mechanism.py [--seeds 0,1] [--root results/sweep_val] [--jobs 4]
"""
import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

# One XLA thread per worker, as in scripts/run_job.py, so the refits can reproduce the
# sweep's outputs bit for bit.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                                   "intra_op_parallelism_threads=1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from carbon_world_model.config import SELECTED_CONFIG, default_config  # noqa: E402
from carbon_world_model.data.panel import FUELS, Panel  # noqa: E402
from carbon_world_model.eval.rolling import (  # noqa: E402
    VAL_CUTOFFS,
    VAL_HORIZON,
    horizon_for,
    job_path,
)
from carbon_world_model.model.dynamics import _features, softclip  # noqa: E402
from carbon_world_model.model.nn import mlp_apply  # noqa: E402
from carbon_world_model.model.world_model import _year_index, observed_inits  # noqa: E402

OUT = "results/tables/stall_mechanism.json"
MATURE = ("United Kingdom", "Hungary", "Belgium", "Denmark", "Finland", "Greece",
          "United States", "Italy", "Sweden", "France", "Germany", "Japan")  # as transfer_diagnostics.py


def _groups(n_fuels):
    """Input columns of the selected configuration's network (per-capita levels, momentum,
    no calendar time): shares, five level/growth inputs, intensity momentum, share momenta,
    then static features and the embedding, which do not change during a forecast."""
    F = n_fuels
    g = {"shares": list(range(F)), "energy per capita": [F], "GDP per capita": [F + 1],
         "GDP growth": [F + 2], "intensity": [F + 3], "population": [F + 4],
         "intensity momentum": [F + 5]}
    for j, f in enumerate(FUELS):
        g[f"share momentum: {f}"] = [F + 6 + j]
    return g


def _job(job):
    from carbon_world_model.train.fit import fit
    cutoff, seed = job
    cfg = default_config()
    panel = Panel.load(cfg)
    H = horizon_for(panel, cutoff, VAL_HORIZON)
    r = fit(panel, cfg, train_year_max=cutoff, seed=seed, verbose=False)
    fc = np.asarray(r.wm.forecast(r.params, panel, cutoff, end_year=cutoff + H)["co2"])
    return cutoff, seed, fc, jax.tree_util.tree_map(np.asarray, r.params)


def _trace(wm, params, panel, cutoff, mature, others, freeze_share_momenta=False):
    """Step-by-step forecast without controls (model.dynamics.rollout), recording the parts
    of the intensity change and each step's network input."""
    hp, norm = wm.hp, wm.norm
    k = _year_index(panel, cutoff)
    H = horizon_for(panel, cutoff, VAL_HORIZON)
    gdp = jnp.asarray(panel.gdp[:, k:k + H + 1])
    pop = jnp.asarray(panel.population[:, k:k + H + 1])
    logGDP, logPop = jnp.log(gdp + 1e-6), jnp.log(pop + 1e-6)
    growth = jnp.concatenate([logGDP[:, 1:], logGDP[:, -1:]], axis=1) - logGDP
    ini = observed_inits(panel, k, hp["momentum"], hp["ema_decay"])
    I, s = jnp.asarray(ini["I0"]), jnp.asarray(ini["s0"])
    mI, ms = jnp.asarray(ini["mom_I0"]), jnp.asarray(ini["mom_s0"])
    I0, ms0 = I, ms
    phi_I, phi_s = jax.nn.sigmoid(params["phi"][0]), jax.nn.sigmoid(params["phi"][1])
    rho, max_int, max_share = hp["ema_decay"], hp["max_int"], hp["max_share"]

    def net(x):
        h = jax.nn.silu(mlp_apply(params["trunk"], x))
        return max_int * jnp.tanh((h @ params["int_head"]["w"] + params["int_head"]["b"])[:, 0]), h

    steps, xs = [], []
    for t in range(H):
        D = gdp[:, t] * jnp.exp(I)
        x = _features(params["embed"], jnp.asarray(wm.static), norm, hp, s, I, jnp.log(D + 1e-6),
                      logGDP[:, t], growth[:, t], logPop[:, t], jnp.zeros(I.shape), mI,
                      ms0 if freeze_share_momenta else ms)
        corr, h = net(x)
        dI = softclip(phi_I * mI + corr, max_int)
        ds = softclip(phi_s * ms + max_share * jnp.tanh(h @ params["mix_head"]["w"] + params["mix_head"]["b"]),
                      max_share)
        steps.append({"year": int(panel.years[k + t]),
                      "mature_momentum": float(100 * mI[mature].mean()),
                      "mature_damped_momentum": float(100 * (phi_I * mI)[mature].mean()),
                      "mature_network_correction": float(100 * corr[mature].mean()),
                      "mature_change": float(100 * dI[mature].mean())})
        xs.append(np.asarray(x))
        s_nat = jax.nn.relu(s + ds)
        s_nat = s_nat / (jnp.sum(s_nat, axis=1, keepdims=True) + 1e-9)
        mI, ms = rho * mI + (1 - rho) * dI, rho * ms + (1 - rho) * (s_nat - s)
        I, s = I + dI, s_nat
    trend = {g: float(100 * np.mean([(float(I[i]) - float(I0[i])) / H for i in rows]))
             for g, rows in (("mature", mature), ("others", others))}
    return steps, xs, np.asarray(I), trend, net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--root", default="results/sweep_val",
                    help="validation sweep whose cached outputs the refits are checked against")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    cfg = default_config()
    panel = Panel.load(cfg)
    mature = np.array([panel.countries.index(n) for n in MATURE])
    others = np.array([i for i in range(panel.n_countries) if i not in mature])
    groups = _groups(len(FUELS))
    lI = np.log(panel.primary_energy) - np.log(panel.gdp)

    with ProcessPoolExecutor(max_workers=a.jobs, mp_context=mp.get_context("spawn")) as pool:
        fits = list(pool.map(_job, [(co, s) for co in VAL_CUTOFFS for s in seeds]))

    from carbon_world_model.model.world_model import WorldModel
    out = {"config": SELECTED_CONFIG, "seeds": seeds, "mature": list(MATURE), "root": a.root,
           "by_cutoff": {}}
    for co in VAL_CUTOFFS:
        k = _year_index(panel, co)
        H = horizon_for(panel, co, VAL_HORIZON)
        wm = WorldModel.build(panel, cfg, upto_year=co)
        hp = wm.hp
        assert hp["level_mode"] == "per_capita" and hp["momentum"] != "none" and not hp["time_features"], \
            "the input layout in _groups assumes the selected configuration's options"
        assert wm.din == len(FUELS) * 2 + 6 + wm.static.shape[1] + wm.cfg.model.embed_dim
        row = {"share_momentum_scale": dict(zip(FUELS, map(float, wm.norm.ds_scale))),
               "realised_pct_per_year": {
                   g: float(100 * np.mean([(lI[i, k + H] - lI[i, k]) / H for i in rows]))
                   for g, rows in (("mature", mature), ("others", others))},
               "by_seed": {}}
        for cutoff, seed, fc, params in fits:
            if cutoff != co:
                continue
            p = jax.tree_util.tree_map(jnp.asarray, params)
            path = job_path(a.root, SELECTED_CONFIG, co, seed)
            steps, xs, I_end, trend, net = _trace(wm, p, panel, co, mature, others)
            prod = np.asarray(wm.forecast(p, panel, co, end_year=co + H)["intensity"][:, -1])
            _, _, _, frozen, _ = _trace(wm, p, panel, co, mature, others, freeze_share_momenta=True)
            drivers = []
            for t in range(1, H):
                base = float(net(jnp.asarray(xs[t]))[0][mature].mean())
                parts = {}
                for g, cols in groups.items():
                    x2 = xs[t].copy()
                    x2[:, cols] = xs[t - 1][:, cols]
                    parts[g] = float(100 * (base - float(net(jnp.asarray(x2))[0][mature].mean())))
                drivers.append({"step": f"{co + t}-{co + t + 1}", "change_pct_per_year": parts})
            row["by_seed"][seed] = {
                "reproduces_sweep_output": (bool(np.array_equal(fc, np.load(path)["co2_realized"]))
                                            if os.path.exists(path) else None),
                "trace_matches_production_forecast": bool(np.allclose(prod, I_end, atol=1e-5)),
                "phi_intensity": float(jax.nn.sigmoid(p["phi"][0])),
                "steps": steps,
                "network_correction_drivers": drivers,
                "forecast_trend_pct_per_year": trend,
                "share_momenta_frozen_trend_pct_per_year": frozen,
            }
        out["by_cutoff"][co] = row

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    for co, row in out["by_cutoff"].items():
        r = row["realised_pct_per_year"]
        print(f"cutoff {co} (realised: mature {r['mature']:+.2f}, others {r['others']:+.2f} %/yr)")
        for seed, v in row["by_seed"].items():
            t, f = v["forecast_trend_pct_per_year"], v["share_momenta_frozen_trend_pct_per_year"]
            big = max(((d["step"], g, x) for d in v["network_correction_drivers"]
                       for g, x in d["change_pct_per_year"].items()), key=lambda z: abs(z[2]))
            print(f"  seed {seed}: forecast mature {t['mature']:+.2f}, others {t['others']:+.2f}; share momenta "
                  f"frozen: mature {f['mature']:+.2f}, others {f['others']:+.2f} %/yr; largest driver of a "
                  f"step's swing: {big[1]} {big[2]:+.2f} ({big[0]}); reproduces sweep "
                  f"{v['reproduces_sweep_output']}, trace matches {v['trace_matches_production_forecast']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
