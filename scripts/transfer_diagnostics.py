"""Where the spatial-transfer error comes from, measured on the validation period only.

The transfer test (eval/transfer.py) fits the shared dynamics on 48 countries and adapts
each of 12 held-out countries' latent from its own data. Its error mixes the cost of not
having trained on those countries with how well the model forecasts them at all. This
script separates the two on the validation cutoffs (2005, 2007, 2009; 6-year horizons, so
nothing after 2015 is scored):

1. Per cutoff and seed: the transfer protocol's error on the 12 held-out countries, and
   the error on the same countries of the validation sweep's own model (trained on all 60
   countries, the 12 included), with trend-Kaya for reference; per country too.
2. The energy-intensity trend of 12 mature economies in the validation models'
   forecasts, against their pre-cutoff (10-year) and realised trends; and in the
   planning ensemble's BAU (2022-2032, heuristic drivers) against 2012-2022.
3. For the cutoff-2009 validation models: each forecast year's natural intensity change
   split into its damped-momentum and network-correction parts. The models are refitted
   (the result records whether the refit reproduces the sweep's output), and the split
   is recomputed step by step from the production feature and network code and checked
   against the production forecast.

Needs the validation sweep's outputs for the selected configuration
(scripts/06_sweep.py --phase val --configs <selected> --seeds 0,1). Writes
results/tables/transfer_diagnostics.json.

Usage:  python scripts/transfer_diagnostics.py [--seeds 0,1] [--jobs 4]
"""
import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

# One XLA thread per worker, as in scripts/run_job.py (and so the refits in part 3 can
# reproduce the sweep's outputs bit for bit).
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                                   "intra_op_parallelism_threads=1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from carbon_world_model.config import SELECTED_CONFIG, default_config  # noqa: E402
from carbon_world_model.data.panel import Panel  # noqa: E402
from carbon_world_model.eval import metrics as M  # noqa: E402
from carbon_world_model.eval.baselines import trend_kaya  # noqa: E402
from carbon_world_model.eval.rolling import VAL_CUTOFFS, VAL_HORIZON, load_jobs  # noqa: E402
from carbon_world_model.model.dynamics import _features, softclip  # noqa: E402
from carbon_world_model.model.nn import mlp_apply  # noqa: E402
from carbon_world_model.model.world_model import WorldModel, observed_inits, _year_index  # noqa: E402

OUT = "results/tables/transfer_diagnostics.json"
VAL_ROOT = "results/sweep_val"
ENSEMBLE = "results/model/ensemble"
CHECKPOINT = "results/model/params_full.pkl"
TEST_EVERY = 5   # spatial_transfer's held-out split: every fifth country by emission rank
MATURE = ("United Kingdom", "Hungary", "Belgium", "Denmark", "Finland", "Greece",
          "United States", "Italy", "Sweden", "France", "Germany", "Japan")
SPLIT_CUTOFF = 2009


def _held(panel):
    return list(range(0, panel.n_countries, TEST_EVERY))


def _transfer(job):
    from carbon_world_model.eval.transfer import spatial_transfer
    cutoff, seed = job
    cfg = default_config()
    panel = Panel.load(cfg)
    tr = spatial_transfer(panel, cfg, test_every=TEST_EVERY, cutoff=cutoff, seed=seed,
                          end_year=cutoff + VAL_HORIZON, verbose=False)
    return cutoff, seed, tr


def _intensity_split(params, wm, panel, cutoff, end):
    """Per-country means over the forecast years of (damped momentum, network correction,
    total natural change) in log energy intensity, re-implementing one step of
    model.dynamics.rollout without controls. Returns them and whether the recomputed path
    matches the production forecast."""
    hp, norm = wm.hp, wm.norm
    k = _year_index(panel, cutoff)
    gdp = jnp.asarray(panel.gdp[:, k:end - panel.years[0] + 1])
    pop = jnp.asarray(panel.population[:, k:end - panel.years[0] + 1])
    logGDP, logPop = jnp.log(gdp + 1e-6), jnp.log(pop + 1e-6)
    growth = jnp.concatenate([logGDP[:, 1:], logGDP[:, -1:]], axis=1) - logGDP
    ini = observed_inits(panel, k, hp["momentum"], hp["ema_decay"])
    I, s = jnp.asarray(ini["I0"]), jnp.asarray(ini["s0"])
    mI, ms = jnp.asarray(ini["mom_I0"]), jnp.asarray(ini["mom_s0"])
    damped = hp.get("damped_trend", False) and hp["momentum"] != "none"
    phi_I = jax.nn.sigmoid(params["phi"][0]) if damped else 0.0
    phi_s = jax.nn.sigmoid(params["phi"][1]) if damped else 0.0
    rho, max_int, max_share = hp.get("ema_decay", 0.7), hp["max_int"], hp["max_share"]
    tf = jnp.asarray(wm.tfrac(k + np.arange(gdp.shape[1])), dtype=jnp.float32)
    parts = []
    for t in range(gdp.shape[1] - 1):
        D = gdp[:, t] * jnp.exp(I)
        x = _features(params["embed"], jnp.asarray(wm.static), norm, hp, s, I,
                      jnp.log(D + 1e-6), logGDP[:, t], growth[:, t], logPop[:, t],
                      jnp.broadcast_to(tf[t], I.shape), mI, ms)
        h = jax.nn.silu(mlp_apply(params["trunk"], x))
        net_I = max_int * jnp.tanh((h @ params["int_head"]["w"] + params["int_head"]["b"])[:, 0])
        net_s = max_share * jnp.tanh(h @ params["mix_head"]["w"] + params["mix_head"]["b"])
        if damped:
            dI = softclip(phi_I * mI + net_I, max_int)
            ds = softclip(phi_s * ms + net_s, max_share)
        else:
            dI, ds = net_I, net_s
        parts.append((np.asarray(phi_I * mI if damped else 0.0 * mI), np.asarray(net_I), np.asarray(dI)))
        s_nat = jax.nn.relu(s + ds)
        s_nat = s_nat / (jnp.sum(s_nat, axis=1, keepdims=True) + 1e-9)
        if hp["momentum"] == "1yr":
            mI, ms = dI, s_nat - s
        elif hp["momentum"] == "ema":
            mI, ms = rho * mI + (1 - rho) * dI, rho * ms + (1 - rho) * (s_nat - s)
        I, s = I + dI, s_nat
    prod = np.asarray(wm.forecast(params, panel, cutoff, end_year=end)["intensity"][:, -1])
    mom, net, tot = (np.mean(np.stack([p[i] for p in parts]), axis=0) for i in range(3))
    return mom, net, tot, bool(np.allclose(prod, np.asarray(I), atol=1e-5))


def _split_job(job):
    from carbon_world_model.train.fit import fit
    seed, cached_co2 = job
    cfg = default_config()
    panel = Panel.load(cfg)
    r = fit(panel, cfg, train_year_max=SPLIT_CUTOFF, seed=seed, verbose=False)
    end = SPLIT_CUTOFF + VAL_HORIZON
    fc = np.asarray(r.wm.forecast(r.params, panel, SPLIT_CUTOFF, end_year=end)["co2"])
    mom, net, tot, ok = _intensity_split(r.params, r.wm, panel, SPLIT_CUTOFF, end)
    mature = [panel.countries.index(n) for n in MATURE]
    others = [i for i in range(panel.n_countries) if i not in mature]
    pct = lambda a, rows: float(100 * a[rows].mean())  # noqa: E731
    return seed, {
        "reproduces_sweep_output": bool(np.array_equal(fc, cached_co2)),
        "split_matches_production_forecast": ok,
        "mature_pct_per_year": {"damped_momentum": pct(mom, mature),
                                "network_correction": pct(net, mature), "total": pct(tot, mature)},
        "others_pct_per_year": {"damped_momentum": pct(mom, others),
                                "network_correction": pct(net, others), "total": pct(tot, others)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    cfg = default_config()
    panel = Panel.load(cfg)
    held = _held(panel)
    names = [panel.countries[i] for i in held]
    runs, _ = load_jobs(VAL_ROOT, SELECTED_CONFIG, VAL_CUTOFFS, seeds, steps=cfg.train.steps)
    missing = [(co, len(runs.get(co, []))) for co in VAL_CUTOFFS if len(runs.get(co, [])) != len(seeds)]
    if missing:
        raise SystemExit(f"validation outputs missing for {SELECTED_CONFIG} (cutoff, found): {missing}. "
                         f"Run scripts/06_sweep.py --phase val --configs {SELECTED_CONFIG} "
                         f"--seeds {a.seeds} first.")

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=a.jobs, mp_context=ctx) as pool:
        tjobs = [pool.submit(_transfer, (co, s)) for co in VAL_CUTOFFS for s in seeds]
        cached = {s: runs[SPLIT_CUTOFF][i]["co2_realized"] for i, s in enumerate(seeds)}
        sjobs = [pool.submit(_split_job, (s, cached[s])) for s in seeds]
        transfers = [j.result() for j in tjobs]
        splits = dict(j.result() for j in sjobs)

    lI = np.log(panel.primary_energy) - np.log(panel.gdp)
    mature = [panel.countries.index(n) for n in MATURE]
    out = {"config": SELECTED_CONFIG, "held_out": names, "mature": list(MATURE), "seeds": seeds,
           "cutoffs": list(VAL_CUTOFFS), "horizon": VAL_HORIZON, "transfer": {},
           "mature_intensity_trend_pct_per_year": {}}
    for co in VAL_CUTOFFS:
        k = _year_index(panel, co)
        H = VAL_HORIZON
        true, mask = panel.co2[:, k + 1:k + H + 1], panel.mask[:, k + 1:k + H + 1]
        tk = trend_kaya(panel, co)[:, k + 1:k + H + 1]
        rows = np.zeros_like(mask)
        rows[held] = mask[held]
        intrain = {s: runs[co][i]["co2_realized"][:, 1:] for i, s in enumerate(seeds)}
        trs = {s: tr for c, s, tr in transfers if c == co}
        per_country = {}
        for i, n in zip(held, names):
            one = lambda pred: M.mape(pred[i:i + 1], true[i:i + 1], mask[i:i + 1])  # noqa: E731
            per_country[n] = {"transfer": float(np.mean([trs[s]["per_country"][n]["world_model"] for s in seeds])),
                              "in_training": float(np.mean([one(intrain[s]) for s in seeds])),
                              "trend_kaya": float(one(tk))}
        out["transfer"][co] = {
            "by_seed": {s: {"transfer": trs[s]["world_model"]["mape"],
                            "in_training": M.mape(intrain[s], true, rows),
                            "transfer_adapt_insample": trs[s]["adapt_insample_mape"]} for s in seeds},
            "trend_kaya": M.mape(tk, true, rows),
            "per_country": per_country,
        }
        trend = lambda D, i: (np.log(D[i, -1] / panel.gdp[i, k + H]) - np.log(D[i, 0] / panel.gdp[i, k])) / H  # noqa: E731
        out["mature_intensity_trend_pct_per_year"][co] = {
            "model_by_seed": {s: float(100 * np.mean([trend(runs[co][j]["D_realized"], i) for i in mature]))
                              for j, s in enumerate(seeds)},
            "pre_cutoff_10y": float(100 * np.mean([(lI[i, k] - lI[i, k - 10]) / 10 for i in mature])),
            "realised": float(100 * np.mean([(lI[i, k + H] - lI[i, k]) / H for i in mature])),
        }

    from carbon_world_model.plan import optimize as O
    from carbon_world_model.plan.ensemble import load_ensemble
    from carbon_world_model.plan.scenario import build_scenario
    wm = WorldModel.build(panel, cfg)
    scen = build_scenario(panel, cfg, end_year=2050, ssp=None)
    ens_seeds, members = load_ensemble(ENSEMBLE, cfg, checkpoint=CHECKPOINT)
    T = panel.n_years
    bau = {}
    for s, p in zip(ens_seeds, members):
        I = np.log(O.business_as_usual(wm, p, scen).D) - np.log(scen.gdp)
        bau[s] = float(100 * np.mean([(I[i, 10] - I[i, 0]) / 10 for i in mature]))
    out["mature_intensity_trend_pct_per_year"]["planning_bau_2022_2032"] = {
        "by_seed": bau, "observed_2012_2022": float(100 * np.mean([(lI[i, T - 1] - lI[i, T - 11]) / 10
                                                                   for i in mature]))}
    out["intensity_change_split"] = {"cutoff": SPLIT_CUTOFF, "by_seed": splits}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    tm = lambda key: 100 * np.mean([out["transfer"][co]["by_seed"][s][key] for co in VAL_CUTOFFS for s in seeds])  # noqa: E731
    print(f"12 held-out countries, mean over cutoffs {list(VAL_CUTOFFS)} and seeds {seeds}: transfer "
          f"{tm('transfer'):.1f}%, in-training {tm('in_training'):.1f}%, trend-Kaya "
          f"{100 * np.mean([out['transfer'][co]['trend_kaya'] for co in VAL_CUTOFFS]):.1f}%")
    for co in VAL_CUTOFFS:
        m = out["mature_intensity_trend_pct_per_year"][co]
        print(f"mature-economy intensity trend, cutoff {co}: model "
              + ", ".join(f"{v:+.2f}" for v in m["model_by_seed"].values())
              + f" %/yr | pre-cutoff {m['pre_cutoff_10y']:+.2f}, realised {m['realised']:+.2f}")
    b = out["mature_intensity_trend_pct_per_year"]["planning_bau_2022_2032"]
    print(f"planning BAU 2022-2032: mean {np.mean(list(b['by_seed'].values())):+.2f} %/yr "
          f"(observed 2012-2022 {b['observed_2012_2022']:+.2f})")
    for s, d in splits.items():
        print(f"split, cutoff {SPLIT_CUTOFF}, seed {s}: mature {d['mature_pct_per_year']} | refit reproduces "
              f"sweep: {d['reproduces_sweep_output']} | split matches forecast: {d['split_matches_production_forecast']}")
    print("[ok] wrote", OUT)


if __name__ == "__main__":
    main()
