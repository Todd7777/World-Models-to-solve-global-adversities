"""Rolling-origin evaluation with a strict validation / test split.

Protocol
--------
* Every fit uses only data up to its cutoff year (the world model is built with
  ``upto_year=cutoff``: emission factors, normalisation and static features included).
* Every forecast starts from the OBSERVED state in the cutoff year -- exactly what the
  baselines do -- and rolls forward. Process CO2 is held at its cutoff value.
* VALIDATION cutoffs (2005, 2007, 2009; 6-year horizons ending <= 2015) are the only
  ones used to choose a configuration. TEST cutoffs (2015, 2016, 2017; holdouts after
  2015) are scored once, for reporting.
* Drivers: 'realized' gives every method the realised GDP/population path (a
  conditional forecast); 'ssp' gives every method an SSP path harmonised at the cutoff
  (an ex-ante forecast, using the 2014-vintage SSP database for 2015-2017 cutoffs).
* Provenance: every job output records how it was trained (the step count, and the whole
  configuration with the job's seed). Outputs are pooled into one score only if they were
  trained the same way apart from the seed, and report entries record the step count.
"""
from __future__ import annotations

import dataclasses
import json
import os

import numpy as np

from ..config import Config
from ..data.ssp import load_processed, ssp_driver_paths
from .baselines import BASELINES
from .configs import FIT_SECTIONS, ProvenanceError, config_diff, config_dict, make_config
from . import metrics as M

VAL_CUTOFFS = (2005, 2007, 2009)
VAL_HORIZON = 6
TEST_CUTOFFS = (2015, 2016, 2017)
TEST_HORIZON = 7

# The committed round-1 validation report (results/sweep_val/report.json) predates step
# recording. It was run with --steps 3000: rerunning its jobs reproduces its per-cutoff
# metrics exactly only at 3000 steps. Report entries without a 'steps' field are read so.
LEGACY_REPORT_STEPS = 3000

NO_RECORD = ("no record of how it was trained (it was written before job outputs recorded "
             "their training configuration)")


def horizon_for(panel, cutoff: int, horizon: int) -> int:
    return int(min(horizon, int(panel.years[-1]) - cutoff))


def job_path(root: str, config: str, cutoff: int, seed: int) -> str:
    return os.path.join(root, config, f"c{cutoff}_s{seed}.npz")


def with_seed(cfg: Config, seed: int) -> Config:
    """`cfg` with the training seed that a job actually uses."""
    return dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, seed=seed))


def training_record(cfg: Config, seed: int) -> dict:
    """Arrays that record how a job was trained: the step count, and the whole configuration
    (with the job's seed) as JSON."""
    return {"steps": np.array(cfg.train.steps),
            "config": np.array(json.dumps(config_dict(with_seed(cfg, seed))))}


def job_record(run) -> dict | None:
    """{'steps', 'config'} recorded in a job output (run_job's dict or np.load's result), or
    None if the output predates the record."""
    if "steps" not in run or "config" not in run:
        return None
    return {"steps": int(run["steps"]), "config": json.loads(run["config"].item())}


def job_mismatch(path: str, cfg: Config) -> list[str]:
    """How the job output at `path` was trained differently from `cfg` (the configuration
    it would be trained with now, seed included); empty if it was not."""
    with np.load(path) as run:
        rec = job_record(run)
    if rec is None:
        return [NO_RECORD]
    return [f"{k}: recorded {x!r}, expected {y!r}"
            for k, x, y in config_diff(rec["config"], config_dict(cfg), FIT_SECTIONS)]


def stale_jobs(root: str, configs, cutoffs, seeds, steps: int | None = None) -> list[str]:
    """Existing outputs under `root` that were not trained the way this sweep would train them
    (make_config(name, steps=steps) with the job's seed), one line each. A resumable sweep
    skips jobs whose output exists, so it would otherwise reuse these silently."""
    out = []
    for c in configs:
        cfg = make_config(c, steps=steps)
        for co in cutoffs:
            for s in seeds:
                path = job_path(root, c, co, s)
                if os.path.exists(path):
                    diff = job_mismatch(path, with_seed(cfg, s))
                    if diff:
                        out.append(f"{path}: {'; '.join(diff)}")
    return out


def _pooled_fit(config: dict) -> dict:
    """The sections of a recorded config that determine a fit, without the per-job seed."""
    fit = {s: config.get(s) for s in FIT_SECTIONS}
    fit["train"] = {k: v for k, v in (fit["train"] or {}).items() if k != "seed"}
    return fit


def load_jobs(root: str, config: str, cutoffs, seeds, steps: int | None = None):
    """Load the existing outputs of `config` for these cutoffs and seeds, checking that they
    can be pooled into one score.

    Returns (runs, fit): `runs` maps each cutoff that has outputs to the loaded outputs, one per
    seed found; `fit` is how all of them were trained (FIT_SECTIONS, without the seed), or None
    if there are none. Raises ProvenanceError if an output predates the training record, if the
    outputs were trained differently apart from the seed, or if `steps` is given and they were
    trained for another number of steps.
    """
    runs: dict[int, list] = {}
    fits: dict[str, dict] = {}
    for co in cutoffs:
        for s in seeds:
            path = job_path(root, config, co, s)
            if not os.path.exists(path):
                continue
            run = np.load(path)
            rec = job_record(run)
            if rec is None:
                raise ProvenanceError(f"{path} has {NO_RECORD}, so it cannot be pooled. Delete "
                                      "it and rerun the sweep, or use a fresh --root.")
            runs.setdefault(int(co), []).append(run)
            fits[path] = _pooled_fit(rec["config"])
    if not fits:
        return runs, None
    groups: dict[str, tuple[dict, list[str]]] = {}
    for path, f in fits.items():
        groups.setdefault(json.dumps(f, sort_keys=True), (f, []))[1].append(path)
    (fit, _), *others = groups.values()
    if others:
        fields = dict.fromkeys(k for f, _ in others for k, _, _ in config_diff(fit, f))
        raise ProvenanceError(
            f"refusing to pool the outputs of {config!r} in {root}: they were not all trained "
            f"the same way (differing fields: {', '.join(fields)}).\n"
            + "\n".join(f"  {f['train'].get('steps')} steps: "
                        + ", ".join(os.path.relpath(p, root) for p in ps)
                        for f, ps in groups.values())
            + "\nDelete the outputs that do not match and rerun the sweep, or keep each "
              "training length under its own --root.")
    if steps is not None and fit["train"].get("steps") != steps:
        raise ProvenanceError(f"the outputs of {config!r} in {root} were trained for "
                              f"{fit['train'].get('steps')} steps, not the {steps} required.")
    return runs, fit


def run_job(panel, cfg: Config, cutoff: int, seed: int, horizon: int,
            ssp_vintage: str | None = None, ssps=("SSP2",)) -> dict:
    """Fit on data <= cutoff and forecast `horizon` years from the observed state.

    The output records how the fit was trained (training_record), so cached outputs can be
    checked before they are reused or pooled."""
    from ..train.fit import fit  # local import keeps module import light
    H = horizon_for(panel, cutoff, horizon)
    r = fit(panel, cfg, train_year_max=cutoff, seed=seed, verbose=False)
    fc = r.wm.forecast(r.params, panel, cutoff, end_year=cutoff + H)
    out = {"co2_realized": np.asarray(fc["co2"]), "D_realized": np.asarray(fc["D"]),
           "shares_realized": np.asarray(fc["shares"]), "H": np.array(H),
           "final_loss": np.array(r.history[-1]["loss"]), **training_record(cfg, seed)}
    if ssp_vintage:
        series = load_processed(ssp_vintage)
        years = np.arange(cutoff, cutoff + H + 1)
        for s in ssps:
            g, p = ssp_driver_paths(panel, series, s, cutoff, years)
            fcs = r.wm.forecast(r.params, panel, cutoff, gdp=g, pop=p)
            out[f"co2_{s}"] = np.asarray(fcs["co2"])
    return out


def panel_with_drivers(panel, cutoff: int, gdp_path: np.ndarray, pop_path: np.ndarray):
    """Copy of the panel whose GDP/population after `cutoff` follow the given paths."""
    k = int(np.where(panel.years == cutoff)[0][0])
    H = gdp_path.shape[1] - 1
    g, p = panel.gdp.copy(), panel.population.copy()
    g[:, k:k + H + 1] = gdp_path
    p[:, k:k + H + 1] = pop_path
    return dataclasses.replace(panel, gdp=g, population=p)


def _metrics(pred: np.ndarray, true: np.ndarray, mask: np.ndarray, china: int) -> dict:
    """pred/true/mask: [C, H] over holdout years (index 0 = first year after cutoff)."""
    agg_p, agg_t = (pred * mask).sum(0), (true * mask).sum(0)
    agg_err = (agg_p - agg_t) / agg_t
    return {
        "country_mape": M.mape(pred, true, mask),
        "rmse_mt": M.rmse(pred, true, mask),
        "agg_err_final": float(agg_err[-1]),
        "agg_abs_err_mean": float(np.mean(np.abs(agg_err))),
        "china_err_final": float((pred[china, -1] - true[china, -1]) / true[china, -1]),
        "mape_by_horizon": [M.mape(pred[:, h], true[:, h], mask[:, h])
                            for h in range(pred.shape[1])],
    }


def evaluate(panel, config: str, cutoffs, seeds, root: str, driver: str = "realized",
             ssp: str = "SSP2", ssp_vintage: str = "v9", steps: int | None = None) -> dict:
    """Score the seed-ensemble world model and all baselines at each cutoff.

    Pools only outputs trained the same way apart from the seed (see load_jobs); if `steps` is
    given they must also have been trained for that many steps. The result records the shared
    step count and training configuration (`steps`, `train`), and each cutoff's row its steps.
    """
    runs_by_cutoff, fit = load_jobs(root, config, cutoffs, seeds, steps=steps)
    china = panel.countries.index("China")
    res = {"config": config, "driver": driver,
           "steps": fit["train"]["steps"] if fit else None,
           "train": fit["train"] if fit else None, "cutoffs": {}}
    for co in cutoffs:
        runs = runs_by_cutoff.get(int(co))
        if not runs:
            continue
        key = "co2_realized" if driver == "realized" else f"co2_{ssp}"
        H = int(runs[0]["H"])
        wm_fc = np.mean([r[key] for r in runs], axis=0)       # [C, H+1]
        k = int(np.where(panel.years == co)[0][0])
        true = panel.co2[:, k + 1:k + H + 1]
        mask = panel.mask[:, k + 1:k + H + 1]
        bpanel = panel
        if driver != "realized":
            series = load_processed(ssp_vintage)
            g, p = ssp_driver_paths(panel, series, ssp, co, np.arange(co, co + H + 1))
            bpanel = panel_with_drivers(panel, co, g, p)
        row = {"world_model": _metrics(wm_fc[:, 1:], true, mask, china), "n_seeds": len(runs),
               "steps": res["steps"]}
        for name, fn in BASELINES.items():
            row[name] = _metrics(fn(bpanel, co)[:, k + 1:k + H + 1], true, mask, china)
        res["cutoffs"][int(co)] = row
    return res


_ROW_INFO = ("n_seeds", "steps")  # fields of an evaluate() row that are not methods


def summarise(res: dict, methods=None) -> dict:
    """Average metrics over cutoffs, per method; adds a selection score.

    Refuses to average cutoffs whose rows record different step counts. Rows of legacy
    reports record none and are averaged as before.
    """
    cuts = list(res["cutoffs"].values())
    if not cuts:
        return {}
    if len({c.get("steps") for c in cuts}) > 1:
        per_cutoff = ", ".join(f"{co}: {c.get('steps')}" for co, c in res["cutoffs"].items())
        raise ProvenanceError(f"refusing to average the cutoffs of {res.get('config')!r}: their "
                              f"forecasts were trained for different step counts ({per_cutoff})")
    methods = methods or [m for m in cuts[0] if m not in _ROW_INFO]
    out = {}
    for m in methods:
        keys = ("country_mape", "rmse_mt", "agg_err_final", "agg_abs_err_mean", "china_err_final")
        agg = {k: float(np.mean([c[m][k] for c in cuts])) for k in keys}
        agg["abs_china_err_final"] = float(np.mean([abs(c[m]["china_err_final"]) for c in cuts]))
        agg["score"] = 100 * (agg["country_mape"] + agg["agg_abs_err_mean"])
        out[m] = agg
    return out


# ------------------------------------------------------------------ report.json entries
# A report maps "<config>|<driver>" to {"steps", "summary", "detail"}, where "detail" is the
# evaluate() result. One report holds one training length and one set of folds.

def report_entry(res: dict) -> dict:
    """The report entry for one evaluate() result."""
    return {"steps": res["steps"], "summary": summarise(res), "detail": res}


def entry_config(key: str) -> str:
    return key.split("|")[0]


def entry_steps(entry: dict) -> int:
    """Training steps of a report entry; legacy entries record none (LEGACY_REPORT_STEPS)."""
    return int(entry.get("steps", LEGACY_REPORT_STEPS))


def describe_steps(entries) -> str:
    """'N steps' for report entries, noting legacy entries that do not record it."""
    entries = list(entries)
    text = "/".join(str(n) for n in sorted({entry_steps(e) for e in entries})) + " steps"
    legacy = sum("steps" not in e for e in entries)
    if legacy:
        text += (f" ({legacy} legacy entries record no step count and are read as the "
                 f"round-1 validation sweep, run with --steps {LEGACY_REPORT_STEPS})")
    return text


def report_conflicts(old: dict, steps: dict, cutoffs) -> list[str]:
    """Why entries for the configurations in `steps` (name -> training steps), scored at
    `cutoffs`, cannot be merged into the report `old`; empty if they can."""
    kept = {k: v for k, v in old.items() if entry_config(k) not in steps}
    kept_steps = {entry_steps(v) for v in kept.values()}
    new_steps = sorted(set(steps.values()))
    problems = []
    if len(new_steps) > 1:
        problems.append("this run's configurations train for different numbers of steps ("
                        + ", ".join(f"{c}: {n}" for c, n in steps.items()) + ")")
    elif kept and kept_steps != set(new_steps):
        n_kept = len({entry_config(k) for k in kept})
        problems.append(f"it holds {n_kept} other configurations trained for "
                        f"{describe_steps(kept.values())}, but this run trains for "
                        f"{new_steps[0]} steps"
                        + (f" (pass --steps {next(iter(kept_steps))} to match)"
                           if len(kept_steps) == 1 else ""))
    kept_cutoffs = {int(c) for v in kept.values() for c in v["detail"]["cutoffs"]}
    if not kept_cutoffs <= set(cutoffs):
        problems.append(f"its entries were scored at cutoffs {sorted(kept_cutoffs)}, not this "
                        f"run's {list(cutoffs)}")
    return problems


def merge_report(old: dict, new: dict, configs) -> dict:
    """`old` with the entries of `configs` replaced by theirs in `new`, in their old place
    (configurations new to the report are appended). A configuration in `configs` without
    entries in `new` is dropped."""
    configs = set(configs)
    out, placed = {}, set()
    for k, v in old.items():
        c = entry_config(k)
        if c not in configs:
            out[k] = v
        elif c not in placed:
            out.update({nk: nv for nk, nv in new.items() if entry_config(nk) == c})
            placed.add(c)
    out.update({k: v for k, v in new.items() if entry_config(k) not in placed})
    return out


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
