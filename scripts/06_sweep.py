"""Rolling-origin sweep driver: run (config x cutoff x seed) jobs in parallel worker
processes (resumable: finished jobs are cached on disk), then score them and merge the
scores into <root>/report.json, replacing only the entries of the configs passed.

Model selection must use --phase val only. --phase test is for the final report.

Every job output records how it was trained. Before fitting anything the sweep refuses
  * cached outputs that were not trained the way this run would train them (for example
    with another --steps): the sweep would otherwise reuse them;
  * merging into a report whose other configs were trained for a different number of
    steps, or scored on other folds. A report holds one training length.
--overwrite replaces the report with this run's configs instead of merging. Report
entries without a step count are the round-1 validation sweep, run with --steps 3000.

Usage:
  python scripts/06_sweep.py --phase val --configs orig,ms,ms_full --seeds 0,1 --jobs 4
  python scripts/06_sweep.py --phase test --configs ms_full,orig --seeds 0,1,2 --jobs 4 --ssp
"""
import argparse
import json
import os
import subprocess
import sys
import time

from carbon_world_model.config import default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.eval.configs import REGISTRY, ProvenanceError, make_config
from carbon_world_model.eval.rolling import (
    VAL_CUTOFFS, VAL_HORIZON, TEST_CUTOFFS, TEST_HORIZON, job_path, evaluate, report_entry,
    entry_config, describe_steps, report_conflicts, merge_report, stale_jobs, save_json,
)


def run_pool(cmds, jobs: int) -> None:
    pending, running = list(cmds), []
    t0 = time.time()
    while pending or running:
        while pending and len(running) < jobs:
            cmd = pending.pop(0)
            running.append((cmd, subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                                  stderr=subprocess.STDOUT, text=True)))
        time.sleep(0.5)
        still = []
        for cmd, p in running:
            if p.poll() is None:
                still.append((cmd, p))
                continue
            out = p.stdout.read().strip().splitlines()
            tail = out[-1] if out else ""
            status = "ok" if p.returncode == 0 else f"FAILED rc={p.returncode}"
            print(f"  [{time.time()-t0:6.0f}s] {status}: {tail}", flush=True)
            if p.returncode != 0:
                print("\n".join(out[-15:]), flush=True)
        running = still


def print_report(report: dict, path: str, phase: str, cutoffs, updated) -> None:
    n = len({entry_config(k) for k in report})
    print(f"\n[{phase}] {path}: {n} configs, trained for {describe_steps(report.values())}")
    print(f"mean over cutoffs {list(cutoffs)} (lower is better; score = country MAPE% + mean "
          f"|aggregate error|%); * = scored in this run")
    print(f"{'config|driver':34s} {'method':20s} {'MAPE%':>6s} {'|agg|%':>7s} {'agg_fin%':>8s} "
          f"{'China%':>7s} {'score':>6s}")
    drivers_seen = set()
    for key, r in report.items():
        s = r["summary"]
        driver = key.split("|", 1)[1]
        label = key + ("*" if entry_config(key) in updated else "")
        for m in ["world_model", "trend_kaya", "growth_persistence"]:
            if m not in s:
                continue
            if m != "world_model" and driver in drivers_seen:
                continue  # baselines are identical across configs; print once per driver
            v = s[m]
            print(f"{label:34s} {m:20s} {100*v['country_mape']:6.2f} "
                  f"{100*v['agg_abs_err_mean']:7.2f} {100*v['agg_err_final']:+8.2f} "
                  f"{100*v['china_err_final']:+7.2f} {v['score']:6.2f}")
        drivers_seen.add(driver)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["val", "test"], required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--ssp", action="store_true", help="also make SSP-v9 ex-ante forecasts")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace <root>/report.json with this run's configs instead of merging")
    a = ap.parse_args()

    configs = a.configs.split(",")
    unknown = [c for c in configs if c not in REGISTRY]
    if unknown:
        ap.error(f"unknown config(s) {', '.join(unknown)}; known: {', '.join(REGISTRY)}")
    seeds = [int(s) for s in a.seeds.split(",")]
    cutoffs, horizon = (VAL_CUTOFFS, VAL_HORIZON) if a.phase == "val" else (TEST_CUTOFFS, TEST_HORIZON)
    root = a.root or f"results/sweep_{a.phase}"
    report_path = os.path.join(root, "report.json")
    steps = {c: make_config(c, steps=a.steps).train.steps for c in configs}

    # Refuse before fitting anything: a report that cannot take these entries, and cached
    # outputs that the resumable sweep would reuse although they were trained differently.
    old = {}
    if os.path.exists(report_path):
        if a.overwrite:
            print(f"[sweep] --overwrite: {report_path} will hold only {configs}", flush=True)
        else:
            with open(report_path) as fh:
                old = json.load(fh)
    conflicts = report_conflicts(old, steps, cutoffs)
    if conflicts:
        raise ProvenanceError(
            f"cannot merge into {report_path}: " + "; ".join(conflicts) + ". Use --root to "
            "write a separate report, or --overwrite to replace this one with this run's "
            "configs only.")
    stale = stale_jobs(root, configs, cutoffs, seeds, a.steps)
    if stale:
        raise ProvenanceError(
            "these cached outputs were not trained the way this run would train them, and the "
            "sweep would reuse them:\n  " + "\n  ".join(stale) + "\nDelete them to have them "
            "rerun, or use a separate --root.")

    cmds = []
    for c in configs:
        for co in cutoffs:
            for s in seeds:
                if os.path.exists(job_path(root, c, co, s)):
                    continue
                cmd = [sys.executable, "scripts/run_job.py", "--config", c, "--cutoff", str(co),
                       "--seed", str(s), "--horizon", str(horizon), "--root", root]
                if a.steps:
                    cmd += ["--steps", str(a.steps)]
                if a.ssp and a.phase == "test":
                    cmd += ["--ssp-vintage", "v9", "--ssps", "SSP1,SSP2,SSP3,SSP4,SSP5"]
                cmds.append(cmd)
    print(f"[sweep] phase={a.phase} configs={configs} cutoffs={list(cutoffs)} seeds={seeds} "
          f"steps={sorted(set(steps.values()))} -> {len(cmds)} jobs to run ({a.jobs} parallel)",
          flush=True)
    run_pool(cmds, a.jobs)

    panel = Panel.load(default_config())
    drivers = ["realized"] + (["ssp"] if (a.ssp and a.phase == "test") else [])
    new = {}
    for c in configs:
        for d in drivers:
            res = evaluate(panel, c, cutoffs, seeds, root, driver=d, steps=steps[c])
            if res["cutoffs"]:
                new[f"{c}|{d}"] = report_entry(res)
    missing = [c for c in configs if not any(entry_config(k) == c for k in new)]
    if missing:
        print(f"[sweep] no job outputs for {', '.join(missing)}, so the report has no entries "
              "for them", flush=True)
    report = merge_report(old, new, configs)
    save_json(report, report_path)
    print_report(report, report_path, a.phase, cutoffs, set(configs))


if __name__ == "__main__":
    try:
        main()
    except ProvenanceError as e:
        sys.exit(f"06_sweep.py: {e}")
