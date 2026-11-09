"""Rolling-origin sweep driver: run (config x cutoff x seed) jobs in parallel worker
processes (resumable: finished jobs are cached on disk), then score and summarise.

Model selection must use --phase val only. --phase test is for the final report.

Every job output records how it was trained. Before fitting anything the sweep refuses
cached outputs that were not trained the way this run would train them (for example with
another --steps): the sweep would otherwise reuse them. Report entries record the step
count of the outputs they score.

Usage:
  python scripts/06_sweep.py --phase val --configs orig,ms,ms_full --seeds 0,1 --jobs 4
  python scripts/06_sweep.py --phase test --configs ms_full,orig --seeds 0,1,2 --jobs 4 --ssp
"""
import argparse
import os
import subprocess
import sys
import time

from carbon_world_model.config import default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.eval.configs import REGISTRY, ProvenanceError, make_config
from carbon_world_model.eval.rolling import (
    VAL_CUTOFFS, VAL_HORIZON, TEST_CUTOFFS, TEST_HORIZON, job_path, evaluate, report_entry,
    stale_jobs, save_json,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["val", "test"], required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--ssp", action="store_true", help="also make SSP-v9 ex-ante forecasts")
    a = ap.parse_args()

    configs = a.configs.split(",")
    unknown = [c for c in configs if c not in REGISTRY]
    if unknown:
        ap.error(f"unknown config(s) {', '.join(unknown)}; known: {', '.join(REGISTRY)}")
    seeds = [int(s) for s in a.seeds.split(",")]
    cutoffs, horizon = (VAL_CUTOFFS, VAL_HORIZON) if a.phase == "val" else (TEST_CUTOFFS, TEST_HORIZON)
    root = a.root or f"results/sweep_{a.phase}"
    steps = {c: make_config(c, steps=a.steps).train.steps for c in configs}

    # Refuse before fitting anything: cached outputs that the resumable sweep would reuse
    # although they were trained differently.
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
    report = {}
    for c in configs:
        for d in drivers:
            res = evaluate(panel, c, cutoffs, seeds, root, driver=d, steps=steps[c])
            report[f"{c}|{d}"] = report_entry(res)
    save_json(report, os.path.join(root, "report.json"))

    print(f"\n[{a.phase}] mean over cutoffs {list(cutoffs)} (lower is better; score = "
          f"country MAPE% + mean |aggregate error|%)")
    print(f"{'config|driver':34s} {'method':20s} {'MAPE%':>6s} {'|agg|%':>7s} {'agg_fin%':>8s} "
          f"{'China%':>7s} {'score':>6s}")
    for key, r in report.items():
        s = r["summary"]
        for m in ["world_model", "trend_kaya", "growth_persistence"]:
            if m not in s:
                continue
            v = s[m]
            if m != "world_model" and not key.startswith(configs[0]):
                continue  # baselines are identical across configs; print once
            print(f"{key:34s} {m:20s} {100*v['country_mape']:6.2f} {100*v['agg_abs_err_mean']:7.2f} "
                  f"{100*v['agg_err_final']:+8.2f} {100*v['china_err_final']:+7.2f} {v['score']:6.2f}")


if __name__ == "__main__":
    try:
        main()
    except ProvenanceError as e:
        sys.exit(f"06_sweep.py: {e}")
