"""Write one line of scripts/run_job.py arguments per (config, cutoff, seed) job, for a
SLURM array on FASRC Cannon (or any cluster). Jobs whose output already exists are
skipped unless --all is given, so re-submitting only runs what is missing. An existing
output that was not trained the way its job would be (for example with another --steps)
is an error rather than a skip: delete it, pass --all, or use a separate --root.

Usage:
  python scripts/slurm/make_joblist.py --phase test --configs ms,orig --seeds 0-9 --ssp > jobs.txt
"""
import argparse
import os
import sys

from carbon_world_model.eval.configs import REGISTRY, ProvenanceError
from carbon_world_model.eval.rolling import (
    VAL_CUTOFFS, VAL_HORIZON, TEST_CUTOFFS, TEST_HORIZON, job_path, stale_jobs,
)


def parse_seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["val", "test"], required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--ssp", action="store_true", help="test phase: also SSP-v9 ex-ante forecasts")
    ap.add_argument("--all", action="store_true", help="include jobs whose output already exists")
    a = ap.parse_args()
    cutoffs, horizon = (VAL_CUTOFFS, VAL_HORIZON) if a.phase == "val" else (TEST_CUTOFFS, TEST_HORIZON)
    root = a.root or f"results/sweep_{a.phase}"
    configs, seeds = a.configs.split(","), parse_seeds(a.seeds)
    unknown = [c for c in configs if c not in REGISTRY]
    if unknown:
        ap.error(f"unknown config(s) {', '.join(unknown)}; known: {', '.join(REGISTRY)}")
    stale = [] if a.all else stale_jobs(root, configs, cutoffs, seeds, a.steps)
    if stale:
        raise ProvenanceError(
            "these existing outputs were not trained the way their jobs would be, and would be "
            "skipped:\n  " + "\n  ".join(stale) + "\nDelete them, pass --all to rerun every "
            "job, or use a separate --root.")
    for c in configs:
        for co in cutoffs:
            for s in seeds:
                if not a.all and os.path.exists(job_path(root, c, co, s)):
                    continue
                line = f"--config {c} --cutoff {co} --seed {s} --horizon {horizon} --root {root}"
                if a.steps:
                    line += f" --steps {a.steps}"
                if a.ssp and a.phase == "test":
                    line += " --ssp-vintage v9 --ssps SSP1,SSP2,SSP3,SSP4,SSP5"
                print(line)


if __name__ == "__main__":
    try:
        main()
    except ProvenanceError as e:
        sys.exit(f"make_joblist.py: {e}")
