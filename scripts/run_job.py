"""Worker: fit one configuration on data <= cutoff and save its forecasts.

Usage: python scripts/run_job.py --config ms_full --cutoff 2009 --seed 0 --horizon 6 \
           --root results/sweep [--steps 3000] [--ssp-vintage v9 --ssps SSP1,SSP2]
"""
import argparse
import os

# One BLAS/XLA thread per worker: the sweep runs several workers in parallel.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                                   "intra_op_parallelism_threads=1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

from carbon_world_model.config import default_config  # noqa: E402
from carbon_world_model.data.panel import Panel  # noqa: E402
from carbon_world_model.eval.configs import make_config  # noqa: E402
from carbon_world_model.eval.rolling import run_job, job_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cutoff", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--root", default="results/sweep")
    ap.add_argument("--ssp-vintage", default=None)
    ap.add_argument("--ssps", default="SSP2")
    a = ap.parse_args()
    panel = Panel.load(default_config())
    cfg = make_config(a.config, steps=a.steps)
    out = run_job(panel, cfg, a.cutoff, a.seed, a.horizon, a.ssp_vintage, tuple(a.ssps.split(",")))
    path = job_path(a.root, a.config, a.cutoff, a.seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    print(f"[done] {a.config} c{a.cutoff} s{a.seed} loss={float(out['final_loss']):.4f}")


if __name__ == "__main__":
    main()
