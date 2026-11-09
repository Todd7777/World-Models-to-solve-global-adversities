# Running the sweeps on FASRC Cannon (or any SLURM cluster)

Every rolling-origin job (fit one configuration on data up to a cutoff, forecast from the
observed state) is independent and single-threaded, so the evaluation scales linearly on
a cluster. This kit runs the exact same jobs as `scripts/06_sweep.py`, as a SLURM array.

These scripts were smoke-tested locally by simulating the array-task environment; they
have not been submitted to Cannon itself. Partition and Python module names on FASRC
change over time, so check `sinfo` and `module spider python` if a default fails.

## 1. One-time setup (login node or `salloc -p test -c 2 -t 30`)

```bash
git clone https://github.com/Todd7777/World-Models-to-solve-global-adversities.git
cd World-Models-to-solve-global-adversities
git checkout claude/amazing-ritchie-026u0w
bash scripts/slurm/setup_fasrc.sh          # venv in ~/cwm-venv (override with CWM_VENV)
```

The processed panel and SSP series are committed, so no data download is needed.

## 2. Build a job list and submit

Validation folds (model selection) or test folds (reporting), any configs and seeds.
Already-finished jobs are skipped, so re-running these two lines resumes cleanly.

```bash
source ~/cwm-venv/bin/activate
python scripts/slurm/make_joblist.py --phase test --configs ms,orig --seeds 0-19 --ssp > jobs.txt
mkdir -p logs && sbatch --array=1-$(wc -l < jobs.txt) scripts/slurm/sweep_array.sbatch jobs.txt
```

Use `--partition=serial_requeue` on the `sbatch` line for the cheapest (preemptible)
queue; jobs are short (about 1-3 minutes each) and safely re-runnable.

Each output records how it was trained. If you pass `--steps N` here, pass the same
`--steps N` to `06_sweep.py` below: outputs trained for another number of steps (or
older outputs without the record) are refused, not skipped or pooled.

## 3. Aggregate

When the array finishes, score everything (this runs no new fits when all outputs exist):

```bash
python scripts/06_sweep.py --phase test --configs ms,orig --seeds 0,1,2,3,4,5,6,7,8,9 --ssp
```

This merges the configs passed into `results/sweep_test/report.json` (other entries are
kept; `--overwrite` replaces the file), which is tracked by git; commit it to bring the
results back. A report holds one training length: the sweep refuses to add entries
trained for another number of steps. The per-job `.npz` caches are git-ignored; copy
them with `rsync` if you want to re-analyse them elsewhere.
