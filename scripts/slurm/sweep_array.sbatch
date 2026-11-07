#!/bin/bash
# SLURM array job: each task runs one (config, cutoff, seed) fit+forecast.
# FASRC Cannon partitions: shared (default here), serial_requeue (cheapest, preemptible),
# sapphire, test (short debugging). One CPU per task is right: each fit is single-threaded.
#SBATCH --job-name=cwm-sweep
#SBATCH --partition=shared
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cwm_%A_%a.out
#
# Usage (from the repository root):
#   mkdir -p logs
#   sbatch --array=1-$(wc -l < jobs.txt) scripts/slurm/sweep_array.sbatch jobs.txt
set -euo pipefail
JOBS="${1:-jobs.txt}"
TASK="${SLURM_ARRAY_TASK_ID:?run this as a SLURM array task}"
ARGS="$(sed -n "${TASK}p" "$JOBS")"
[ -n "$ARGS" ] || { echo "no line $TASK in $JOBS"; exit 1; }
VENV="${CWM_VENV:-$HOME/cwm-venv}"
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
echo "[task $TASK] python scripts/run_job.py $ARGS"
# shellcheck disable=SC2086
python scripts/run_job.py $ARGS
