#!/bin/bash
# One-time setup on FASRC Cannon (run from the repository root on a login node or in an
# interactive session: salloc -p test -c 2 -t 30). Needs Python >= 3.10; if the default
# module is older, pick one from `module spider python`.
set -euo pipefail
module load python 2>/dev/null || true
VENV="${CWM_VENV:-$HOME/cwm-venv}"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -e ".[dev]"
python -c "import jax, carbon_world_model; print('jax', jax.__version__, jax.devices()); print('package', carbon_world_model.__file__)"
