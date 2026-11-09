# CLAUDE.md

Guidance for Claude Code sessions in this repository, local or cloud.

## What this is

Carbon World Model: a differentiable mechanistic-neural simulator of the energy–emissions systems of 60
countries (about 93% of global CO₂), trained by differentiable simulation in JAX, with cost-optimal
decarbonisation pathways found by gradient descent through the model. It is a research prototype, not a
forecast and not policy advice.

## Ground rules

- Results are conditional on documented assumptions. Report limitations as prominently as results (README,
  "Limitations (read these)"), and never present outputs as forecasts or policy advice.
- Evaluate leak-free: nothing after a cutoff year may influence a model evaluated at that cutoff.
- Run the checks before and after every change. A new failure is a regression until shown otherwise.
- Commit verified work in small steps and push before the session ends. Work left only inside a cloud
  session can be lost with it.

## Setup and checks

```bash
pip install -e ".[dev]"
pytest             # 2026-11-08: 40 pass, 3 fail (see below)
```

## State on 2026-11-08

- **Three tests fail on this branch:** `test_trained_insample_accuracy` (r² 0.54, needs > 0.98),
  `test_max_ambition_below_bau` and `test_effort_monotonic`. Commit `3fc2559` changed the dynamics, and the
  committed checkpoint `results/model/params_full.pkl` predates it, as that commit's message says. Its parent
  `77b9669` passes all ten dynamics and planning tests. The fix is the retrain that follows the rolling-origin
  selection sweep (`scripts/06_sweep.py`, with a SLURM kit in `scripts/slurm/`).
- The README's headline numbers and `results/tables/` also predate `3fc2559`. Regenerate them after the retrain.
- Branch `claude/clever-fermi-rm4429` holds a separate Space Solar World Model (`src/space_solar_world_model/`).
  It split from this branch at `ed4c64f` and has not been merged.

## Where the next steps are

README "Limitations (read these)" and `results/sweep_val/report.json`.
