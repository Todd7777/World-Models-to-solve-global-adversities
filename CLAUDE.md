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
pytest             # 2026-11-09: 43 pass
```

If `pytest` cannot import numpy, a standalone pytest on PATH is shadowing the project's interpreter; run
`python -m pytest`.

## State on 2026-11-09

- **All 43 tests pass** with `results/model/params_full.pkl`, retrained on 2026-11-09 with the default (`orig`)
  configuration because model selection is still pending. `scripts/02_train.py` reproduces it bit for bit
  (seed 0). The green suite depends on that seed: `scripts/seed_sensitivity.py` shows that with seed 1
  `test_max_ambition_below_bau` and `test_effort_monotonic` fail, and that the seed-0 projections oscillate.
- **Selection is pending.** `results/sweep_val/report.json` holds round 1 only (12 configs, 2 seeds), run with
  `--steps 3000`, which the report does not record; `02_train.py` trains for 4000. Round 2 (`ms_ema`,
  `ms_damped`, `ms_percap`, `ms_percap_damped`) has not run. Select at the training length you deploy.
- `06_sweep.py` rewrites `<root>/report.json` with only the configs passed. Use `--root` for side runs.
- After selection, make the selected config the default in `config.py` before retraining. `02_train.py`, the
  tests, `04_plan.py` and `05_make_figures.py` all use `default_config()`. `03_evaluate.py --config X` loads
  `params_full.pkl` whatever it was trained with.
- The out-of-sample tables (test folds, calibration, transfer) and the figures wait for selection.
  `results/figures/` predates `3fc2559`. The pre-`3fc2559` backtest, calibration and transfer tables leaked
  post-cutoff data and were removed.
- Branch `claude/clever-fermi-rm4429` holds a separate Space Solar World Model (`src/space_solar_world_model/`).
  It split from this branch at `ed4c64f` and has not been merged.

## Where the next steps are

README "Limitations (read these)" and "What changed on 2026-11-09", `results/sweep_val/report.json`, and
`results/tables/seed_sensitivity.json`.
