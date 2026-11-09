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
pytest             # 2026-11-09: 54 pass
```

If `pytest` cannot import numpy, a standalone pytest on PATH is shadowing the project's interpreter; run
`python -m pytest`.

## State on 2026-11-09

- **All 54 tests pass** with `results/model/params_full.pkl`, retrained on 2026-11-09 with the default (`orig`)
  configuration because model selection is still pending. `scripts/02_train.py` reproduces it bit for bit
  (seed 0). The green suite depends on that seed: `scripts/seed_sensitivity.py` shows that with seed 1
  `test_max_ambition_below_bau` and `test_effort_monotonic` fail, and that the seed-0 projections oscillate.
- **Selection is pending.** `results/sweep_val/report.json` holds round 1 only (12 configs, 2 seeds), run with
  `--steps 3000`; `02_train.py` trains for 4000. The report predates step recording: its entries have no `steps`
  field and are read as 3000 (`LEGACY_REPORT_STEPS` in `eval/rolling.py`). Round 2 (`ms_ema`, `ms_damped`,
  `ms_percap`, `ms_percap_damped`) has not run. Select at the training length you deploy.
- Sweep job outputs record their step count and full config; report entries record their step count. Scores
  never pool outputs trained differently (apart from the seed) or without that record. Older job caches lack
  it; delete and rerun them.
- `06_sweep.py` merges into `<root>/report.json`, replacing only the configs passed; `--overwrite` replaces the
  whole report. Before fitting anything it refuses to put a second training length (or other folds) into a
  report, and to reuse cached outputs trained differently from what the run asks for. Adding round 2 to the
  committed report therefore needs `--steps 3000`. Selecting at 4000 needs a fresh `--root`, a run that
  re-scores every config in the report (all 16 then replace round 1), or `--overwrite` (which drops the configs
  not passed). Use `--root` for side runs.
- After selection, make the selected config the default in `config.py` before retraining. `02_train.py`, the
  tests, `03_backtest.py`, `04_plan.py` and `05_make_figures.py` all use `default_config()`, which is why
  `02_train.py` has no `--config` flag. `03_evaluate.py --config X` refuses to run (before writing anything)
  unless `results/tables/config.json`, written by `02_train.py` with the checkpoint, matches `make_config(X)`,
  and unless every sweep output it reads was trained for as many steps as the checkpoint.
- The out-of-sample tables (test folds, calibration, transfer) and the figures wait for selection.
  `results/figures/` predates `3fc2559`. The pre-`3fc2559` backtest, calibration and transfer tables leaked
  post-cutoff data and were removed.
- Branch `claude/clever-fermi-rm4429` holds a separate Space Solar World Model (`src/space_solar_world_model/`).
  It split from this branch at `ed4c64f` and has not been merged.

## Where the next steps are

README "Limitations (read these)" and "What changed on 2026-11-09", `results/sweep_val/report.json`, and
`results/tables/seed_sensitivity.json`.
