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
pytest             # 2026-11-10: 60 pass
```

If `pytest` cannot import numpy, a standalone pytest on PATH is shadowing the project's interpreter; run
`python -m pytest`.

## State on 2026-11-09

- **All 60 tests pass** with `results/model/params_full.pkl`, retrained on 2026-11-09 with the default (`orig`)
  configuration because model selection is still pending. `scripts/02_train.py` reproduces it bit for bit
  (seed 0). The green suite depends on that seed: `scripts/seed_sensitivity.py` shows that with seed 1
  `test_max_ambition_below_bau` and `test_effort_monotonic` fail, and that the seed-0 projections oscillate.
- **Selection is pending.** `results/sweep_val/report.json` (2026-11-10) scores all 16 configs at 4000 steps, the
  length `02_train.py` trains for, with 2 seeds. Multiple shooting wins on every seed. The leaders
  (`ms_percap_damped` 11.05, `ms_full_shootonly` 11.29, `ms_full` 11.70) are closer than the seed noise (one
  config's score moves by about 1.5 between seeds), so they need more seeds, the same set for each, before one is
  chosen. trend-Kaya (9.85) still beats every config. Round 1 (12 configs at 3000 steps; its entries have no
  `steps` field and are read as 3000 via `LEGACY_REPORT_STEPS`) is in `tests/data/report_round1_3000steps.json`.
- Sweep job outputs record their step count and full config; report entries record their step count. Scores
  never pool outputs trained differently (apart from the seed) or without that record. Older job caches lack
  it; delete and rerun them.
- `06_sweep.py` merges into `<root>/report.json`, replacing only the configs passed; `--overwrite` replaces the
  whole report. Before fitting anything it refuses to put a second training length (or other folds) into a
  report, and to reuse cached outputs trained differently from what the run asks for. It does not check seeds:
  an ensemble of more seeds usually scores better, so compare configs only on the same seed set. Use `--root`
  for side runs, such as more seeds for the leaders.
- After selection, make the selected config the default in `config.py`, then retrain at once. `02_train.py`, the
  tests, `03_backtest.py`, `04_plan.py` and `05_make_figures.py` all use `default_config()`, which is why
  `02_train.py` has no `--config` flag. `results/tables/config.json`, written by `02_train.py` with the
  checkpoint, records what it was trained with. Until the retrain it no longer matches `default_config()`, so
  `test_committed_checkpoint_matches_default_config` fails and `03_backtest.py`, `04_plan.py` and
  `05_make_figures.py` refuse to run. `03_evaluate.py --config X` refuses unless `config.json` matches
  `make_config(X)` and every sweep output it reads was trained for as many steps as the checkpoint. All of
  these scripts refuse before writing anything.
- The out-of-sample tables (test folds, calibration, transfer) and the figures wait for selection.
  `results/figures/` predates `3fc2559`. The pre-`3fc2559` backtest, calibration and transfer tables leaked
  post-cutoff data and were removed.
- Branch `claude/clever-fermi-rm4429` holds a separate Space Solar World Model (`src/space_solar_world_model/`).
  It split from this branch at `ed4c64f` and has not been merged.

## Where the next steps are

README "Limitations (read these)" and "What changed on 2026-11-09", `results/sweep_val/report.json`, and
`results/tables/seed_sensitivity.json`.
