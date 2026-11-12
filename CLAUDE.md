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
pytest             # 2026-11-12: 70 pass
```

If `pytest` cannot import numpy, a standalone pytest on PATH is shadowing the project's interpreter; run
`python -m pytest`.

## State on 2026-11-12

- **All 70 tests pass** with `results/model/params_full.pkl`, retrained on 2026-11-10 with the selected config
  `ms_full` (`SELECTED_CONFIG` in `config.py`). `scripts/02_train.py` reproduces it bit for bit (seed 0, about 2.5
  minutes on 4 cores). Seeds 0-3 all pass the planning tests (`results/tables/seed_sensitivity.json`).
- **Planning uses a ten-seed ensemble** (`results/model/ensemble/`, `scripts/02b_train_ensemble.py`; seed 0 is a
  copy of the checkpoint). `04_plan.py` evaluates every pathway under each member and optimises plans against all
  of them (`plan/ensemble.py`); `plan.json` holds ensemble means and every seed's values. The seeds disagree
  widely: maximum-effort cut 26-59% (mean 41%), and the checkpoint is the most optimistic of the ten. A plan robust
  to every SSP and seed costs +119% effort. After any retrain of the checkpoint, rerun `02b_train_ensemble.py`:
  until then `04_plan.py` and `test_committed_ensemble_contains_the_checkpoint` refuse the stale ensemble.
- **Selection re-checked after a leak; it holds.** Every fit with a cutoff leaked post-cutoff drivers through the
  full-run loss's curvature term until `46cc5ce`; full-data fits (checkpoint, ensemble, tie-break checks) were never
  affected. The leak-free rerun in `results/sweep_val/report.json` (all 16 configs, 4000 steps, seeds 0-1; leaky
  report in history at `2648dda`) gives `ms_percap_damped` 10.79, `ms_percap` 10.96 (was 7th),
  `ms_full_shootonly` 11.29 (unaffected, identical), `ms_full` 11.38, `ms_notime_percap` 12.06; trend-Kaya 9.85.
  Multiple shooting still wins on every seed. The tie-break (planning checks with smooth projections on four seeds
  of full-data fits, `results/tables/selection/`, via `seed_sensitivity.py --config`) now covers all five; only
  `ms_full` passes on every seed. Round 1 (3000 steps; no `steps` fields, read as 3000 via `LEGACY_REPORT_STEPS`)
  is in `tests/data/report_round1_3000steps.json`.
- **The test folds are spent, and their tables predate the leak fix.** `results/sweep_test/report.json` (`ms_full`
  and `orig`, 5 seeds, realised and SSP-v9 drivers) was scored once, after selection, from leaky fits: `ms_full`
  9.40 against trend-Kaya 9.49 with realised drivers, 14.09 against growth persistence 11.15 ex ante; `orig` 19.96.
  `03_evaluate.py --config ms_full` wrote the out-of-sample tables from them (calibration: 0.859 coverage at 80%;
  transfer: MAPE 21% against trend-Kaya's 8%). Re-scoring the same two configs leak-free would correct these
  without revisiting the selection; it waits on the owner. Never use the test folds to revisit the selection.
- **The transfer test now trains its shared dynamics with `fit(country_w=...)`**, the deployed recipe (`0c93b5c`);
  it used plain Adam, which was unstable for `ms_full`. `spatial_transfer(..., end_year=...)` scores only up to
  `end_year`; a validation-period transfer (cutoff 2009, `end_year=2015`) gave 12.5-13.7% over three seeds
  against trend-Kaya's 9.0%.
- Sweep job outputs record their step count and full config; report entries record their step count. Scores
  never pool outputs trained differently (apart from the seed) or without that record. Older job caches lack
  it; delete and rerun them. Caches written before `46cc5ce` (2026-11-11) carry the same record but were
  trained with the leaky loss, which the guards cannot see: delete them too (caches are git-ignored, so this
  concerns machines that ran sweeps before then, such as a cluster).
- `06_sweep.py` merges into `<root>/report.json`, replacing only the configs passed; `--overwrite` replaces the
  whole report. Before fitting anything it refuses to put a second training length (or other folds) into a
  report, and to reuse cached outputs trained differently from what the run asks for. It does not check seeds:
  an ensemble of more seeds usually scores better, so compare configs only on the same seed set. Use `--root`
  for side runs.
- `default_config()` is `make_config(SELECTED_CONFIG)`, and every name in `eval/configs.py` is a set of overrides
  of `Config()`, the original model, so names keep their meaning whatever is selected. `02_train.py`, the tests,
  `03_backtest.py`, `04_plan.py` and `05_make_figures.py` all use `default_config()`, which is why `02_train.py`
  has no `--config` flag. To select another config, change `SELECTED_CONFIG` and retrain at once:
  `results/tables/config.json`, written by `02_train.py` with the checkpoint, no longer matches until then, so
  `test_committed_checkpoint_matches_default_config` fails and `03_backtest.py`, `04_plan.py` and
  `05_make_figures.py` refuse to run. Then retrain the ensemble (`02b_train_ensemble.py`) and regenerate the
  tables, the test folds for the new config, `03_evaluate.py --config <name>` and the figures.
  `03_evaluate.py --config X` refuses unless `config.json` matches `make_config(X)` and every sweep output it
  reads was trained for as many steps as the checkpoint. All of these scripts refuse before writing anything.
- Branch `claude/clever-fermi-rm4429` holds a separate Space Solar World Model (`src/space_solar_world_model/`).
  It split from this branch at `ed4c64f` and has not been merged.

## Where the next steps are

README "Limitations (read these)" and "What changed on 2026-11-12". First, with the owner's go-ahead: re-score
the test folds leak-free for `ms_full` and `orig` (`06_sweep.py --phase test --configs ms_full,orig --seeds
0,1,2,3,4 --ssp --overwrite` into a clean `results/sweep_test`, then `03_evaluate.py --config ms_full` and
`05_make_figures.py`). The open questions: forecasts only match trend-Kaya given the true drivers, the model
transfers worse than trend extrapolation, and training seeds of the selected config disagree on the achievable
cut (26-59%), so robust plans cost twice the effort.
