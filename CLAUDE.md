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
pytest             # 2026-11-14: 79 pass
```

If `pytest` cannot import numpy, a standalone pytest on PATH is shadowing the project's interpreter; run
`python -m pytest`.

## State on 2026-11-14

- **All 79 tests pass** with `results/model/params_full.pkl`, retrained on 2026-11-10 with the selected config
  `ms_full` (`SELECTED_CONFIG` in `config.py`). `scripts/02_train.py` reproduces it bit for bit (seed 0, about 2.5
  minutes on 4 cores). Seeds 0-3 all pass the planning tests (`results/tables/seed_sensitivity.json`).
- **Planning uses a ten-seed ensemble** (`results/model/ensemble/`, `scripts/02b_train_ensemble.py`; seed 0 is a
  copy of the checkpoint). `04_plan.py` evaluates every pathway under each member and optimises plans against all
  of them (`plan/ensemble.py`); `plan.json` holds ensemble means and every seed's values. At the frontier the
  maximum-effort cut is 61-70% (mean 67%, heuristic drivers); a plan robust to every SSP and seed costs +62%
  effort. After any retrain of the checkpoint, rerun `02b_train_ensemble.py`: until then `04_plan.py` and
  `test_committed_ensemble_contains_the_checkpoint` refuse the stale ensemble. Then rerun `04_plan.py` (about seven
  minutes), `control_response.py`, `plan_sensitivity.py`, `seed_sensitivity.py` and the planning figures.
- **The planner plans at the feasibility frontier (2026-11-13, owner's decision).** The network runs on business as
  usual only (`plan/optimize.py`: `make_worlds`, then `apply_plan`); a plan adds extra annual rates of intensity
  fall and low-carbon gain, capped per world (member x scenario) at the room below the limits, and the planned
  low-carbon share displaces BAU's fossil mix of the year until none is left (a running-minimum recursion, in
  closed form). The old closed loop, which fed controls into the network, gave cuts of 26-59% because seeds
  extrapolate differently off the data; `plan/response.py` keeps it only for the diagnostic
  (`scripts/control_response.py`, `results/tables/control_response.json`: closed loop 26-59%, open loop 79-84%,
  frontier 61-70%), which `test_committed_control_response_matches_the_ensemble` checks. Plans take 20,000 Adam
  steps with a cosine-decaying rate (`STEPS`): 1,000-1,500 were far from converged, quasi-Newton methods fail on
  the per-world caps, and the robust mode bounds a log-sum-exp (tau 0.5 Gt) because a penalty per world converges
  too slowly. The tie-break's planning checks now hold for any config by construction, so it no longer separates
  `ms_full` from `ms_percap_damped` (best on score; it failed only effort monotonicity, on one seed). Whether to
  revisit the selection is the owner's decision. `make_worlds(..., rebound=, crowd_out=)` takes back those shares
  of what a plan achieves, still charging its full effort (default 0: planned change persists).
  `scripts/plan_sensitivity.py` (`results/tables/plan_sensitivity.json`, checked against the ensemble and
  `plan.json` by `test_committed_plan_sensitivity_matches_the_ensemble_and_plan_json`) relaxes the assumptions for
  maximum effort: 67% falls to 63% with 25% rebound, 60% with 25% crowding out, 50% with the limits at x0.75, 40%
  with all three, 31% with the limits at x0.5.
- **Selection re-checked after a leak; it holds.** Every fit with a cutoff leaked post-cutoff drivers through the
  full-run loss's curvature term until `46cc5ce`; full-data fits (checkpoint, ensemble, tie-break checks) were never
  affected. The leak-free rerun in `results/sweep_val/report.json` (all 16 configs, 4000 steps, seeds 0-1; leaky
  report in history at `2648dda`) gives `ms_percap_damped` 10.79, `ms_percap` 10.96 (was 7th),
  `ms_full_shootonly` 11.29 (unaffected, identical), `ms_full` 11.38, `ms_notime_percap` 12.06; trend-Kaya 9.85.
  Multiple shooting still wins on every seed. The tie-break (planning checks with smooth projections on four seeds
  of full-data fits, `results/tables/selection/`, via `seed_sensitivity.py --config`) now covers all five; only
  `ms_full` passes on every seed, with the closed-loop planner (those tables are kept as the record). Round 1 (3000 steps; no `steps` fields, read as 3000 via `LEGACY_REPORT_STEPS`)
  is in `tests/data/report_round1_3000steps.json`.
- **The test folds are spent.** They were scored after selection, for `ms_full` and `orig` only (5 seeds, realised
  and SSP-v9 drivers): first from leaky fits on 2026-11-10, then once more, leak-free, on 2026-11-12 with the
  owner's go-ahead, as a correction. `results/sweep_test/report.json` and the tables now hold the leak-free run:
  `ms_full` 8.33 against trend-Kaya 9.49 with realised drivers (leaky: 9.40), 13.68 against growth persistence
  11.15 ex ante (leaky: 14.09); `orig` 18.84. `03_evaluate.py --config ms_full` wrote the out-of-sample tables
  (calibration: 0.872 coverage at 80%; transfer: MAPE 11.9% over the five test-fold seeds, 9.1-15.4%, against
  trend-Kaya's 7.6%) and `05_make_figures.py` the figures. Never use the test folds to revisit the selection.
- **The transfer test now trains its shared dynamics with `fit(country_w=...)`**, the deployed recipe (`0c93b5c`);
  it used plain Adam, which was unstable for `ms_full`. `spatial_transfer(..., end_year=...)` scores only up to
  `end_year`; a validation-period transfer (cutoff 2009, `end_year=2015`) gave 12.5-13.7% over three seeds
  against trend-Kaya's 9.0%. `03_evaluate.py` runs the transfer test for every test-fold seed (owner-approved on
  2026-11-12; seed 0 alone gave 15.4%, the worst of the five). `transfer.json` also records the test-fold model's
  error on the same 12 countries (trained on all 60): 7.9% per seed against transfer's 11.9%, a penalty of about 4
  points.
  `scripts/transfer_diagnostics.py` (validation period only, about 15 minutes with 4 workers; needs the val sweep
  outputs) finds only about 1 point there (12.6% transfer, 11.6% in-training, trend-Kaya 9.6%). Most of the
  validation gap is a bias of the cutoff-2005-2009 models: they stall 12 mature economies' efficiency trend
  (forecast -0.8 to +0.6 %/yr against -1.8 to -2.5 realised), through weak damped momentum plus single-step
  swings in the network's correction (`scripts/stall_mechanism.py`). Configurations with calendar-time inputs
  avoid it only by extrapolating calendar time, so they do not fix it (`scripts/stall_by_config.py`,
  `scripts/time_inputs.py`; see "Where the next steps are"). The deployed ensemble's BAU mostly keeps that trend
  (-1.9 against -2.2 %/yr).
  Choosing a new country's starting latent (random or the training mean) changed nothing beyond adaptation noise,
  which is about 1 point.
- Sweep job outputs record their step count and full config; report entries record their step count. Scores
  never pool outputs trained differently (apart from the seed) or without that record. Older job caches lack
  it; delete and rerun them. Caches written before `46cc5ce` (2026-11-11) carry the same record but were
  trained with the leaky loss, which the guards cannot see: delete them too (caches are git-ignored, so this
  concerns machines that ran sweeps before then, such as a cluster). `05_make_figures.py` draws fig5 from the
  `results/sweep_test` job outputs: where the leak-free ones are absent (a fresh clone, or only older caches),
  it drops or misdraws the world-model lines, so keep the committed fig5 (`git checkout` it) instead of committing
  a redrawn one.
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

README "Limitations (read these)" and "What changed on" 2026-11-14, 2026-11-13 and 2026-11-12. The open
questions there:
- Forecasts are only level with trend-Kaya given the true drivers (ahead on the test folds, behind on the
  validation folds) and behind driver-free baselines ex ante.
- Transfer to unseen countries is worse than trend extrapolation (11.9% against 7.6% over five seeds). Transfer
  itself costs about 4 points on the test period and 1 on the validation period, on top of the model's general
  forecast error, and single transfer seeds differ by up to 6.4 points. A fresh holdout year from a newer OWID
  release (the panel ends in 2022) would give a check that no decision has touched.
- The cutoff-2005-2009 models stall mature economies' efficiency trend. Calendar-time inputs do not fix it, though
  in backtests they seem to. The first item reads existing outputs. Every prediction tested on new runs (seeds 2-3,
  calendar time held in backtests, planning BAU) was written down before the runs, and all held.
  - `scripts/stall_by_config.py` (reads validation outputs, fits nothing): all 7 configs without time inputs
    stall (mature trend -0.2 to -1.1 %/yr, realised -2.1). All 9 with them keep the trend (-1.5 to -2.7;
    `results/tables/stall_by_config.json`, seeds 0-1). Over seeds 0-3 (`results/sweep_val_4seeds/`,
    `results/tables/stall_by_config_4seeds.json`), `ms_percap_damped` forecasts -1.9 %/yr against `ms_full`'s
    -0.6 and scores 10.70 against 11.54.
  - That needs calendar time (`index / (T_ref - 1)`, 1 in 2022) fed past the cutoff: backtests feed it up to
    1.3-1.4x its trained maximum, while planning holds it at 2022 (`Scenario.tfrac`). The sweep never sets
    `forecast(tfrac_cap=...)`.
  - `scripts/time_inputs.py` (`results/tables/time_inputs.json`, seeds 0-3) holds calendar time at the cutoff;
    uncapped, its refits reproduce the sweep's outputs bit for bit. Held, `ms_percap_damped` stalls like
    `ms_full` (-0.7 against -0.6 %/yr) and scores 12.57 against 11.54.
  - In planning (full-data fits, heuristic drivers), its BAU intensity falls only 0.5 %/yr in the mature
    economies and 0.7 elsewhere over 2022-2032 (`ms_full` 2.0 and 1.1; observed 2.2 and 1.3). Its BAU reaches
    55.8 Gt in 2050 against 39.9. With calendar time running on to 1.875, the trend returns but BAU falls to
    28.0 Gt.
  - The other time-input configs' validation scores carry the same caveat. `ms_percap` (10.96) has not been
    scored with calendar time held.

  The mechanism (`scripts/stall_mechanism.py`, `results/tables/stall_mechanism.json`): ms_full refits at
  2005/2007/2009, seeds 0-1, each reproducing the sweep bit for bit.
  - The network's correction for the mature economies is no steady push. It swings by +2.6 to +6.4 %/yr in a
    single step:
    - at 2008->2009 for the 2005 and 2007 models, driven mostly by GDP growth. They treat the recession as if
      energy use were sticky; in fact it fell faster than GDP (-6.9 against -4.4%);
    - at 2010->2011 for the 2009 models, driven by the share-momentum inputs (biofuel, other renewables).
      Those inputs are divided by per-fuel scales that sit at the 1e-3 floor.
  - Each swing enters the momentum, the model's only trend memory. It falls from -2.5 to -3.0 %/yr at the cutoff
    to -0.2 to -1.2 six years later.
  - The damping learned from little data is strong: phi_I 0.58 at 2005, 0.69-0.70 at 2007, 0.78 at 2009, against
    0.86-0.89 on full data. It is data-driven: weight decay is negligible, and Adam could have moved it further.
  - Holding the share-momentum inputs at their observed values recovers 37, 52 and 81% of the missing trend at
    2005, 2007 and 2009.

  Six fixes within `ms_full` were probed on 2026-11-13 and -14 in scratch, not committed, on the validation folds:
  - A longer momentum memory (`ema_decay` 0.9 or 0.95) worsened the score (12.70 and 14.12, against 11.38) and
    left the trend stalled.
  - Damping the intensity momentum toward each country's trailing 10-year trend (trend-Kaya's window) instead of
    toward zero improved the score by 0.55 points (10.83), less than seed noise, and restored only about a fifth
    of the missing trend. Adding the fuel-mix shares to it did not help (11.05).
  - Dropping the static features (seeds 0-3; rule written down first: more than 1 point better, at least half the
    missing trend restored, neither MAPE nor |agg| worse) scored 10.90 against 11.54 and restored 17% of the
    missing trend; MAPE worsened (9.26% against 9.10%), |agg| improved (1.64% against 2.44%). It fails.
  - One common share-momentum input scale for all fuels (the pooled typical share change, about 0.013; seeds 0-3,
    same rule) scored 10.97 and recovered none of the missing trend (-1%). MAPE worsened (9.48%), |agg|
    improved (1.49%). The swings return through other inputs. It fails.
  - Decoupled momentum fails: it scored 14.67, recovered 22%, and worsened both MAPE (10.52%) and |agg|
    (4.15%). Both momenta were updated only from their own damped part, so network corrections could not enter
    the trend memory. The momentum feedback carries real information.
  - A slow-state network fails on the stall: it recovered 8% of the missing trend. Its network did not see
    contemporaneous GDP growth or the share momenta (both zeroed in `_features`; seeds 0-3, same rule).
    - In the trained models, holding those inputs had recovered 66/93/82% (growth alone 32/60/-4%). Retrained,
      the network swings by +3 to +5 %/yr at 2009-2010 through the level inputs instead: energy and GDP per
      capita, and intensity momentum.
    - It still scored 10.42 against 11.54, better on all four seeds, with MAPE (8.84%) and |agg| (1.59%) both
      better. It is the only variant to clear the 1-point bar without calendar-time inputs.
    - Its full-data fits pass the planning checks on all four seeds (`seed_sensitivity.py`'s checks, patched in
      scratch). In-sample MAPE is 3.8-3.9% against 3.2-3.3%. Heuristic BAU 2050 is 37.1-39.4 Gt against
      38.6-42.1, and max effort 12.5-13.3 against 12.3-13.2. BAU changes direction 2-4 times.

  None is adopted. The early-cutoff networks respond steeply to whichever input a shock moves (growth, levels or
  share momenta), and removing inputs only reroutes the swing. Two options remain:
  - limiting the network's sensitivity: stronger regularisation of its weights, or a tighter bound on its
    correction than the 8 %/yr on total change;
  - more data, since the stall fades with later cutoffs.
  Each needs four seeds and a rule written down before the runs. Any adoption is a re-selection for the owner.
- Training seeds of the selected config disagreed on the achievable cut (26-59%) under the old closed-loop
  planner; at the frontier it is 61-70%, and a plan robust to every SSP and seed costs +62% effort. The
  frontier's assumptions are worth a lot (`scripts/plan_sensitivity.py`): moderate rebound and crowding out (25%
  each) with the limits at three quarters lower the 67% cut to 40%, and the limits matter most. Not yet done: the
  cost frontier and robust plans under relaxed assumptions, and evidence on what rebound, crowding out and
  sustainable limits are.
- Whether to revisit the selection now that the tie-break no longer separates `ms_full` from `ms_percap_damped`
  is the owner's decision; the test folds are spent. The two differ only in the calendar-time inputs, and
  `ms_percap_damped`'s lead on score needs calendar time fed past the cutoff. Held at the cutoff, as planning holds
  it, the config scores 12.57 against 11.54, and its planning BAU stalls (above). The slow-state network (above)
  scores 10.42 without calendar-time inputs and passes the planning checks. But it failed its own rule, and the
  validation folds have now served many probes. It is not a registered config: adding a `ModelConfig` field
  changes every recorded config, which the provenance guards compare.
