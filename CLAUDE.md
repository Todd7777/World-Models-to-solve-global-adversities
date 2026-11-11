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
python -m pytest -q   # 2026-11-12: 63 pass (python -m: a standalone `pytest` may use another interpreter)
```

## State on 2026-11-12

- All tests pass. The checkpoint `results/model/params_full.pkl` is the selected config
  `ms_full_shootonly_stab100` (`eval/configs.SELECTED`, chosen on validation folds only, by a rule written
  into `eval/configs.py` before the round-3 sweep ran), and `params_full.json` records that. Scripts rebuild
  the model from that record (`train.fit.load_trained`). Retrain with `scripts/02_train.py` whenever the
  dynamics or training change; `tests/test_checkpoint.py` fails otherwise.
- Training has a stability term (`TrainConfig.w_stab`, `train/fit.stability_loss`): it penalises year-to-year
  changes of the natural rates on a 30-year free run beyond the cutoff, driven by probe drivers built from
  data up to the cutoff only. It removed the learned China oscillation and improved validation for all six
  configs it was tried on. On the test folds (realised drivers) the selected model now beats trend-Kaya
  (8.70 vs 9.49), but loses ex-ante to growth persistence and transfers worse (MAPE 12.9% vs 7.6%).
- Main open problem: planning still depends on structure the data do not identify. The stabilised tied
  configs give a fitted max-effort 2050 cut of 24-49% (open loop 66-68%), mostly from how strongly they
  resist an efficiency push (`plan/diagnostics.py`, `results/tables/structural.json`; the unconstrained set
  is in `structural_unconstrained.json`). Report ranges, not single-model planning numbers.
- Long-run level checks (diagnostics, never used for selection; `scripts/08_longrun.py`,
  `results/tables/longrun.json`): 14-20-year backtests (`06_sweep.py --phase longval|longtest`, 11 cutoffs
  1997-2007; seeds 0-3 for the two stabilised tied configs, 0-1 for the others) show mean absolute
  cumulative-CO2 errors of 5-8% for the selected model and 5-7% for trend-Kaya. Every world model
  undershoots from 1997-2002 cutoffs and overshoots from 2003-2007 (China's boom, then slowdown), so the 11
  overlapping cutoffs are close to one regime shift seen twice. Use `--aggregate-only` to rebuild a report
  that mixes seed counts. The energy-ladder check (`eval/plausibility.py`) finds
  no unprecedented projections; every model decouples more slowly than each country's own 1994-2022 past,
  the shoot-only configs least (selected: +0.3/+0.7%/yr by seed, vs +1.3/+1.4 for ms_percap_damped_stab10).
- Backtest error is carried into the planning ranges (`eval/propagate.py`, `scripts/09_backtest_error.py`,
  `results/tables/planning_with_backtest_error.json`): each long backtest gives a trend-rate error delta (44
  per config), and every planning path is corrected by exp(-delta*h). Headline ranges are the 10th-90th
  percentile over cases (decided before the expansion; envelopes widen with more draws), envelopes alongside.
  Maximum effort is below the 1.5C budget in 16.5% of open-loop cases and 0% with fitted dynamics; it stays
  within the 2C budget in 349/352 cases (exceptions: ms_percap_damped_stab10's two most negative drifts) and
  in all 176 of the selected config's. Headline: "2C robust (99%), 1.5C undecided".
- Sweep caches (`results/sweep_*/*/*.npz`) are git-ignored; only `report.json` files are committed.
- Branch `claude/clever-fermi-rm4429` holds a separate Space Solar World Model (`src/space_solar_world_model/`).
  It split from this branch at `ed4c64f` and has not been merged.

## Where the next steps are

README "Limitations (read these)", especially the first: a way to constrain or test how the dynamics
respond to controls. Backtest error dominates the planning spread, but more cutoffs in 1997-2007 add little
independent information (one China regime shift); genuinely new information would need other regions'
shocks or a different validation design. Driver (GDP/population) error beyond SSP1-5 and budget/TCRE
uncertainty are not yet propagated. Untried: tuning w_stab beyond {10, 100}, and why shoot-only training transfers worse.
