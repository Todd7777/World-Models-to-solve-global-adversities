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
python -m pytest -q   # 2026-11-12: 70 pass (python -m: a standalone `pytest` may use another interpreter)
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
  Fitted max effort: 712-1311 Gt (q10-q90) global fossil CO2 2023-2050.
- GDP/population driver error is carried in too (`eval/propagate.driver_error_vectors`, `perturb_drivers`,
  `scripts/10_driver_error.py`, `results/tables/planning_with_driver_error.json`). Joint per-country
  trend-rate errors come from naive trailing-10-year projections at the 11 long cutoffs (14-20 years) and
  from the 2014 SSPs over 2015-2022; the models are rerun under SSP2 with perturbed drivers. Combined with the
  emission drifts paired by cutoff (headline; independent as sensitivity): fitted max effort 714-1480 Gt
  (q10-q90), open loop 552-954 Gt; selected config fitted 692-976 Gt. The errors compounded
  historically (both ran low from 1997-2002), so pairing widens the upper tail more than independence.
  (The 1.5C/2C shares in 09/10 compare fossil CO2 2023-2050 with 50% budgets only; use 11 for verdicts.)
- Carbon budgets are now the published ones (`climate/budgets.py`: Forster et al. 2025 ESSD Table 8, from
  2025, rebased to 2023 with GCB 2024 totals 40.6 + 41.6 Gt; `tcre.remaining_budget_gtco2` returns the 50%
  value: 212 Gt for 1.5C, 1132 Gt for 2C). The earlier formula (target - 1.24)/TCRE ignored non-CO2 warming
  and gave 578 / 1689 Gt (`budgets.tcre_only_budget`). `scripts/11_budgets.py` turns every case into a
  likelihood of staying below 1.5/1.7/2C with full accounting: fossil 2023-2050 + land-use CO2 (4.2 Gt/yr
  falling to 0 in 2050) + a post-2050 tail falling linearly to net zero in 2070 (assumption), with
  sensitivities. Result (both errors, paired): 1.5C <=17% in every case; 2C at max effort median 43%
  (fitted) / 70% (open loop), BAU 18%. Headline: "1.5C out of reach, 2C at max effort undecided".
- Sweep caches (`results/sweep_*/*/*.npz`) are git-ignored; only `report.json` files are committed.
- Branch `claude/clever-fermi-rm4429` holds a separate Space Solar World Model (`src/space_solar_world_model/`).
  It split from this branch at `ed4c64f` and has not been merged.

## Where the next steps are

README "Limitations (read these)", especially the first: a way to constrain or test how the dynamics
respond to controls. Backtest error dominates the planning spread, but more cutoffs in 1997-2007 add little
independent information (one China regime shift); genuinely new information would need other regions'
shocks or a different validation design. The post-2050 tail is an assumption, and the 2C verdict moves
~10 points per decade of net-zero timing; a model run to net zero would remove it. Driver errors are
measured only from 1997-2022 naive projections and one SSP vintage. Untried: tuning w_stab beyond {10, 100}, and why shoot-only training transfers worse.
