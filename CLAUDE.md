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
python -m pytest -q   # 2026-11-14: 82 pass (python -m: a standalone `pytest` may use another interpreter)
```

## State on 2026-11-14

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
  configs give a fitted max-effort 2050 cut of 27-53% (open loop 69-70%), mostly from how strongly they
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
  Fitted max effort: 694-1282 Gt (q10-q90) global fossil CO2 2023-2050.
- GDP/population driver error is carried in too (`eval/propagate.driver_error_vectors`, `perturb_drivers`,
  `scripts/10_driver_error.py`, `results/tables/planning_with_driver_error.json`). Joint per-country
  trend-rate errors come from naive trailing-10-year projections at the 11 long cutoffs (14-20 years) and
  from the 2014 SSPs over 2015-2022; the models are rerun under SSP2 with perturbed drivers. Combined with the
  emission drifts paired by cutoff (headline; independent as sensitivity): fitted max effort 698-1450 Gt
  (q10-q90), open loop 537-925 Gt; selected config fitted 674-951 Gt. The errors compounded
  historically (both ran low from 1997-2002), so pairing widens the upper tail more than independence.
  (The 1.5C/2C shares in 09/10 compare fossil CO2 2023-2050 with 50% budgets only; use 11 for verdicts.)
- Carbon budgets are now the published ones (`climate/budgets.py`: Forster et al. 2025 ESSD Table 8, from
  2025, rebased to 2023 with GCB 2024 totals 40.6 + 41.6 Gt; `tcre.remaining_budget_gtco2` returns the 50%
  value: 212 Gt for 1.5C, 1132 Gt for 2C). The earlier formula (target - 1.24)/TCRE ignored non-CO2 warming
  and gave 578 / 1689 Gt (`budgets.tcre_only_budget`). `scripts/11_budgets.py` turns every case into a
  likelihood of staying below 1.5/1.7/2C with full accounting: fossil 2023-2050 + land-use CO2 (4.2 Gt/yr
  falling to 0 in 2050) + a post-2050 tail falling linearly to net zero in 2070 (assumption), with
  sensitivities. Result (both errors, paired): 1.5C <=17% in every case; 2C at max effort median 46%
  (fitted) / 73% (open loop), BAU 18%.
- Runs to 2100 (`scripts/12_to_2100.py`, `results/tables/to_2100.json`) replace the assumed tail with the
  model: errors frozen at 2050, total = fossil 2023-2100 + land use (a lower bound, since no case reaches zero).
  No max-effort case reaches net zero: fossil shares stall at 20-40% (learned pushback in never-seen
  near-zero-fossil states; in the open loop, pinned BAU changes keep adding oil/gas, an artefact).
  2C at max effort falls to <=21% (fitted) / <=56% (open loop) from 46% / 73% with the assumed tail.
  Headline: "1.5C out of reach; 2C at max effort ~20-75% depending on post-2050 dynamics".
- Process CO2 lever (2026-11-13): max effort and the optimisers cut process CO2 (cement, flaring; 6.5% of
  panel CO2, constant under BAU) by up to p_max = 2.7%/yr compounding (`plan/scenario.process_bound`: median
  best decade-long decline, capped at 3.5%/yr; a weak proxy, since historical declines are mostly demand
  collapses). Controls carry `u_proc`; `optimize._abated_process` applies it to the process driver. It added
  3 points to the 2C likelihoods with the assumed tail, 3-11 points to 2100. Optimiser cost weights it like h.
- Technology-based process bound (`build_scenario(..., process_lever="technology")`,
  `plan/scenario.technology_process_floor`): max effort follows published roadmaps instead: cement GCCA
  Concrete Future (2021: -20% per t cement by 2030 vs 2020, net zero by 2050) and World Bank Zero Routine
  Flaring by 2030, applied to each country's 2022 component split (`data/processed/process_components.json`,
  from OWID via `scripts/01c_process_components.py`). Assumptions: no 2020-22 progress, 1% residual, 5%
  non-routine flaring, other industry follows cement. Max-effort runs only (rates explode near 2050). In
  `scripts/12_to_2100.py` it replaces the old 2070 phase-out sensitivity. Result: only 1-5 points on 2C at max
  effort vs the historical lever; process CO2 is not the binding constraint, residual fossil energy is.
- Negative emissions (`climate/cdr.py`, in `scripts/12_to_2100.py`): max feasible NOVEL CDR, logistic from
  0.002 Gt/yr in 2025 at 40%/yr to a ceiling of 3.5 Gt/yr (State of CDR 3rd ed., June 2026: measured level,
  growth and the Paris-scenario 2050 level); ceilings 1.75 / 7 as sensitivities. Conventional CDR is already
  in the land-use term. Budgets are compared with PEAK cumulative CO2 (lower bound if net CO2 > 0 in 2100).
  Central CDR adds 6-17 points for 2C at max effort (fitted 46->52% tail, 21->27% to 2100; open loop 73->80%,
  56->73%, 70% of runs reach net zero, median 2084) and nothing for 1.5C (peak >= ~550 Gt vs 402 Gt for 17%):
  removals arrive after the 1.5C budget is spent.
- Land and cost limits on removals (`climate/cdr.py`: `cdr_supply`, `cdr_ceiling`, `CDR_VARIANTS`): bottom-up
  ceiling from AR6 WGIII Ch12 Table 12.6 (DACCS, BECCS, biochar, enhanced weathering; potential midpoints spread
  evenly over cost ranges), a price cap (USD 100/200/300 per t) and a biomass land cap on BECCS+biochar (2.4
  Gt/yr = AR6 C3 BECCS 2050; 1.1/5.3 sensitivities). USD 100/t + 2.4 gives 3.2 Gt/yr and reproduces the
  scenario-level results within 1-2 points; the land cap matters only at low prices (2-5 points). At USD
  200/300 DACCS dominates (ceiling 17/28 Gt/yr), 2C at max effort becomes likely, but implies 45-225 EJ/yr and
  USD 2-5 tn/yr; spending is reported, not limited. 1.5C stays <=17% in every case.
- DACCS energy limit (`cdr.energy_limited_cdr_path`, ENERGY_VARIANTS in `scripts/12_to_2100.py`): 7 EJ per
  GtCO2 (AR6 WGIII Ch12: 40-100 EJ/yr at 10 Gt), DACCS energy <= f x the run's own low-carbon primary energy
  (f = 10%, 5/20% sensitivities; choices, not sourced); only the DACCS share is cut. At USD 200/t, fitted
  dynamics to 2100: 2C median 64% -> 42/50/62% (5/10/20%), net zero by 2100 62% -> 30/48/62%; tail 69 ->
  60/64/69%. USD 300 + 10% = USD 200 + 10% (energy binds). Open loop stays at the table max (83%) but its
  q10 falls 55 -> 33-52%; its post-2050 energy limit is an artefact (low-carbon energy collapses under the
  pinned -3%/yr efficiency push). 1.5C unchanged. Cases record lowcarbon_ej_2050/2100 and daccs_2050/2100.
- The clean-energy push now follows the CURRENT fossil mix (`dynamics.rollout` u_clean/clean_w/fossil_mask,
  `optimize._push`); the old 2022-mix split wasted the push once a fuel ran out. Effect through 2050: a few Gt.
- Sweep caches (`results/sweep_*/*/*.npz`) are git-ignored; only `report.json` files are committed.
- Branch `claude/clever-fermi-rm4429` holds a separate Space Solar World Model (`src/space_solar_world_model/`).
  It split from this branch at `ed4c64f` and has not been merged.

## Where the next steps are

README "Limitations (read these)", especially the first: a way to constrain or test how the dynamics
respond to controls. Backtest error dominates the planning spread, but more cutoffs in 1997-2007 add little
independent information (one China regime shift); genuinely new information would need other regions'
shocks or a different validation design. Post-2050 behaviour is now the largest open question: the
assumed tail and the model run to 2100 differ by 15-25 points for 2C. The model cannot represent
near-zero-fossil states (out of distribution), treats negative emissions only as an accounting lever
(land, cost and DACCS-energy limited, but spending, storage and the growth rate unbounded; the energy
fraction f is a labelled choice, and in the open loop post-2050 low-carbon energy collapses, so its energy
limit after 2050 is an artefact), and the
stability term only covers 30 years of the uncontrolled run. Process CO2 now has both a historical and a
roadmap-based bound, and the choice barely matters (1-5 points). Driver errors are
measured only from 1997-2022 naive projections and one SSP vintage. Untried: tuning w_stab beyond {10, 100}, and why shoot-only training transfers worse.
