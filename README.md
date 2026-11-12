# Carbon World Model (CWM)

A differentiable, mechanistic-neural **world model of the global energy–emissions
system**, calibrated to real data for 60 countries (≈93% of global CO₂), built to
ask a question ordinary forecasts cannot:

> *Given how national energy systems actually evolve, what is the cheapest feasible
> path to deep decarbonisation, and how much warming does it avoid?*

A "world model" in the modern sense is a learned simulator of an environment that an
agent can **plan inside**. Here the environment is the national energy-economy-emissions
system, the agent is a policymaker choosing where to deploy clean energy and efficiency,
and planning is trajectory optimisation. Because the entire simulator is differentiable,
pathways are found **by gradient descent through the model itself**.

CWM couples three things that are usually kept apart:

1. **Physics**: CO₂ reconstructed from per-fuel energy use via emission factors
   (a layer the model cannot distort).
2. **Learning**: a neural model of how demand and the 9-fuel mix evolve, shared across
   countries with per-country latents, trained by differentiable simulation.
3. **Planning**: optimal control of clean-deployment and efficiency effort, bounded by
   the fastest transitions humanity has actually sustained.

> **Status / honesty note.** This is a transparent research prototype, not a forecast of
> the future and not policy advice. Its results are conditional on documented
> assumptions, and its limitations (below) are reported as prominently as its results.
> The most important: **planning results still depend on model structure the data do not
> pin down.** A stability term in training removes learned oscillations and narrows that
> spread, but does not remove it (see "Planning"). Once the model's own long-horizon
> backtest errors and the historical errors of GDP and population projections are carried
> into the planning ranges, **this model can no longer rule 1.5 °C in or out**: under
> maximum feasible effort it is met in 19% of cases without a learned reaction to the plan,
> and in none with the fitted dynamics. Maximum effort stays within the 2 °C budget in 98%
> of cases, and in every case for the selected configuration.

---

## Headline results

All evaluation is **leak-free**: every fit uses only data up to its cutoff year (emission
factors, normalisation, static features and the stability term's probe drivers included),
and every forecast starts from the observed state at the cutoff, exactly as the baselines
do. Configurations are chosen on **validation** folds only (cutoffs 2005/2007/2009, 6-year
horizons ending ≤ 2015) and scored once on **test** folds (cutoffs 2015/2016/2017, 7-year
horizons). Score = per-country MAPE % + mean |aggregate error| %; lower is better.

### The physics layer and in-sample fit are faithful

- The physics layer alone reconstructs reported CO₂ with **R² = 0.9986** (median error
  4.7%), with emission factors recovered from data in the correct physical order:
  coal 0.316 > oil 0.231 > gas 0.198 MtCO₂/TWh.
- Running free from 1990 (given only GDP and population), the trained model reproduces
  33 years of history for all 60 countries: CO₂ **R² 0.9989** (MAPE 4.8%), primary energy
  R² 0.9986 (MAPE 7.1%), fuel mix MAE 1.1 points. The selected configuration is trained
  only on 8-year windows started from observed states, so it fits the long free run
  slightly less closely than earlier versions (CO₂ MAPE 3.0%). In-sample fit is
  necessary, not evidence of forecasting skill.

### Forecasting: best per-country accuracy, and now level with or ahead of the baselines

Selected configuration `ms_full_shootonly_stab100`: multiple-shooting training only,
per-capita levels, EMA momentum, damped trend, and the **stability term** at weight 100
([`eval/configs.py`](src/carbon_world_model/eval/configs.py)). It had the lowest
validation score of 22 configurations (9.85, level with trend-Kaya's 9.85), against 11.43
for the same configuration without the stability term and 21.20 for the original model.
22 configurations compared on 2 seeds make the best validation score optimistic, so the
test folds are the real check.

Test folds, mean over three cutoffs, five seeds:

| drivers | method | country MAPE | mean \|agg. err.\| | final-year agg. err. | China final-year | score |
|---|---|---|---|---|---|---|
| realised GDP/pop | **world model (stabilised)** | **6.86%** | 1.84% | −3.9% | −11.2% | **8.70** |
| | world model, previous selection (`ms_percap_damped`) | 6.84% | 4.40% | −8.9% | −18.5% | 11.25 |
| | trend-Kaya | 7.83% | **1.67%** | −0.2% | −3.8% | 9.49 |
| | growth persistence | 8.84% | 2.31% | +0.7% | −8.4% | 11.15 |
| | original model (`orig`) | 13.95% | 6.00% | +3.4% | +3.0% | 19.96 |
| SSP2 (2014 vintage), ex-ante | world model (stabilised) | 9.69% | 2.12% | +1.0% | −7.4% | 11.81 |
| | trend-Kaya | 10.96% | 6.52% | +9.6% | +6.5% | 17.48 |
| | growth persistence | **8.84%** | 2.31% | +0.7% | −8.4% | **11.15** |

What this does and does not show:

- **With realised drivers the world model now has the best score on held-out data**
  (8.70 vs 9.49 for trend-Kaya) and the best per-country accuracy. Trend-Kaya still has a
  slightly smaller aggregate error (1.67% vs 1.84%), and the margin rests on one test
  period (2015–2022, including 2020).
- **The China undershoot shrank but remains**: final-year China error −6% to −15% by
  cutoff, now split between demand (−3% to −10%) and carbon intensity (−3% to −7%)
  ([`china_decomposition.json`](results/tables/china_decomposition.json)). The model
  still extrapolates some of China's historical decoupling too far.
- **Ex-ante, growth persistence wins** (11.15 vs 11.81). The 2014-vintage SSP GDP paths
  run above realised GDP and raise the model's final-year aggregate by about +5% (SSP2;
  +4% to +8% across SSPs) ([`driver_decomposition.json`](results/tables/driver_decomposition.json)). With the
  previous selection that shift happened to offset a larger undershoot; now it does not.
- **Uncertainty is calibrated**: split-conformal 80% intervals, calibrated on validation
  residuals, cover **83.2%** of test outcomes.
- **Transfer to unseen countries got worse.** Trained on 48 countries and adapting only
  each held-out country's latent on pre-2016 data, the model forecasts 12 unseen
  countries at R² 0.990 (MAPE 12.9%), against MAPE 9.3% for the previous selection and
  7.6% for trend-Kaya.

### Planning: one robust conclusion (2 °C), wide uncertainty about 1.5 °C

Feasibility limits come from the fastest *sustained* historical transitions: low-carbon
share +1.8 points/yr, energy intensity −3.0%/yr.

**The learned dynamics react to the plan.** Controls enter additively, but the network's
own proposed change depends on the state, so a plan that moves a country off its
historical path changes what the network proposes next. The **open-loop counterfactual**
(same controls, natural changes pinned to business-as-usual) removes that reaction.
With the selected model under SSP2, maximum feasible effort cuts 2050 emissions **49%**
below BAU (39.2 → 19.9 Gt) with the fitted dynamics and **67%** in the open loop. The
fitted dynamics keep 95% of the clean-mix push and 47% of the efficiency push
([`plan/diagnostics.py`](src/carbon_world_model/plan/diagnostics.py)). Historical data
show how energy systems evolve on their own, not how they would respond to pushes far
outside that range.

**The stability term.** Without it, the previous selection's free run had a learned
~12-year oscillation in China's energy intensity (annual change swinging between −0.07
and +0.05, against a steady −0.03 historically), invisible over a 6-year validation
horizon but dominant over 28 years. The term penalises year-to-year changes in the
model's natural rates on a 30-year free run beyond the training cutoff. It does not
say where emissions should go ([`train/fit.py`](src/carbon_world_model/train/fit.py),
`stability_loss`). It reduces rate-acceleration under SSP2 inputs about 18-fold (30.9 →
1.7 × 10⁻³ for the selected configuration's parent), costs no in-sample fit, and improved
every one of the six configurations it was tried on, on validation. Under SSP2 the
selected model projects China's energy intensity falling 1.9%/yr to 2030 and 1.2%/yr
after, slower than the 2.9%/yr of 2012–2022. That is the model's own extrapolation,
not something the term imposes.

**Structural spread.** The configurations within 0.5 validation points of the best in
their family, two seeds each, under SSP2
([`structural.json`](results/tables/structural.json),
[`structural_unconstrained.json`](results/tables/structural_unconstrained.json),
`scripts/07_structural.py`):

| quantity (SSP2, 60 countries) | stabilised: 4 tied models | unconstrained: 6 tied models |
|---|---|---|
| business-as-usual 2050 emissions (2022: 34.8 Gt) | 39.2 – 48.4 Gt | 31.9 – 56.7 Gt |
| China BAU change 2022→2035 | +19% – +29% | +12% – +48% |
| max-effort 2050 cut vs BAU, fitted dynamics | 24% – 49% | 12% – 46% |
| max-effort 2050 cut vs BAU, open loop | 66% – 68% | 64% – 66% |
| efficiency push retained by the fitted dynamics | 12% – 47% | 3% – 45% |
| warming avoided by max effort (TCRE central) | 0.09 – 0.14 °C fitted, 0.20 – 0.23 °C open loop | 0.06 – 0.14 °C, 0.18 – 0.27 °C |

The range of 2050 BAU emissions shrinks from 25 to 9 Gt and that of China's 2035 change from
36 to 10 points, though part of that comes from comparing 4 models with 6. The two
stabilised configurations still disagree on how strongly the system pushes back against
an efficiency drive, which drives the remaining spread in the fitted-dynamics cut. They
also disagree on long-run energy levels: the energy-ladder check below shows that
`ms_percap_damped_stab10` slows decoupling by about 1.4%/yr relative to each country's own
past (energy-weighted, and positive for every major economy), against about 0.3-0.7%/yr
for the selected configuration. The BAU gap is entirely a difference in energy demand:
the second configuration uses 30% more energy in 2050 with a 5% cleaner mix, hence 24%
more CO₂.

**Carbon budgets, model as fitted.** Scaled to global, cumulative 2023–2050 CO₂ under
maximum feasible effort is 647 – 1,026 Gt across the 8 stabilised cases (4 models × fitted
or open-loop dynamics), and 631 – 1,226 Gt across the 12 unconstrained cases. Taken at face
value, every case exceeds the ≈578 Gt remaining 1.5 °C budget, by 12% at the closest.
**That verdict does not survive the model's own forecast errors** (see "Planning ranges
with backtest error carried in" below). The conclusion that does survive is that maximum
feasible effort keeps cumulative emissions within the ≈1,689 Gt 2 °C budget: in 98% of
cases once both model and driver errors are included, and in all of the selected
configuration's.

The model also produces a cost-of-abatement frontier, per-country allocations and a
**robust plan** optimised against all five SSPs at once. Under the selected model's
fitted dynamics, meeting a budget in every SSP costs 25% more effort than a plan tuned
to SSP2 alone, and that SSP2-only plan overshoots by 27 Gt under SSP5. These demonstrate
the method; their magnitudes inherit the structural uncertainty above.

### Long-run level checks

The validation and test folds use 6-7-year horizons, and the stability term only checks
smoothness. These two checks ask whether long-run *levels* hold up
([`longrun.json`](results/tables/longrun.json), `scripts/08_longrun.py`,
`fig11_longrun_checks.png`). Both are diagnostics, fixed in advance as not used for
model selection.

**1. Long-horizon backtests.** Fit at an early cutoff and forecast to a fixed end year with
realised drivers: long validation, cutoffs 1997-2001 → 2015 (14-18 years); long test,
cutoffs 2002-2007 → 2022 (15-20 years, overlapping the test period, report only). Seeds
0-3 for the two stabilised tied configurations, 0-1 for the others. Neighbouring cutoffs
share most of their forecast years, so 11 cutoffs are far from 11 independent tests.
Means over cutoffs, long validation / long test:

| method | mean \|cumulative CO₂ error\| | signed cumulative error [min … max over cutoffs] | \|final-year agg. error\| | China final-year error | country MAPE |
|---|---|---|---|---|---|
| **world model, selected** (`ms_full_shootonly_stab100`) | 5.1% / 8.2% | −5.1% [−7.4 … −2.2] / +5.3% [−8.5 … +13.7] | **6.4%** / 16.5% | −36% / +19% | **13.3% / 15.2%** |
| `ms_percap_damped_stab10` (tied on validation) | 13.7% / 12.8% | −13.7% [−18.4 … −9.2] / +11.2% [−5.0 … +31.1] | 24.7% / 25.0% | −50% / +24% | 15.5% / 19.6% |
| `ms_percap_damped` (previous selection, unconstrained) | 27.0% / 9.0% | −27.0% [−38.9 … −15.8] / +4.9% [−12.1 … +21.3] | 50.7% / 18.0% | −64% / **−13%** | 22.6% / 20.3% |
| `orig` | 8.6% / 9.1% | −6.8% [−10.8 … +4.4] / +6.4% [−8.0 … +18.7] | 17.7% / **14.5%** | −43% / +19% | 19.1% / 19.6% |
| trend-Kaya | **4.7% / 7.2%** | +3.5% [−2.6 … +10.9] / +6.8% [−1.2 … +11.5] | 10.3% / 16.9% | **−14%** / +45% | 25.7% / 18.1% |
| growth persistence | 6.5% / 24.4% | −4.2% [−10.4 … +5.7] / +23.0% [−4.2 … +39.6] | 7.3% / 61.8% | −42% / +114% | 24.8% / 28.6% |

- **No method gets 14-20-year levels right.** The windows straddle China's 2000s boom and
  its post-2013 slowdown, and every method's China error is tens of percent. The world
  models undershoot cumulative CO₂ from every 1997-2002 cutoff (one exception: `orig` from
  1997) and overshoot from every 2003-2007 cutoff: the errors are dominated by one regime
  shift, not independent noise.
- **The selected model has the best per-country accuracy at both lengths**, and its mean
  absolute cumulative error (5.1%, 8.2%) is close to trend-Kaya's (4.7%, 7.2%), the best on
  that measure.
- **Two cutoffs per phase were too few.** With the earlier 1998/2000 and 2003/2005 cutoffs,
  the selected model's long-test final-year error looked much worse than trend-Kaya's (25%
  vs 16%) and `orig` looked best; with six long-test cutoffs they are level (16.5% vs 16.9%).
- **The check would have caught the oscillating model**: the previous selection misses
  cumulative CO₂ by 27% on average from 1997-2001. Its stabilised variant halves that, but
  still does much worse than the selected configuration.
- **Implication for planning:** cumulative errors of about ±10% over two decades are the
  realistic precision here. The next section carries them into the planning ranges.

**2. Energy-ladder plausibility.** Across countries, energy per person rises with income
along a stable, concave "ladder" (log energy/cap quadratic in log GDP/cap; R² 0.75 on all
observed 1990-2022 country-years). The check measures how a 2022→2050 projection moves
each country relative to that ladder ([`eval/plausibility.py`](src/carbon_world_model/eval/plausibility.py)).
Because the ladder accounts for income, a country that grows more slowly in the
projection is expected to decouple more slowly.

- **Nothing is unprecedented.** No model moves any country (by energy) outside the central
  95% of all observed 28-year drifts. That band is wide, since it includes post-Soviet
  transitions and petrostates, so this only rules out the absurd.
- **Against each country's own past, every model decouples more slowly than history.**
  Compared with each country's own 1994→2022 drift off the ladder, energy-weighted:

  | SSP2 BAU 2050 | slower decoupling than own past |
  |---|---|
  | selected (`ms_full_shootonly_stab100`), seeds 0/1 | +0.33 / +0.65 %/yr |
  | `ms_percap_damped_stab10`, seeds 0/1 | +1.42 / +1.30 %/yr |
  | unconstrained tied models | +0.37 to +1.65 %/yr |
  | reference: own 2012-2022 intensity trend continued | +0.95 %/yr |
  | reference: constant energy intensity | +1.96 %/yr |

  The selected model stays closest to history; the other stabilised configuration is
  closer to freezing intensity. History does not say which is right, since decoupling can
  slow (saturating efficiency gains) or speed up (structural change). The reference
  window matters too. The selected model's China intensity falls 1.4%/yr on average to
  2050: slower than 2012-2022 (−2.9%/yr), but faster than 1994-2022 (−1.15%/yr), a period
  that includes the energy-intensive 2002-2011 boom. Measured on the ladder, which allows
  for China moving up in income, it decouples 0.5%/yr more slowly than its own 1994-2022
  past. Russia shows the largest excess in every model (+2 to +3%/yr), partly because its
  own 1994→2022 reference includes post-Soviet restructuring.

### Planning ranges with backtest error carried in

The long-horizon backtests show the model's aggregate CO₂ error growing roughly linearly in
log level with horizon, i.e. a wrong trend rate. A line through the origin fits 10 of the
selected configuration's 11 seed-averaged backtests (R² 0.60–0.94) and 9 of 11 for
`ms_percap_damped_stab10` (R² 0.71–0.97). Each backtest therefore gives one drift δ (log
error per year), and every planning path is corrected as path × e^(−δh), h years after 2022
([`eval/propagate.py`](src/carbon_world_model/eval/propagate.py),
`scripts/09_backtest_error.py`,
[`planning_with_backtest_error.json`](results/tables/planning_with_backtest_error.json),
`fig12_planning_with_backtest_error.png`).

- **Drifts used:** each configuration's own 44 backtests (11 cutoffs × 4 seeds). Selected
  configuration: mean +0.04 %/yr (essentially unbiased), 10th–90th percentile −0.87 to
  +1.08 %/yr, extremes −1.33 to +1.69. `ms_percap_damped_stab10`: mean −0.27, 10th–90th
  percentile −2.38 to +2.00, extremes −3.62 to +3.90. Cutoffs from 1997-2002 give negative
  drifts and 2003-2007 positive ones, so the near-zero mean comes from two eras
  cancelling.
- **Summary statistic:** the headline ranges are the 10th–90th percentile over all cases
  (model × dynamics × drift), with the full envelope alongside. This was decided before the
  expanded backtests ran, because a min–max envelope widens automatically as draws are added.
- **Assumptions:** the same trend-rate error applies to BAU and controlled paths (it is an
  error in the natural dynamics). In the main case the error keeps growing to 2050, 8-14
  years beyond the backtests; a sensitivity case caps it at 20 years. The earliest backtest
  models are fitted on only 8-10 years of data, so their errors probably overstate the
  full-data model's.
- **Not a confidence interval:** these are errors the model has actually made, from
  overlapping windows that all straddle China's boom and slowdown.

Global cumulative CO₂ 2023–2050 (Gt), stabilised tied models, SSP2:

| | business-as-usual | max effort, fitted dynamics | max effort, open loop |
|---|---|---|---|
| model as fitted (all 4 models) | 1,139 – 1,259 | 797 – 1,026 | 647 – 706 |
| + backtest error to 2050, 10th–90th percentile | 942 – 1,668 | 711 – 1,311 | 553 – 886 |
| + backtest error to 2050, all cases | 725 – 2,283 | 581 – 1,823 | 443 – 1,172 |
| + backtest error capped at 20 years, 10th–90th percentile | 964 – 1,618 | 717 – 1,276 | 561 – 869 |
| previous sample (8 backtests), to 2050, 10th–90th percentile | 930 – 1,798 | 669 – 1,373 | 547 – 936 |
| selected model, SSP1–5 × its 44 errors, 10th–90th percentile [all] | 973 – 1,329 [879 – 1,459] | 692 – 916 [632 – 998] | 566 – 735 [524 – 791] |

What this shows:

- **1.5 °C stays undecided.** Without a learned reaction to the plan, maximum effort falls
  below the ≈578 Gt budget in 16.5% of cases (25% with the previous 8 backtests); with the
  fitted dynamics in none, though the lowest case (581 Gt) sits on the budget line. For the
  selected model alone across SSP1–5 the shares are 16% and 0%.
- **2 °C holds in 99% of cases.** Maximum effort stays within the ≈1,689 Gt budget in 349 of
  352 cases (351 with the cap). All exceptions are `ms_percap_damped_stab10` with fitted
  dynamics and its two most negative drifts (−3.3 and −3.6 %/yr, from 1997 and 1999 backtest
  models fitted on 8-10 years of data). None of the selected configuration's 176 cases
  exceeds it. Business-as-usual exceeds the 2 °C budget in 10% of cases.
- **More backtests narrowed the central range and widened the extremes.** The 10th–90th
  percentile range for maximum effort with fitted dynamics went from 669–1,373 to 711–1,311 Gt;
  the envelope grew, as envelopes do with more draws.
- **Backtest error still dominates the planning spread**, well beyond the structural
  disagreement between tied models (797–1,026 Gt as fitted).
- **Not yet included here:** error in the GDP and population drivers (next section), and the
  uncertainty in the budgets and the TCRE themselves.

### Planning ranges with GDP and population driver error carried in

The planning runs use SSP drivers. The spread across SSP1–5 is a range of scenarios, not
forecast error: each SSP is itself a projection that will be wrong. This section measures
how wrong such projections have been and reruns the models with the drivers perturbed
accordingly ([`eval/propagate.py`](src/carbon_world_model/eval/propagate.py),
`scripts/10_driver_error.py`,
[`planning_with_driver_error.json`](results/tables/planning_with_driver_error.json),
`fig13_planning_by_error_source.png`).

**Measured driver errors**, as per-country trend-rate errors in GDP and population (the
same form as the emission errors). Each historical projection gives one joint vector over
all 60 countries, which keeps the cross-country correlation that matters for the global
total.

- **The 2014-vintage SSPs (SSP1–5)**, harmonised at 2014 and checked against realised
  2015–2022 data. They over-projected aggregate GDP growth by 1.2–1.6 %/yr (SSP2: 1.3 %/yr,
  China 1.6 %/yr), partly because of COVID. Population errors were small (most countries
  within ±0.7 %/yr).
- **Naive projections from the long backtest cutoffs 1997–2007:** each country's trailing
  10-year growth continued for 14–20 years. Their aggregate GDP error ranges from −0.3 to
  +1.6 %/yr, with the same era pattern as the emission errors (too low from the 1990s, too
  high from the mid-2000s). China's growth was always under-projected (−1.9 to 0 %/yr) and
  the US always over-projected (+0.6 to +1.9 %/yr).
- **Why naive projections stand in for SSPs:** over the one window where both can be
  checked (2015–2022), the naive trend's aggregate GDP error (+1.50 %/yr) was close to,
  and slightly larger than, the 2014 SSP2's (+1.31 %/yr). No SSP vintage can yet be checked
  over 14–20 years. The naive errors are used unscaled, which leans conservative.

**Combining with the model's own error.** In log terms, a forecast's total error is
exactly its error with realised drivers plus the effect of the driver error. So each
perturbed run is combined with the same configuration's emission drifts:

- **paired** (headline): each driver vector from cutoff c with the emission drifts from
  that same cutoff, keeping the errors the two actually had together;
- **independent** (sensitivity): every driver vector with every emission drift.

Both choices were fixed before the runs.

Global cumulative CO₂ 2023–2050 (Gt), stabilised tied models, SSP2, 10th–90th percentile
[all cases]:

| | business-as-usual | max effort, fitted dynamics | max effort, open loop |
|---|---|---|---|
| model as fitted | 1,148 – 1,255 | 806 – 1,008 | 651 – 701 |
| + emission backtest error (previous section) | 942 – 1,668 | 711 – 1,311 | 553 – 886 |
| + driver error only, naive projections | 1,147 – 1,387 | 813 – 1,128 | 648 – 746 |
| + driver error only, 2014 SSPs | 1,040 – 1,133 | 731 – 899 | 599 – 650 |
| **+ both, paired by cutoff** | **943 – 1,880** [738 – 2,628] | **713 – 1,482** [611 – 2,103] | **550 – 955** [446 – 1,280] |
| + both, independent | 965 – 1,767 [663 – 2,649] | 721 – 1,409 [535 – 2,104] | 561 – 916 [411 – 1,291] |
| selected configuration only, both, paired | — | 692 – 978 [≤ 1,032] | 564 – 773 |

What this shows:

- **Driver error alone is smaller than the model's own error**, and its sign depends on
  the era. The 1990s naive projections under-projected growth, so correcting for them raises
  emissions; the 2014 SSPs over-projected growth, so correcting for them lowers emissions.
- **Historically the two errors compounded.** From the 1997–2002 cutoffs, both the GDP
  projections and the model ran low (China's boom). Pairing them therefore widens the upper
  end more than treating them as independent: the paired 90th percentile for fitted max
  effort is 1,482 Gt, against 1,409 Gt independent.
- **1.5 °C stays undecided.** Maximum effort falls below the ≈578 Gt budget in 18.8% of
  open-loop cases and none with the fitted dynamics (12.5% and 0% for the selected
  configuration).
- **2 °C now holds in 98% of cases.** Maximum effort exceeds the ≈1,689 Gt budget in 8 of
  352 paired cases (2.3%; 1.2% independent). All 8 are `ms_percap_damped_stab10` with fitted
  dynamics from the 1997–1999 cutoffs, whose backtest models were fitted on 8–10 years of
  data. The selected configuration never exceeds it (at most 1,032 Gt). Business-as-usual
  exceeds the 2 °C budget in 17% of paired cases.
- **Still missing:** uncertainty in the carbon budgets and the TCRE themselves, and driver
  errors from any period other than 1997–2022.

---

## Figures

All in [`results/figures/`](results/figures), regenerated by `scripts/05_make_figures.py`.

| file | content |
|---|---|
| `fig1_reconstruction.png` | Physics layer: reconstructed vs reported CO₂; emission factors |
| `fig2_insample.png` | Free-running 1990–2022 fit (global + five countries) |
| `fig3_ablation_validation.png` | All 22 configurations on the validation folds |
| `fig4_test_and_china.png` | Test folds vs baselines; China final-year error by origin |
| `fig5_china_forecasts.png` | China forecasts from 2015: selected, original, trend-Kaya |
| `fig6_ssp_drivers.png` | SSP GDP/population drivers, 2014 and 2025 vintages |
| `fig7_ssp_pathways_robust.png` | BAU and max effort (fitted and open loop) per SSP; robust planning |
| `fig8_mix_ssp2.png` | Fuel-mix transition, BAU vs maximum feasible effort (SSP2) |
| `fig9_frontier_ssp2.png` | Cost-of-abatement frontier (SSP2) |
| `fig10_structural_ssp2.png` | Planning results across the stabilised validation-tied models vs carbon budgets |
| `fig11_longrun_checks.png` | Long-horizon backtests (cumulative CO₂ error); energy-ladder decoupling vs own past |
| `fig12_planning_with_backtest_error.png` | Planning ranges with and without backtest error, against the 1.5 °C and 2 °C budgets |
| `fig13_planning_by_error_source.png` | Planning ranges by error source: model, GDP/population drivers, both |

---

## Method

The energy system of each country is described by two state variables per year:

```
I_t = log( primary_energy_t / GDP_t )     # log energy intensity (a Kaya factor)
s_t = fuel-mix shares on the 9-fuel simplex
```

Demand is reconstructed as `D_t = GDP_t · exp(I_t)` and emissions by the fixed physics
layer. GDP and population are **exogenous drivers** (observed in training, scenario
inputs in planning), the same framing integrated-assessment models use with SSP
pathways. A shared SiLU MLP with per-country embeddings predicts the **bounded** annual
change in intensity and mix (bounds encode capital-stock inertia); a learned per-country
emission-factor correction absorbs refining/petrostate residuals. Structure switches
(calendar-time features, absolute vs per-capita levels, momentum, damped trend) are
described in [`model/dynamics.py`](src/carbon_world_model/model/dynamics.py). Everything
is written in JAX and trained end-to-end by differentiable simulation (`optax` AdamW, 4000
steps, about 4 min on one CPU core).

Training objectives ([`train/fit.py`](src/carbon_world_model/train/fit.py)): a long free
run from 1990, multiple shooting (8-year runs started from the observed state in every
year), and the stability term. The stability term rolls every country 30 years beyond the
training cutoff from its observed cutoff state. The probe drivers grow per-capita GDP and
population at their trailing 10-year rates (clipped), tapering to 1.5%/yr and zero. It
penalises the mean squared year-to-year change in the model's natural rates of intensity
and mix change. A test checks that the probe drivers are unchanged when every value
after the cutoff is scrambled.

Checkpoints record the configuration that produced them (`results/model/*.json`), and
every script rebuilds the model from that record; a test fails if the committed
checkpoint does not match the selected configuration.

Planning controls (clean-deployment effort and efficiency effort) enter the dynamics
additively and are zero during fitting. Emissions response to effort is **monotonic**
(verified in the test suite). Warming is obtained from cumulative emissions via the
TCRE relationship (IPCC AR6 central value 0.45 °C per 1000 GtCO₂, likely range
0.27–0.63).

| module | file | what it does |
|---|---|---|
| A: Physics | `model/emissions.py` | CO₂ = Σ EFₖ·energyₖ + process; factors fit by NNLS, held fixed |
| B: Dynamics | `model/dynamics.py`, `model/world_model.py` | learned differentiable rollout of demand + mix |
| Training | `train/fit.py`, `train/losses.py` | full run, multiple shooting, stability term |
| Evaluation | `eval/` | rolling-origin protocol, baselines, transfer, conformal calibration |
| C: Planning | `plan/optimize.py`, `plan/scenario.py`, `plan/diagnostics.py` | optimal and robust control; open-loop counterfactual; control retention |
| D: Climate | `climate/tcre.py` | cumulative CO₂ → warming and carbon budgets |

---

## Reproduce

```bash
pip install -e ".[dev]"              # numpy, pandas, jax, optax, scipy, matplotlib
python -m pytest -q                  # 66 tests
```

The processed panel, SSP series, checkpoints and validation report are committed, so the
later steps run without downloading data. Full pipeline (about 3.5 h on 4 CPU cores;
`scripts/slurm/` runs the sweeps as a SLURM array):

```bash
bash scripts/download_data.sh        # OWID CO2 + energy data (CC-BY); only for step 01
python scripts/01_build_dataset.py   # 60-country panel
python scripts/06_sweep.py --phase val --seeds 0,1 --configs orig,notime,percap,ema,damped,ms,ms_notime,ms_notime_percap,ms_notime_percap_ema,ms_full,ms_full_shootonly,notime_percap_damped,ms_ema,ms_damped,ms_percap,ms_percap_damped,ms_percap_damped_stab10,ms_percap_damped_stab100,ms_stab10,ms_stab100,ms_full_shootonly_stab10,ms_full_shootonly_stab100
python scripts/02_train.py           # full-data model, selected config (eval/configs.SELECTED)
python scripts/06_sweep.py --phase test --configs ms_full_shootonly_stab100,ms_percap_damped,orig --seeds 0,1,2,3,4 --ssp
python scripts/03_evaluate.py --config ms_full_shootonly_stab100
python scripts/04_plan.py            # BAU / max effort (fitted + open loop) / frontier / robust
python scripts/07_structural.py                         # stabilised tied set -> structural.json
python scripts/07_structural.py --family unconstrained  # -> structural_unconstrained.json
for ph in longval longtest; do
  python scripts/06_sweep.py --phase $ph --seeds 0,1,2,3 --configs ms_full_shootonly_stab100,ms_percap_damped_stab10
  python scripts/06_sweep.py --phase $ph --seeds 0,1 --configs ms_full_shootonly,ms_percap_damped,orig
  python scripts/06_sweep.py --phase $ph --seeds 0,1,2,3 --aggregate-only --configs ms_full_shootonly_stab100,ms_percap_damped_stab10,ms_full_shootonly,ms_percap_damped,orig
done
python scripts/08_longrun.py         # long-run level checks -> longrun.json
python scripts/09_backtest_error.py  # planning ranges with backtest error carried in
python scripts/10_driver_error.py    # ... and with GDP/population driver error
python scripts/05_make_figures.py
```

---

## Limitations (read these)

- **Planning results are uncertain in two ways the data cannot resolve.** First, structure:
  stabilised configurations tied on validation give 2050 BAU emissions from 39 to 48 Gt and
  max-effort cuts from 24% to 49% (fitted dynamics), and how strongly the dynamics push back
  against a plan is not identified. Second, forecast error: carrying the model's own
  long-horizon backtest errors and the historical errors of GDP and population projections
  into the ranges widens the fitted max-effort range from 806–1,008 to 713–1,482 Gt (10th–90th
  percentile), so 1.5 °C can be neither ruled in nor out. Use the ranges, not single-model
  planning numbers.
- **Long-run levels are checked, but only weakly.** Over 14-20-year backtests no method,
  including this model, gets aggregate levels right; cumulative-CO₂ errors of about ±10% are
  realistic. The 11 cutoffs overlap heavily and all straddle China's boom and slowdown, so
  they amount to little more than one regime shift seen from two sides. They use realised
  drivers, so they understate error for 2050 projections, whose drivers are also uncertain.
  Carrying them to 2050 assumes the error keeps growing at the same trend rate for 8-14 more
  years than any backtest covers. The energy-ladder check confirms projections are not unprecedented, but its
  own-past reference is a comparison, not a truth: whether decoupling will slow, as every
  model projects, is not something historical data can settle.
- **The forecasting advantage is narrow.** The stabilised model beats trend-Kaya on the
  test folds with realised drivers (8.70 vs 9.49), but loses ex-ante to growth persistence,
  transfers to unseen countries worse than trend-Kaya, and the test period is a single
  7-year window that includes 2020.
- **Selection is optimistic.** 22 configurations were compared on 2 seeds; the stability
  weights (10, 100) were not tuned further. Seed-to-seed variation is comparable to the
  gaps between top configurations.
- **Drivers are exogenous.** GDP and population paths are assumptions, and results are
  conditional on them. Their forecast error is now propagated, but measured only from
  naive projections in 1997–2022 and from one SSP vintage over 8 years (including COVID),
  and extrapolated from 14–20-year windows to 28 years.
- **Feasibility bounds are judgement calls.** They come from the fastest *sustained*
  decade-long national rates, with caps to exclude petrostate GDP-volatility artefacts.
  Reasonable people would choose somewhat different numbers.
- **Territorial, production-based CO₂ only.** No consumption/trade accounting, no
  land-use CO₂, no non-CO₂ gases, no negative-emissions technologies, no electricity
  trade between countries.
- **Coverage is 60 countries (~93% of CO₂), not the whole world.** Global figures are
  scaled by that coverage, which is approximate.

---

## Data & license

Data: [Our World in Data](https://github.com/owid/co2-data) CO₂/GHG and
[energy](https://github.com/owid/energy-data) datasets, released under CC-BY. SSP
drivers: IIASA SSP database (v9, 2014; v3.2, 2025). Code in this repository is
MIT-licensed.
