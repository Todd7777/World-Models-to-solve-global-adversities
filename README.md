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
> backtest errors are carried into the planning ranges (new), **this model can no longer
> rule 1.5 °C in or out**: under maximum feasible effort, it is missed in 56 of 64 cases and
> met in 8. What survives every case is that maximum effort stays within the 2 °C budget.

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
feasible effort keeps cumulative emissions within the ≈1,689 Gt 2 °C budget.

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
realised drivers: long validation, cutoffs 1998/2000 → 2015 (15-17 years); long test,
cutoffs 2003/2005 → 2022 (17-19 years, overlapping the test period, report only). Means
over the two cutoffs, two seeds:

| method | cumulative CO₂ error, long val / long test | \|final-year agg. error\| | China final-year error | country MAPE |
|---|---|---|---|---|
| **world model, selected** (`ms_full_shootonly_stab100`) | −4.2% / +9.8% | 6.4% / 25.2% | −36% / +49% | **14.6% / 16.9%** |
| `ms_percap_damped_stab10` (tied on validation) | −14.8% / +21.7% | 27.5% / 41.9% | −52% / +46% | 16.3% / 24.9% |
| `ms_percap_damped` (previous selection, unconstrained) | −25.9% / +11.2% | 51.6% / 24.7% | −63% / −20% | 24.1% / 23.7% |
| `orig` | −9.5% / **+4.7%** | 20.3% / **7.5%** | −47% / **−2%** | 17.2% / 19.0% |
| trend-Kaya | **+2.6%** / +6.6% | 8.1% / 15.6% | −16% / +38% | 23.8% / 18.9% |
| growth persistence | −5.7% / +22.1% | **2.3%** / 60.0% | −49% / +111% | 22.2% / 27.2% |

- **No method gets 15-19-year levels right.** The windows straddle China's 2000s boom
  (missed from 1998/2000) and its post-2013 slowdown (overshot from 2003/2005), and every
  method's China error is tens of percent.
- **The selected model has the best per-country accuracy at both lengths**, and its
  cumulative error (−4%, +10%) is comparable to trend-Kaya's (+3%, +7%). It is not the best
  on aggregate levels: `orig`, the worst model on 6-year validation, is the best on the
  long test.
- **The check would have caught the oscillating model**: the previous selection misses the
  2015 aggregate by 52% from 1998/2000. Its stabilised variant halves that, but still does
  much worse than the selected configuration.
- **Implication for planning:** cumulative errors of ±10% over two decades are the realistic
  precision here. The next section carries them into the planning ranges.

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
log level with horizon, i.e. a wrong trend rate. A line through the origin fits 3 of the
selected configuration's 4 seed-averaged backtests (R² 0.71–0.92; from 1998 the error
barely drifts) and all 4 of `ms_percap_damped_stab10`'s (R² 0.82–0.97). Each backtest therefore gives one drift δ
(log error per year), and every planning path is corrected as path × e^(−δh), h years after
2022 ([`eval/propagate.py`](src/carbon_world_model/eval/propagate.py),
`scripts/09_backtest_error.py`,
[`planning_with_backtest_error.json`](results/tables/planning_with_backtest_error.json),
`fig12_planning_with_backtest_error.png`).

- **Drifts used:** each configuration's own 8 backtests (4 cutoffs × 2 seeds). For the
  selected configuration δ ranges from −1.1 to +1.6 %/yr; for `ms_percap_damped_stab10`, from
  −2.6 to +3.9 %/yr.
- **Assumptions:** the same trend-rate error applies to BAU and controlled paths (it is an
  error in the natural dynamics). In the main case the error keeps growing to 2050, 9-13 years
  beyond the longest backtest; a sensitivity case caps it at 19 years.
- **Not a confidence interval:** 8 backtests per configuration are a record of errors the
  model has actually made, all from cutoffs straddling China's boom and slowdown.

Global cumulative CO₂ 2023–2050 (Gt), stabilised tied models, SSP2:

| | business-as-usual | max effort, fitted dynamics | max effort, open loop |
|---|---|---|---|
| model as fitted | 1,139 – 1,259 | 797 – 1,026 | 647 – 706 |
| + backtest error, growing to 2050 | 725 – 1,912 | 581 – 1,535 | 443 – 1,006 |
| + backtest error, capped at 19 years | 762 – 1,806 | 605 – 1,457 | 455 – 967 |
| selected model, SSP1–5 × its own errors | 887 – 1,410 | 636 – 967 | 528 – 769 |

What changes:

- **1.5 °C is no longer ruled out.** Of the 64 max-effort cases (4 models × 2 dynamics × 8
  drifts), 8 fall below the ≈578 Gt budget (7 with the cap). All 8 combine no learned
  resistance to the plan (open loop) with a drift under which the model ran high in its
  backtests. With the fitted dynamics none falls below, but the lowest case (581 Gt) sits
  on the budget line.
- **2 °C survives every case.** Maximum feasible effort stays within the ≈1,689 Gt 2 °C
  budget in all 64 cases (highest 1,535 Gt). Business-as-usual does not: its highest
  cases exceed the 2 °C budget.
- **Backtest error now dominates the planning spread.** The fitted-dynamics max-effort range
  grows from 797–1,026 to 581–1,535 Gt (about four times as wide), far more than the
  structural disagreement between tied models.
- **Still missing:** error in the GDP and population drivers beyond the five SSPs, and
  the uncertainty in the budgets and the TCRE themselves.

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
python -m pytest -q                  # 62 tests
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
python scripts/06_sweep.py --phase longval  --seeds 0,1 --configs ms_full_shootonly_stab100,ms_percap_damped_stab10,ms_full_shootonly,ms_percap_damped,orig
python scripts/06_sweep.py --phase longtest --seeds 0,1 --configs ms_full_shootonly_stab100,ms_percap_damped_stab10,ms_full_shootonly,ms_percap_damped,orig
python scripts/08_longrun.py         # long-run level checks -> longrun.json
python scripts/09_backtest_error.py  # planning ranges with backtest error carried in
python scripts/05_make_figures.py
```

---

## Limitations (read these)

- **Planning results are uncertain in two ways the data cannot resolve.** First, structure:
  stabilised configurations tied on validation give 2050 BAU emissions from 39 to 48 Gt and
  max-effort cuts from 24% to 49% (fitted dynamics), and how strongly the dynamics push back
  against a plan is not identified. Second, forecast error: carrying the model's own
  long-horizon backtest errors into the ranges makes the fitted max-effort range about four
  times as wide, so 1.5 °C can be neither ruled in nor out. Use the ranges, not single-model planning numbers.
- **Long-run levels are checked, but only weakly.** Over 15-19-year backtests no method,
  including this model, gets aggregate levels right; cumulative-CO₂ errors of ±10% are
  realistic. The backtests cover only four cutoffs, all straddling China's boom and
  slowdown, and use realised drivers (so they understate error for 2050 projections, whose
  drivers are also uncertain). Carrying them to 2050 assumes the error keeps growing at the
  same trend rate for 9-13 more years than any backtest covers. The energy-ladder check confirms projections are not unprecedented, but its
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
  conditional on them.
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
