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
> assumptions. Its limitations (below) are reported as prominently as its results, and
> the most important one is new: **the planning results depend more on unidentified
> model structure than on anything the data pin down** (see "Planning").

---

## Headline results

All evaluation is **leak-free**: every fit uses only data up to its cutoff year (emission
factors, normalisation and static features included), and every forecast starts from the
observed state at the cutoff, exactly as the baselines do. The model configuration was
chosen on **validation** folds only (cutoffs 2005/2007/2009, 6-year horizons ending
≤ 2015) and scored once on **test** folds (cutoffs 2015/2016/2017, 7-year horizons).
Score = per-country MAPE % + mean |aggregate error| %; lower is better.

### The physics layer and in-sample fit are faithful

- The physics layer alone reconstructs reported CO₂ with **R² = 0.9986** (median error
  4.7%), with emission factors recovered from data in the correct physical order:
  coal 0.316 > oil 0.231 > gas 0.198 MtCO₂/TWh.
- Running free from 1990 (given only GDP and population), the trained model reproduces
  33 years of history for all 60 countries: CO₂ **R² 0.9997** (MAPE 3.0%), primary energy
  R² 0.9994 (MAPE 4.3%), fuel mix MAE 0.8 points. In-sample fit is necessary, not
  evidence of forecasting skill.

### Forecasting: competitive, not dominant

Selected configuration `ms_percap_damped` (multiple-shooting training, per-capita levels,
EMA momentum, damped trend; [`eval/configs.py`](src/carbon_world_model/eval/configs.py)).
It had the lowest validation score of 16 configurations (11.05, against 21.20 for the
original model), **but the margin over the runner-up `ms` (11.37) is within seed noise**:
single seeds of the selected configuration score 11.5 and 13.3.

Test folds, mean over three cutoffs, five seeds:

| drivers | method | country MAPE | mean \|agg. err.\| | final-year agg. err. | China final-year | score |
|---|---|---|---|---|---|---|
| realised GDP/pop | **world model** | **6.84%** | 4.40% | −8.9% | −18.5% | 11.25 |
| | trend-Kaya | 7.83% | **1.67%** | −0.2% | −3.8% | **9.49** |
| | growth persistence | 8.84% | 2.31% | +0.7% | −8.4% | 11.15 |
| | original model (`orig`) | 13.95% | 6.00% | +3.4% | +3.0% | 19.96 |
| SSP2 (2014 vintage), ex-ante | **world model** | 8.87% | **1.72%** | −2.0% | −11.1% | **10.59** |
| | trend-Kaya | 10.96% | 6.52% | +9.6% | +6.5% | 17.48 |
| | growth persistence | 8.84% | 2.31% | +0.7% | −8.4% | 11.15 |

What this does and does not show:

- **Best per-country accuracy** of all methods with realised drivers, and a large
  improvement over the original model (score 19.96 → 11.25).
- **The simple trend-Kaya baseline still wins overall** with realised drivers, as it did on
  validation (9.85 vs 11.05). The world model undershoots the aggregate by ~9% after
  7 years.
- **The undershoot is China's energy demand**: China's final-year CO₂ error (−16% to −20%)
  is mostly demand (−12% to −17%), not carbon intensity (−2% to −5%)
  ([`china_decomposition.json`](results/tables/china_decomposition.json)). The model
  extrapolates China's historical decoupling too far, the same optimistic-decoupling bias
  seen in integrated-assessment models.
- **The ex-ante "win" is mostly error cancellation**: the 2014-vintage SSP GDP paths run
  above realised GDP and raise the model's aggregate by +7 to +9% (SSP2), which offsets
  its demand undershoot ([`driver_decomposition.json`](results/tables/driver_decomposition.json)).
  Do not read it as skill.
- **Uncertainty is calibrated**: split-conformal 80% intervals, calibrated on validation
  residuals, cover **82.9%** of test outcomes.
- **Transfer to unseen countries**: trained on 48 countries and adapting only each held-out
  country's small latent on pre-2016 data, CWM forecasts 12 unseen countries at R² 0.996
  (MAPE 9.3%). That is evidence of transferable structure, but trend-Kaya does better
  on the same countries (R² 0.999, MAPE 7.6%).

### Planning: one robust conclusion, wide structural uncertainty

Feasibility limits come from the fastest *sustained* historical transitions: low-carbon
share +1.8 points/yr, energy intensity −3.0%/yr. Two findings shape how every planning
number should be read.

**1. The learned dynamics react to the plan, and how strongly is not identified.** Controls
enter additively, but the network's own proposed change depends on the state, so a plan
that moves a country off its historical path changes what the network proposes next. The
**open-loop counterfactual** (same controls, natural changes pinned to business-as-usual)
removes that reaction. With the selected model under SSP2, maximum feasible effort cuts
2050 emissions **14%** below BAU with the fitted dynamics but **66%** in the open loop. The
fitted dynamics keep 46% of the clean-mix push and **8%** of the efficiency push
([`plan/diagnostics.py`](src/carbon_world_model/plan/diagnostics.py)). Historical data
show how energy systems evolve on their own, not how they would respond to pushes far
outside that range.

**2. Configurations tied on validation disagree beyond the validation horizon.** The
three configurations within 0.5 points of the best validation score (`ms_percap_damped`,
`ms`, `ms_full_shootonly`; two seeds each) are indistinguishable on 6-year forecasts, yet
under SSP2 ([`structural.json`](results/tables/structural.json), `scripts/07_structural.py`):

| quantity (SSP2, 60 countries) | range across the 6 tied models |
|---|---|
| business-as-usual 2050 emissions (2022: 34.8 Gt) | 31.9 – 56.7 Gt |
| China BAU change 2022→2035 | +12% – +48% |
| max-effort 2050 cut vs BAU, fitted dynamics | 12% – 46% |
| max-effort 2050 cut vs BAU, open loop | 64% – 66% |
| efficiency push retained by the fitted dynamics | 3% – 45% |
| warming avoided by max effort (TCRE central) | 0.06 – 0.14 °C fitted, 0.18 – 0.27 °C open loop |

The selected model's free-running BAU also shows a learned ~12-year oscillation in
China's energy intensity (annual change swinging between about −0.06 and +0.04, against a
steady −0.03 historically). That cycle is invisible over a 6-year validation horizon and
dominates a 28-year run.

**The conclusion that survives all of this:** scaled to global, cumulative 2023–2050 CO₂
under maximum feasible effort is **631 – 1,226 Gt** across all 12 cases (6 models × fitted
or open-loop dynamics). **Every case exceeds the ≈578 Gt remaining 1.5 °C budget** (by 9%
at the closest) and every case stays within the ≈1,689 Gt 2 °C budget. Within historically
achieved transition rates, holding 1.5 °C without negative emissions is out of reach in
this model. The 9% margin is smaller than the uncertainty in the budget itself, so read
this as "at best marginal", not as a precise verdict.

The model also produces a cost-of-abatement frontier, per-country allocations and a
**robust plan** optimised against all five SSPs at once (under the selected model's
fitted dynamics, robustness to every SSP doubles the required effort, and a plan tuned
to SSP2 alone overshoots the budget by 75 Gt under SSP5). These demonstrate the method;
their magnitudes inherit the structural uncertainty above.

---

## Figures

All in [`results/figures/`](results/figures), regenerated by `scripts/05_make_figures.py`.

| file | content |
|---|---|
| `fig1_reconstruction.png` | Physics layer: reconstructed vs reported CO₂; emission factors |
| `fig2_insample.png` | Free-running 1990–2022 fit (global + five countries) |
| `fig3_ablation_validation.png` | All 16 configurations on the validation folds |
| `fig4_test_and_china.png` | Test folds vs baselines; China final-year error by origin |
| `fig5_china_forecasts.png` | China forecasts from 2015: selected, original, trend-Kaya |
| `fig6_ssp_drivers.png` | SSP GDP/population drivers, 2014 and 2025 vintages |
| `fig7_ssp_pathways_robust.png` | BAU and max effort (fitted and open loop) per SSP; robust planning |
| `fig8_mix_ssp2.png` | Fuel-mix transition, BAU vs maximum feasible effort (SSP2) |
| `fig9_frontier_ssp2.png` | Cost-of-abatement frontier (SSP2) |
| `fig10_structural_ssp2.png` | Planning results across all validation-tied models vs carbon budgets |

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
(calendar-time features, absolute vs per-capita levels, momentum, damped trend) and
multiple-shooting training are described in
[`model/dynamics.py`](src/carbon_world_model/model/dynamics.py) and
[`train/fit.py`](src/carbon_world_model/train/fit.py). Everything is written in JAX and
trained end-to-end by differentiable simulation (`optax` AdamW, 4000 steps, about 4 min
on one CPU core).

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
| Training | `train/fit.py`, `train/losses.py` | fit across countries by differentiable simulation |
| Evaluation | `eval/` | rolling-origin protocol, baselines, transfer, conformal calibration |
| C: Planning | `plan/optimize.py`, `plan/scenario.py`, `plan/diagnostics.py` | optimal and robust control; open-loop counterfactual; control retention |
| D: Climate | `climate/tcre.py` | cumulative CO₂ → warming and carbon budgets |

---

## Reproduce

```bash
pip install -e ".[dev]"              # numpy, pandas, jax, optax, scipy, matplotlib
python -m pytest -q                  # 51 tests
```

The processed panel, SSP series, checkpoints and validation report are committed, so the
later steps run without downloading data. Full pipeline (about 2 h on 4 CPU cores;
`scripts/slurm/` runs the sweeps as a SLURM array):

```bash
bash scripts/download_data.sh        # OWID CO2 + energy data (CC-BY); only for step 01
python scripts/01_build_dataset.py   # 60-country panel
python scripts/06_sweep.py --phase val --seeds 0,1 --configs orig,notime,percap,ema,damped,ms,ms_notime,ms_notime_percap,ms_notime_percap_ema,ms_full,ms_full_shootonly,notime_percap_damped,ms_ema,ms_damped,ms_percap,ms_percap_damped
python scripts/02_train.py           # full-data model, selected config (eval/configs.SELECTED)
python scripts/06_sweep.py --phase test --configs ms_percap_damped,orig --seeds 0,1,2,3,4 --ssp
python scripts/03_evaluate.py --config ms_percap_damped
python scripts/04_plan.py            # BAU / max effort (fitted + open loop) / frontier / robust
python scripts/07_structural.py      # planning across all validation-tied models
python scripts/05_make_figures.py
```

---

## Limitations (read these)

- **Planning results are dominated by structural uncertainty.** Configurations that are
  indistinguishable on validation give 2050 BAU emissions from 32 to 57 Gt and
  max-effort cuts from 12% to 46% (fitted dynamics). How strongly the learned dynamics
  push back against a plan is not identified by historical data. Treat single-model
  planning numbers as illustrations of the method, and use the ranges.
- **Short-horizon validation does not validate long-horizon behaviour.** Selection used
  6-year horizons; planning runs 28 years. The selected model's free-running projection
  contains a learned oscillation (China's intensity) that no validation fold can see.
  Adding long-horizon plausibility or stability constraints to training is open work.
- **Short-horizon aggregate forecasting is not where this model wins.** Trend-Kaya beats it
  on aggregate CO₂ on both validation and test folds, and on transfer to unseen countries.
  The gap is driven by China's energy demand.
- **Drivers are exogenous.** GDP and population paths are assumptions, and results are
  conditional on them. Ex-ante test results with 2014-vintage SSPs look better than
  realised-driver results only because two errors cancel.
- **Feasibility bounds are judgement calls.** They come from the fastest *sustained*
  decade-long national rates, with caps to exclude petrostate GDP-volatility artefacts.
  Reasonable people would choose somewhat different numbers.
- **Territorial, production-based CO₂ only.** No consumption/trade accounting, no
  land-use CO₂, no non-CO₂ gases, no negative-emissions technologies, no electricity
  trade between countries.
- **Coverage is 60 countries (~93% of CO₂), not the whole world.** Global figures are
  scaled by that coverage, which is approximate.
- **Few seeds.** Validation used 2 seeds and test 5; the full-data and structural models
  use 1–2. Seed-to-seed variation is comparable to the gaps between top configurations.

---

## Data & license

Data: [Our World in Data](https://github.com/owid/co2-data) CO₂/GHG and
[energy](https://github.com/owid/energy-data) datasets, released under CC-BY. SSP
drivers: IIASA SSP database (v9, 2014; v3.2, 2025). Code in this repository is
MIT-licensed.
