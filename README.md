# Carbon World Model (CWM)

A differentiable, mechanistic-neural **world model of the global energy–emissions
system**, calibrated to real data for 60 countries (≈93% of global CO₂), built to
answer a question ordinary forecasts cannot:

> *Given how national energy systems actually evolve, what is the cheapest feasible
> path to deep decarbonisation — and how much warming does it avoid?*

A "world model" in the modern sense is a learned simulator of an environment that an
agent can **plan inside**. Here the environment is the national energy-economy-emissions
system, the agent is a policymaker choosing where to deploy clean energy and efficiency,
and planning is trajectory optimisation. Because the entire simulator is differentiable,
the cost-optimal decarbonisation pathway is found **by gradient descent through the
model itself**, not by enumerating scenarios.

CWM couples three things that are usually kept apart:

1. **Physics** — CO₂ reconstructed from per-fuel energy use via emission factors
   (a layer the model cannot distort).
2. **Learning** — a neural model of how demand and the 9-fuel mix evolve, shared across
   countries with per-country latents, trained by differentiable simulation.
3. **Planning** — optimal control of clean-deployment and efficiency effort, bounded by
   the fastest transitions humanity has actually sustained.

> **Status / honesty note.** This is a transparent research prototype, not a forecast of
> the future and not policy advice. Its results are conditional on documented
> assumptions, and its limitations (below) are reported as prominently as its strengths.

> **Current state (2026-11-10).** Model selection is still pending: all 16 candidate
> configurations have been scored on the validation folds at the training length the
> model is deployed with, but the leaders cannot yet be told apart on two seeds. The
> committed checkpoint is the default
> configuration, retrained after commit `3fc2559` changed the dynamics. The
> out-of-sample results this README used to report leaked post-cutoff data and are
> withdrawn; leak-free replacements will follow selection. See
> [What changed on 2026-11-09](#what-changed-on-2026-11-09).

---

## Headline results

**The simulator reproduces history.** Running free (autoregressively from 1990, given
only GDP and population paths), it reconstructs 33 years of history for all 60 countries:

| quantity | R² | MAPE |
|---|---|---|
| CO₂ emissions | 0.9982 | 5.3% |
| primary energy demand | 0.9964 | 6.5% |
| fuel-mix shares | — | 1.8 pts (MAE) |

The physics layer alone reconstructs reported CO₂ with **R² = 0.9986** (median error
4.7%), with emission factors recovered from data in the correct physical order:
coal 0.316 > oil 0.231 > gas 0.198 MtCO₂/TWh.

**Out-of-sample accuracy is not established yet, and so far it trails trend
extrapolation.** The leak-free protocol keeps model selection (validation cutoffs 2005,
2007 and 2009, six-year horizons) apart from reporting (test cutoffs 2015–2017, scored
once, after selection). On the validation folds, given realised GDP and population, a
trend-Kaya extrapolation has the lowest per-country error of every method tried (MAPE
8.0%). Of the 16 world-model configurations, trained for 4000 steps, the lowest is 8.7%
(`ms_percap_damped`: multiple shooting, per-capita levels, damped trend), and the
default configuration reaches 11.1% (see Limitations). Test-fold accuracy, calibrated
intervals and transfer to unseen countries will be reported after selection.

**It plans.** Optimising inside the simulator, under feasibility limits taken from the
fastest *sustained* historical transitions (low-carbon share +1.8%/yr, à la Denmark;
energy intensity −3.0%/yr, à la China), with the same heuristic GDP and population paths
as earlier versions of this README:

- **Business-as-usual** emissions for these 60 countries grow from 34.8 Gt (2022) to
  **47.4 Gt by 2050** (46.9 Gt with SSP2 drivers; 42–56 Gt across SSP1–5).
- **Maximum feasible effort** bends that down to **25.4 Gt** (SSP2: 24.4 Gt), a 46% cut
  versus BAU and 27% below 2022.
- Scaled to global, cumulative 2023–2050 emissions are **≈959 GtCO₂ even under maximum
  ambition** (SSP2: ≈941), which **blows through the ≈578 Gt remaining 1.5 °C budget**.
  Within historically-achieved transition rates, **1.5 °C is out of reach** without
  negative emissions; maximum effort avoids ≈0.17 °C (0.10–0.24 °C) of warming versus
  BAU for this group.
- The model produces the full **cost-of-abatement frontier** (minimum effort to meet any
  carbon budget) and the per-country allocation of who must cut most.
- **These planning numbers are fragile.** The committed checkpoint's projections
  oscillate from year to year: the global total follows a two-year sawtooth, with
  consecutive years differing by up to 3 Gt early on and about 1 Gt near 2050, so any
  single year (2050 included) sits about 0.5 Gt off the smoothed path. Across four
  training seeds of the same configuration (this checkpoint is seed 0), BAU 2050 ranges
  from 47 to 64 Gt and the maximum-effort cut from under 1% to 46%, and for one seed
  more effort raises emissions. Treat these numbers as provisional until a
  configuration is selected.

The raw numbers are in [`results/tables/`](results/tables). The figures in
[`results/figures/`](results/figures) predate the retrain (see Figures).

---

## What changed on 2026-11-09

Commit `3fc2559` made the evaluation leak-free and changed both the dynamics (network
inputs reordered; momentum inputs standardised and driven only by the model's natural,
uncontrolled change) and the planner (rollouts start from the observed momentum). The
checkpoint trained before it no longer matched the code (in-sample CO₂ R² 0.54), so it
was retrained with `scripts/02_train.py`. Model selection is still pending, so the
retrain uses the default configuration (`orig` in `eval/configs.py`). Each number below
was regenerated with the script that produces it.

| number | before | now | why it changed |
|---|---|---|---|
| in-sample CO₂ R² / MAPE | 0.9993 / 3.7% | 0.9982 / 5.3% | retrained model; this training run's loss spikes and does not fully recover (seeds 1 and 3 reach 3.5–3.8%, seed 2 8.1%) |
| in-sample demand R² / MAPE | 0.9987 / 4.3% | 0.9964 / 6.5% | as above |
| in-sample fuel-mix MAE | 0.8 pts | 1.8 pts | as above |
| physics layer: R², median error, emission factors | 0.9986, 4.7%, 0.316 / 0.231 / 0.198 | unchanged | does not depend on the trained model |
| BAU 2050, heuristic drivers | 47.7 Gt | 47.4 Gt | retrained model and planner changes |
| maximum-effort 2050 | 29.9 Gt (37% below BAU) | 25.4 Gt (46% below BAU) | as above; see the note below the table |
| cumulative 2023–2050 at maximum effort, scaled to global | ≈866 Gt | ≈959 Gt | as above |
| warming avoided by maximum effort | 0.18 °C (0.11–0.25) | 0.17 °C (0.10–0.24) | as above |
| 2016–2022 backtest (best per-country MAPE, 8.0% vs 8.5%; aggregate −15%), conformal coverage 0.79, transfer R² 0.996 | reported | withdrawn | emission factors, normalisation and static features (including each country's final-year fossil share) were computed from all years, so post-cutoff data reached every forecast |

The old maximum-effort pathway fell steeply to 26.6 Gt in 2030 and then rose; the new
one falls slowly and keeps falling. Since `3fc2559`, controls no longer feed the
momentum state, so the planner cannot amplify its own early cuts. The retrained dynamics
also differ, and the two effects cannot be separated because the old weights do not run
under the new code. This is also why cumulative emissions rose while the 2050 value
fell. The earlier claim that maximum effort still left 2050 emissions above today's
level was already wrong (29.9 Gt was 14% below 2022's 34.8 Gt) and has been dropped.
The 1.5 °C conclusion, the feasibility bounds, 2022 emissions and the carbon budgets are
unchanged. The withdrawn tables remain in git history (`61fcb12`, `ba3c01b`).

---

## Figures

These figures predate commit `3fc2559` and show the previous checkpoint; `fig3` and
`fig4` show the withdrawn evaluation. `scripts/05_make_figures.py` makes a new set once a
configuration is selected, from the tables written by `scripts/03_evaluate.py`.

| | |
|---|---|
| `fig1_reconstruction.png` | Physics layer: reconstructed vs reported CO₂; emission factors |
| `fig2_insample.png` | Free-running 1990–2022 fit (global + five countries) |
| `fig3_backtest.png` | 2016–2022 holdout: aggregate forecast and per-country MAPE (withdrawn) |
| `fig4_transfer.png` | Forecasting 12 countries the shared dynamics never saw (withdrawn) |
| `fig5_planning.png` | Emissions pathways to 2050 + cost-of-abatement frontier |
| `fig6_mix.png` | Fuel-mix transition, business-as-usual vs maximum feasible |

---

## Method

The energy system of each country is described by two state variables per year:

```
I_t = log( primary_energy_t / GDP_t )     # log energy intensity (a Kaya factor)
s_t = fuel-mix shares on the 9-fuel simplex
```

Demand is reconstructed as `D_t = GDP_t · exp(I_t)` and emissions by the fixed physics
layer. GDP and population are **exogenous drivers** (observed in training, scenario
inputs in planning) — the same framing integrated-assessment models use with SSP
pathways. A shared SiLU MLP with per-country embeddings predicts the **bounded** annual
change in intensity and mix (bounds encode capital-stock inertia); a learned per-country
emission-factor correction absorbs refining/petrostate residuals. Everything is written
in JAX and trained end-to-end by differentiable simulation (`optax` AdamW, ~20 s for 4000
steps on 4 CPU cores).

Planning controls — clean-deployment effort and efficiency effort — enter the dynamics
additively and are zero during fitting, so the fitted model reproduces observed behaviour
and the planner perturbs it causally. For the committed checkpoint, emissions fall
**monotonically** as effort rises (verified in the test suite); this does not hold for
every training seed of the current default configuration (see Limitations). Warming is
obtained from cumulative emissions via the TCRE relationship (IPCC AR6 central value
0.45 °C per 1000 GtCO₂).

| module | file | what it does |
|---|---|---|
| A — Physics | `model/emissions.py` | CO₂ = Σ EFₖ·energyₖ + process; factors fit by NNLS, held fixed |
| B — Dynamics | `model/dynamics.py`, `model/world_model.py` | learned differentiable rollout of demand + mix |
| — Training | `train/fit.py`, `train/losses.py` | fit across countries by differentiable simulation |
| — Evaluation | `eval/` | backtest, baselines, transfer, conformal calibration |
| C — Planning | `plan/optimize.py`, `plan/scenario.py` | optimal control of decarbonisation pathways |
| D — Climate | `climate/tcre.py` | cumulative CO₂ → warming and carbon budgets |

---

## Reproduce

```bash
pip install -e .                     # numpy, pandas, jax, optax, scipy, matplotlib
bash scripts/download_data.sh        # fetch OWID CO2 + energy data (CC-BY)
python scripts/01_build_dataset.py   # build the 60-country panel (committed too)
python scripts/02_train.py           # train the world model (~20 s, CPU)
python scripts/03_backtest.py        # physics-layer and in-sample tables
python scripts/04_plan.py            # BAU / max-ambition / frontier / warming, SSP1-5
python scripts/seed_sensitivity.py   # retrain with seeds 0-3: stability and spread
pytest -q                            # 60 tests
```

Model selection and out-of-sample reporting: `scripts/06_sweep.py --phase val` runs the
validation folds (`scripts/slurm/` runs the same jobs as a SLURM array). After
selection, `scripts/03_evaluate.py --config <selected>` writes the test-fold,
calibration and transfer tables, and `scripts/05_make_figures.py` draws the figures
from them. Job outputs and report entries record how many steps each fit was trained
for. The sweep merges into an existing report but refuses to mix training lengths in
one, and every script that loads the checkpoint refuses it if
`results/tables/config.json` shows it was trained with a configuration other than the
one the script runs it under. The processed panel (`data/processed/panel.npz`,
125 KB) and a trained checkpoint are committed, so `03_backtest.py` and `04_plan.py`
run without re-downloading or retraining.

---

## Limitations (read these)

- **Model selection is pending, so out-of-sample performance is not established.** All
  16 configurations in `eval/configs.py` have been scored on the validation folds at
  4000 training steps, the length `scripts/02_train.py` trains for (2 seeds;
  `results/sweep_val/report.json`). Multiple shooting is decisive: the ten
  configurations that use it (scores 11.0–13.9) beat the six that do not (15.9–23.7)
  on each seed separately. The leaders cannot be told apart yet: `ms_percap_damped`
  (11.05) and `ms_full_shootonly` (11.29) differ by 0.24, while one configuration's
  score typically moves by about 1.5 from one seed to another. Training length also
  reorders them: round 1, run at 3000 steps, put `ms` first (11.4); at 4000 steps it
  is eighth (12.8). The round-1 report is kept in
  `tests/data/report_round1_3000steps.json`.
- **Training of the default configuration is unstable and seed-sensitive.** The
  committed run's loss spikes between steps 800 and 1200 and ends higher than before the
  spike, and its projections oscillate from year to year. Across seeds 0–3
  (`scripts/seed_sensitivity.py`, `results/tables/seed_sensitivity.json`), BAU 2050
  ranges from 47 to 64 Gt. The two runs that trained cleanly (seeds 1 and 3) project
  higher emissions and respond weakly to effort (seed 3: a 17% cut) or perversely (seed
  1: more effort raises emissions, which fails two planning tests). The suite passes
  for the committed checkpoint, but planning causality is not yet a robust property of
  the default configuration.
- **Short-horizon forecasting is not where this model wins.** On the leak-free
  validation folds, trend-Kaya extrapolation beats every world-model configuration tried
  on both per-country and aggregate error (validation score 9.85; the best configuration
  scores 11.05). CWM's value is being a *structural,
  differentiable* simulator that supports counterfactual and planning queries no
  extrapolative baseline can perform — not point forecasting.
- **Drivers are exogenous.** GDP and population paths are assumptions; results are
  conditional on them. Planning uses SSP1–5 drivers (IIASA SSP database v3.2) with SSP2
  as the default; the earlier heuristic (recent CAGR tapering to a long-run rate) is
  kept for comparison and for the headline numbers above.
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
[energy](https://github.com/owid/energy-data) datasets, released under CC-BY. Code in
this repository is MIT-licensed.
