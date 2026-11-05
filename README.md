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

---

## Headline results

**The simulator is faithful.** Running free (autoregressively from 1990, given only GDP
and population paths), it reconstructs 33 years of history for all 60 countries:

| quantity | R² | MAPE |
|---|---|---|
| CO₂ emissions | 0.9993 | 3.7% |
| primary energy demand | 0.9987 | 4.3% |
| fuel-mix shares | — | 0.8 pts (MAE) |

The physics layer alone reconstructs reported CO₂ with **R² = 0.9986** (median error
4.7%), with emission factors recovered from data in the correct physical order:
coal 0.316 > oil 0.231 > gas 0.198 MtCO₂/TWh.

**It is competitive and well-calibrated out-of-sample — with an honest caveat.** On a
2016–2022 temporal holdout, CWM has the **best per-country accuracy** of five methods
(MAPE 8.0% vs 8.5% for the strongest baseline). But on the *global aggregate* it
undershoots by ~15%, almost entirely because of **China's post-2015 regime** (the model
extrapolates China's historic efficiency gains too far — the same optimistic-decoupling
bias that affects integrated-assessment models). For short-horizon aggregate CO₂, naive
trend extrapolation is famously hard to beat, and here it wins. Rolling-origin conformal
intervals are, however, **well calibrated**: the 80% band achieves 0.79 leave-one-cutoff-out
coverage. *The model knows what it doesn't know.*

**It transfers to unseen countries.** Trained on 48 countries and shown none of the
other 12, then adapting only each held-out country's small latent on its pre-2015 data,
CWM forecasts those unseen countries at **R² = 0.996** — evidence it learned transferable
structure, the defining property of a world model.

**It plans.** Optimising inside the simulator, under feasibility limits taken from the
fastest *sustained* historical transitions (low-carbon share +1.8%/yr, à la Denmark;
energy intensity −3.0%/yr, à la China):

- **Business-as-usual** emissions for these 60 countries grow from 34.8 Gt (2022) to
  **47.7 Gt by 2050**.
- **Maximum feasible effort** bends that down to **29.9 Gt** — a 37% cut versus BAU, but
  still above today's level, because economic growth offsets much of the gain.
- Scaled to global, cumulative 2023–2050 emissions are **≈866 GtCO₂ even under maximum
  ambition**, which **blows through the ≈578 Gt remaining 1.5 °C budget**. Within
  historically-achieved transition rates, **1.5 °C is out of reach** without negative
  emissions; maximum effort avoids ≈0.18 °C (0.11–0.25 °C) of warming versus BAU for
  this group.
- The model produces the full **cost-of-abatement frontier** (minimum effort to meet any
  carbon budget) and the per-country allocation of who must cut most.

Figures for all of the above are in [`results/figures/`](results/figures) and the raw
numbers in [`results/tables/`](results/tables).

---

## Figures

| | |
|---|---|
| `fig1_reconstruction.png` | Physics layer: reconstructed vs reported CO₂; emission factors |
| `fig2_insample.png` | Free-running 1990–2022 fit (global + five countries) |
| `fig3_backtest.png` | 2016–2022 holdout: aggregate forecast and per-country MAPE |
| `fig4_transfer.png` | Forecasting 12 countries the shared dynamics never saw |
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
in JAX and trained end-to-end by differentiable simulation (`optax` AdamW, ~17 s for 4000
steps on 4 CPU cores).

Planning controls — clean-deployment effort and efficiency effort — enter the dynamics
additively and are zero during fitting, so the fitted model reproduces observed behaviour
and the planner perturbs it causally. Emissions response to effort is **monotonic**
(verified in the test suite). Warming is obtained from cumulative emissions via the
TCRE relationship (IPCC AR6 central value 0.45 °C per 1000 GtCO₂).

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
python scripts/02_train.py           # train the world model (~17 s, CPU)
python scripts/03_backtest.py        # in-sample, backtest, calibration tables
python scripts/04_plan.py            # BAU / max-ambition / frontier / warming
python scripts/05_make_figures.py    # all figures
pytest -q                            # 28 tests
```

The processed panel (`data/processed/panel.npz`, 125 KB) and a trained checkpoint are
committed, so scripts 03–05 run without re-downloading or retraining.

---

## Limitations (read these)

- **Short-horizon aggregate forecasting is not where this model wins.** Trend and
  growth-persistence baselines beat it on 2016–2022 aggregate CO₂; the gap is driven by
  China. CWM's value is being a *structural, differentiable* simulator that supports
  counterfactual and planning queries no extrapolative baseline can perform — not point
  forecasting.
- **Drivers are exogenous.** GDP and population paths are assumptions; results are
  conditional on them. The default future scenario is deliberately simple (recent CAGR
  tapering to a long-run rate); swap in SSP pathways for scenario analysis.
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
