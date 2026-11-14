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

> **Current state (2026-11-12).** The selected configuration is `ms_full` (multiple
> shooting, per-capita levels, EMA momentum, a damped trend, no calendar-time inputs),
> and the checkpoint is trained with it. On 2026-11-11 a leak was found and fixed in how
> backtest models were trained: their loss saw realised GDP and population after the
> cutoff. Re-scored leak-free, the validation folds still lead to `ms_full`, and the
> test folds were then re-scored once, leak-free, for the same configuration. The
> out-of-sample numbers below are the corrected ones. The planning numbers, means over
> ten training seeds of full-data fits, were never affected. See
> [What changed on 2026-11-12](#what-changed-on-2026-11-12).

---

## Headline results

**The simulator reproduces history.** Running free (autoregressively from 1990, given
only GDP and population paths), it reconstructs 33 years of history for all 60 countries:

| quantity | R² | MAPE |
|---|---|---|
| CO₂ emissions | 0.9997 | 3.3% |
| primary energy demand | 0.9994 | 4.4% |
| fuel-mix shares | — | 0.8 pts (MAE) |

The physics layer alone reconstructs reported CO₂ with **R² = 0.9986** (median error
4.7%), with emission factors recovered from data in the correct physical order:
coal 0.316 > oil 0.231 > gas 0.198 MtCO₂/TWh.

**Out of sample, it is about level with trend extrapolation given the true drivers.**
The protocol keeps model selection (validation cutoffs 2005, 2007 and 2009, six-year
horizons) apart from reporting (test cutoffs 2015–2017, horizons to 2022, scored after
selection). On the test folds, given realised GDP and population, the selected
configuration's per-country error (MAPE) is 7.5% against 7.8% for trend-Kaya
extrapolation, and its error on the 60-country total 0.9% against 1.7% (score 8.3
against 9.5). On the validation folds trend-Kaya is ahead (9.9 against 11.4), so the two
are level within seed noise. The original configuration's per-country error is 13.6%.
Given instead the SSP2 paths published before the cutoffs (2014 vintage), its
per-country error rises to 10.6%, still below trend-Kaya's 11.0%, but baselines that
ignore the drivers do better (8.8%). Those paths raise the world model's forecast of the
60-country total in the final year by 4–5% over the realised drivers (3–8% across
SSP1–5). Its 80% prediction intervals, calibrated on the validation folds, cover 87% of
test outcomes. It does not transfer well to countries it was not trained on (see
Limitations).

**It plans.** The network projects business as usual, and a plan adds policy-driven
change on top, up to feasibility limits taken from the fastest *sustained* historical
transitions (low-carbon share +1.8 points/yr, à la Denmark; energy intensity −3.0%/yr, à
la China). Maximum effort runs every country at those limits, or at its
business-as-usual rates where those are faster. With the same heuristic GDP and
population paths as earlier versions of this README, each number is the mean over ten
training seeds of the selected configuration, with the lowest and highest seed in
brackets:

- **Business-as-usual** emissions for these 60 countries go from 34.8 Gt (2022) to
  **37.8 Gt in 2050** (31.9–42.1; with SSP2 drivers 36.9 Gt, and 35.0–40.3 Gt across the
  means for SSP1–5).
- **Maximum feasible effort** bends that down to **12.6 Gt** (11.4–13.2; SSP2: 12.5 Gt),
  a **67% cut** versus BAU (61–70%) and 64% below 2022. It assumes that planned change
  persists, with no rebound and no crowding out, and that every country can run at the
  limits for 28 years. If rebound takes back a quarter of the efficiency gains, a
  quarter of the planned low-carbon energy displaces business as usual's own, and the
  limits are three quarters as fast, the cut is 40% (32–46%; see "What changed on
  2026-11-14"). Until 2026-11-13 the planner let the network respond to the planned
  states, where it extrapolates, and found 41% (26–59%).
- Scaled to global, cumulative 2023–2050 emissions are **≈706 GtCO₂ even under maximum
  ambition** (680–726; SSP2: ≈700), which **blows through the ≈578 Gt remaining 1.5 °C
  budget** under every seed. Within historically-achieved transition rates, **1.5 °C is
  out of reach** without negative emissions; maximum effort avoids ≈0.19 °C of warming
  versus BAU for this group (0.11–0.26 °C over the TCRE likely range; 0.15–0.21 °C over
  the seeds).
- The model produces the full **cost-of-abatement frontier** (minimum effort to meet any
  carbon budget) and the per-country allocation of who must cut most. Plans are
  optimised against all ten seeds at once.
- **Hedging against the model's own uncertainty costs 62% more effort.** A plan that
  meets a cumulative budget under every SSP *and* every seed takes 62% more effort than
  one tuned to SSP2 and the ten-seed mean, which overshoots the budget by up to 52 Gt
  under SSP5 when the least favourable seed is right. With moderate rebound, crowding out
  and limits at three quarters, hedging costs 257% more. The committed checkpoint (seed 0)
  gives a 67% cut alone.

The raw numbers are in [`results/tables/`](results/tables) and the figures in
[`results/figures/`](results/figures).

---

## What changed on 2026-11-14

**What the frontier's assumptions are worth.** Maximum effort's 67% cut rests on three
assumptions. Planned efficiency gains are not taken back by higher energy use (rebound).
Planned low-carbon energy adds to business as usual's own growth instead of displacing
it (crowding out). And every country can run at the feasibility limits for 28 years at
once, though each limit is a rate that individual countries sustained for a decade.
`scripts/plan_sensitivity.py` (`results/tables/plan_sensitivity.json`) relaxes them one
at a time and together. The planner now takes rebound and crowding out as shares of what
a plan achieves (`make_worlds(..., rebound=, crowd_out=)`), and still charges the plan's
full effort. Ten-seed means, with the lowest and highest seed in brackets:

| maximum effort, 2050 | cut vs BAU, heuristic drivers | cut vs BAU, SSP2 | global cumulative 2023–2050, heuristic |
|---|---|---|---|
| as stated | 67% (61–70%) | 66% (61–69%) | ≈706 Gt (680–726) |
| rebound 25% | 63% (57–66%) | 62% (56–65%) | ≈744 Gt (720–764) |
| rebound 50% | 58% (52–62%) | 58% (52–61%) | ≈785 Gt (763–810) |
| crowding out 25% | 60% (55–64%) | 59% (54–62%) | ≈752 Gt (726–769) |
| crowding out 50% | 53% (49–57%) | 52% (48–56%) | ≈798 Gt (772–815) |
| limits ×0.75 | 50% (40–56%) | 49% (40–54%) | ≈827 Gt (783–861) |
| limits ×0.5 | 31% (21–40%) | 30% (21–37%) | ≈947 Gt (879–996) |
| all three moderate (25%, 25%, ×0.75) | 40% (32–46%) | 39% (32–44%) | ≈896 Gt (857–930) |

The limits matter most. At three quarters of the record rates, maximum effort loses a
quarter of its cut, and the seeds spread wider, because the room between their
business-as-usual rates and the lower limits varies more. 1.5 °C stays out of reach
under every setting and seed (remaining budget ≈578 Gt).

**What they are worth in effort.** `scripts/plan_sensitivity.py --costs`
(`results/tables/plan_sensitivity_costs.json`, about 45 minutes) re-optimises
`04_plan.py`'s SSP2 cost-of-abatement frontier at the same budgets, and its robust
planning with the same budget rule, for five of the settings. Under the stated
assumptions it reproduces `plan.json`. The budgets are cumulative emissions of the 60
countries over 2023–2050, met on average over the ten seeds; the effort is in the
planner's cost units; "—" marks a budget that even maximum effort misses:

| effort to meet a budget (SSP2) | 709 Gt | 764 Gt | 819 Gt | 875 Gt | robust budget | price of robustness |
|---|---|---|---|---|---|---|
| as stated | 0.66 | 0.32 | 0.16 | 0.08 | 767 Gt | +62% |
| rebound 25% | 1.13 | 0.42 | 0.20 | 0.09 | 792 Gt | +74% |
| crowding out 25% | 1.54 | 0.54 | 0.26 | 0.12 | 800 Gt | +83% |
| limits ×0.75 | — | 1.53 | 0.34 | 0.13 | 865 Gt | +147% |
| all three moderate | — | — | 2.44 (maximum effort) | 0.31 | 914 Gt | +257% |

Weaker policy raises the effort every budget needs, most steeply near what maximum
effort can reach, and puts the tightest budgets out of reach: 654 Gt, which costs 2.39
under the stated assumptions, is unreachable under every relaxation. With all three
relaxed, 819 Gt is exactly what maximum effort reaches (0.05 Gt to spare), so meeting it
takes maximum effort. Hedging gets dearer too. The members' maximum efforts spread
further apart, so a plan that holds under the least favourable one costs more relative
to one tuned to their mean, which overshoots by up to 74 Gt instead of 52. The robust
budget moves with each setting, since `04_plan.py`'s rule places it between what maximum
effort reaches and business as usual, so these prices compare hedging at comparable
difficulty, not at one budget.

---

## What changed on 2026-11-13

**The planner plans at the feasibility frontier.** Until now it fed its controls back
into the network: a control moved a country's energy intensity or fuel mix, and the
network computed the next year's change from a state no country has been in. Each
training seed extrapolated differently there ("Why the training seeds disagree", below),
so maximum effort cut 2050 emissions by 26–59% depending on the seed, and did not run at
the stated limits. Now the network projects business as usual only. A plan adds extra
annual rates of intensity fall and low-carbon gain on top, capped so that the total
rates never exceed the limits (`plan/optimize.py`), and maximum effort lifts every
country to the limits, or leaves it where business as usual is faster. Business as usual
is unchanged. Ten-seed means under the heuristic drivers, with the lowest and highest
seed in brackets:

| | until 2026-11-12 (closed loop) | now (frontier) |
|---|---|---|
| maximum effort, 2050 | 22.1 Gt (15.0–27.6) | 12.6 Gt (11.4–13.2) |
| cut versus business as usual | 41% (26–59%) | 67% (61–70%) |
| global cumulative 2023–2050 | ≈840 Gt (742–919) | ≈706 Gt (680–726) |
| warming avoided versus business as usual | 0.13 °C | 0.19 °C |
| extra effort for a plan robust to every SSP and seed | +119% | +62% |

The seeds now differ only through their business-as-usual paths, and the planner states
its assumption instead of learning it: planned change persists, with no rebound and no
crowding out (Limitations). 1.5 °C stays out of reach under every seed. The selection's
tie-break used the old planner's checks and would no longer separate `ms_full` from
`ms_percap_damped` (Limitations); the selection has not been revisited.

**Planned low-carbon energy displaces business as usual's fossil mix.** Each unit of
planned low-carbon gain used to come out of the country's 2022 fossil mix. Where
business as usual phases a fuel out (mostly coal), that drove the fuel's share below
zero, where it was clipped and the mix renormalised: applied at the frontier, that rule
delivered less than the plan in about half of all country-years under maximum effort.
The planned low-carbon share now displaces fossil fuels in proportion to their shares in
business as usual that year, until none is left.

**Plans are optimised to convergence.** The frontier and robust plans ran 1,000–1,500
Adam steps at a constant learning rate, far from converged: one budget's plan cost 0.80
effort units after 1,000 steps and 0.32 after 20,000 (0.317 after 50,000), and the
closed-loop frontier undershot its budgets by up to 22 Gt. Plans now take 20,000 steps
with a learning rate decaying to zero, which the frontier makes cheap
(`scripts/04_plan.py` takes about seven minutes), and plans for an ensemble-mean budget
meet it to within 0.03 Gt. A robust plan bounds a smooth maximum of the fifty worlds'
cumulative emissions (log-sum-exp, at most 2 Gt above the true maximum; its worst world
ends 0.8 Gt below the budget). That converges within 20,000 steps, while penalising each
world separately was still improving after 40,000. Unconverged, the frontier planner had
put the price of robustness at +47%.

`scripts/seed_sensitivity.py` was rerun with the new planner
(`results/tables/seed_sensitivity.json`): training, in-sample fit and business as usual
reproduce exactly, and the planning checks hold on all four seeds.

**Calendar-time inputs do not fix the early-cutoff efficiency stall.** Earlier today,
as merged, Limitations said the stall comes with dropping the calendar-time inputs, and
`CLAUDE.md` named restoring them, by re-selecting `ms_percap_damped`, as the fix. The
inputs avoid the stall in the backtests only because a backtest feeds calendar time past
the cutoff, beyond the range the model was trained on.
- Held at the cutoff year's value, as planning holds it at 2022, `ms_percap_damped`
  stalls like `ms_full` and scores worse on the validation folds: 12.57 against
  `ms_full`'s 11.54 on seeds 0–3 (10.70 with calendar time running on).
- In planning, its business as usual lets the mature economies' energy intensity fall by
  only 0.5%/yr over 2022–2032 (observed 2012–2022: 2.2%/yr). It reaches 55.8 Gt in 2050,
  against 39.9 Gt for the selected configuration.

Details: `scripts/time_inputs.py`, `results/tables/time_inputs.json`, Limitations.

---

## What changed on 2026-11-12

**Why the training seeds disagree on maximum effort.** The ten training seeds agree
closely on business as usual but cut it under maximum effort by 26–59%.
`scripts/control_response.py` (`plan/response.py`, `results/tables/control_response.json`)
finds why. The planner's rollouts were closed-loop: a control moved a country's energy
intensity or fuel mix, and the network computed the next year's natural change from the
controlled state. No country has been in those states (that low-carbon share at that
income, say), so the network's response is extrapolation, and each seed extrapolates
differently. Under the heuristic drivers, across the ten seeds:

- only 19–38% of the efficiency effort survives to 2050: the network pushes intensity
  back up;
- 22–74% of the push towards low-carbon energy survives;
- the low-carbon push alone changes the 60 countries' energy demand in 2050 by −39% to
  +39%, though it asks for no change in demand. Under SSP2, it raises one seed's
  emissions;
- China alone spans 5.2–13.1 Gt at maximum effort in 2050, against 14.2 Gt business as
  usual.

The seeds' damping parameters are nearly identical (0.86–0.89 for intensity, 0.50–0.54
for the mix), so the difference lies in the network's state-dependent correction. Two
counterfactuals keep the network on the business-as-usual path it was trained to
reproduce:

| maximum effort, 2050, heuristic drivers | cut vs BAU | intensity fall, %/yr | low-carbon share gain, points/yr |
|---|---|---|---|
| closed loop (the planner until 2026-11-13) | 26–59% | 0.9–4.6 | 0.97–1.80 |
| open loop: controls added to BAU's changes | 79–84% | 3.9–4.7 | 2.11–2.47 |
| frontier: total rates lifted to the limits | 61–70% | 3.1–3.4 | 1.80–1.88 |
| business as usual | — | 0.9–1.7 | 0.48–1.02 |
| feasibility limits | — | 3.0 | 1.79 |

Ranges are over the ten seeds; rates are country averages over 2022–2050, weighted by
2022 energy use. The open loop double-counts: the limits are *total* rates that
countries have sustained for a decade, and it adds them on top of business as usual's
own changes. The frontier lifts each country's total rates to the limits, or keeps
business as usual's where they are faster. Its spread (9 points rather than 33) comes
only from the members' business-as-usual paths. Under SSP2 the cuts are 23–58%, 80–84%
and 61–69%. Since 2026-11-13 the frontier is the planner itself, and the row shows it
with that day's corrected fuel-mix rule (before it, the frontier's low-carbon gain was
1.70–1.84 points/yr).

**A leak in how backtest models were trained.** Every fit with a cutoff (validation
folds, test folds, transfer test) trains on a free rollout that runs past the cutoff on
realised GDP and population. Two terms of that loss were not confined to the years up to
the cutoff. The curvature penalty saw the post-cutoff years (and, in the transfer test,
the held-out countries' trajectories). The level term's normaliser, a scale that cancels,
saw them too, which affected only rounding. Scrambling every post-cutoff value changed
the curvature term about 150-fold. Fifteen of the sixteen configurations were affected;
`ms_full_shootonly`, which trains without that loss, was not. The fix (`46cc5ce`) confines
both terms to the fit window, and a test now checks that a fit on data scrambled after the
cutoff gives bit-for-bit the same parameters. Fits on all the data were not affected:
`scripts/02_train.py` still reproduces the checkpoint byte for byte, so the planning
ensemble and every planning number stand.

**The validation folds, leak-free.** All sixteen configurations were rerun
(`results/sweep_val/report.json`; the leaky report is in git history at `2648dda`). Scores
moved by about the size of seed noise (median 0.9 points, against 1.2 between seeds of
one configuration), in both directions:

| configuration | leaky | leak-free |
|---|---|---|
| `ms_percap_damped` | 11.05 | 10.79 |
| `ms_percap` | 12.20 | 10.96 |
| `ms_full_shootonly` | 11.29 | 11.29 (unaffected) |
| `ms_full` (selected) | 11.70 | 11.38 |
| `ms_notime_percap` | 11.85 | 12.06 |
| `orig` | 15.87 | 19.99 |
| trend-Kaya baseline | 9.85 | 9.85 |

Multiple shooting still beats every other configuration on each seed, and trend-Kaya
still beats every configuration. The leaders changed: `ms_percap` rose from seventh to
second. The tie-break that chose `ms_full` uses planning checks on four training seeds of
full-data fits, which the leak did not touch. It was extended to the two new candidates
(`results/tables/selection/`), and `ms_full` is still the only one whose planning checks
hold with smooth projections on every seed:

- `ms_percap_damped` fails effort monotonicity on one seed.
- `ms_percap` oscillates every year on one seed, and its training spikes on three.
- `ms_full_shootonly` oscillates on three seeds, `ms_notime_percap` on two.

So the selection stands.

**The transfer test trained with the wrong optimiser.** It fitted its shared dynamics
with plain Adam at a constant rate, not the clipped, cosine-scheduled AdamW that every
other fit uses, and for `ms_full` that training spiked and did not recover. It now uses
the deployed recipe (`0c93b5c`). On a transfer scored on the validation period only
(cutoff 2009, years 2010–2015, three seeds), the error on the 12 unseen countries falls
from 13.7–19.7% to 12.5–13.7%, against 9.0% for trend-Kaya: better, but still worse than
extrapolation.

**The test folds, re-scored leak-free.** With the owner's go-ahead, the test folds were
scored a second time, for the same two configurations as on 2026-11-10 (`ms_full`,
already selected, and `orig`). This was a measurement correction; the selection had been
settled on the validation folds first and was not revisited. Every baseline is
bit-for-bit unchanged; only the world-model fits differ:

| test folds 2015–2017 | from leaky fits (2026-11-10) | leak-free |
|---|---|---|
| `ms_full`, realised drivers: score | 9.40 | 8.33 |
| — per-country MAPE / mean aggregate error | 7.6% / 1.8% | 7.5% / 0.9% |
| trend-Kaya, realised drivers: score | 9.49 | 9.49 |
| `ms_full`, ex-ante SSP-v9 drivers: score | 14.09 | 13.68 |
| growth persistence (ignores drivers): score | 11.15 | 11.15 |
| `orig`, realised / ex-ante: score | 19.96 / 27.03 | 18.84 / 24.18 |
| 80% interval coverage | 0.859 | 0.872 |
| transfer to 12 unseen countries, MAPE (seed 0) | 21.2% | 15.4% (trend-Kaya 7.6%) |

The leak had not flattered the model; if anything, it cost it a little, as on the
validation folds. The transfer number also reflects the corrected shared-model training:
the held-out countries' adapted in-sample error falls from 14.9% to 5.2%. China's
final-year CO₂ is still undershot by 6–8%, almost all of it energy demand
(`results/tables/china_decomposition.json`).

**Where the transfer error comes from.** One seed was not enough. With the owner's
go-ahead, the transfer test was rerun for all five test-fold seeds: same protocol, cutoff
2015, same 12 countries. Seed 0, the 15.4% above, was the worst of the five:

| 12 transfer countries, 2016–2022 | per seed | mean |
|---|---|---|
| transfer protocol | 15.4, 9.1, 10.1, 11.4, 13.4% | 11.9% |
| model trained on all 60 countries | 7.9, 9.5, 7.0, 8.3, 6.5% | 7.9% |
| trend-Kaya | | 7.6% |

So on the test period transfer costs about 4 points, and its seed-to-seed spread (about 6
points) is twice the in-training model's. On the validation period
(`scripts/transfer_diagnostics.py`) the cost is about 1 point, and most of the gap there
comes from a bias shared by every model trained up to 2005–2009: they stall mature
economies' efficiency gains (see Limitations).

---

## What changed on 2026-11-11

The planning numbers now come from an ensemble of ten training seeds of the selected
configuration, not from the committed checkpoint alone. On 2026-11-10 that checkpoint
(seed 0) gave a 59% maximum-effort cut, while seeds 1–3 gave 33–43%. Ten seeds were
trained (`scripts/02b_train_ensemble.py`, `results/model/ensemble/`; seed 0 is the
checkpoint itself), and `scripts/04_plan.py` now evaluates every pathway under each of
them. It optimises the frontier for the ten-seed mean, and the robust plan for every
SSP and every seed. `results/tables/plan.json` keeps each seed's values.

| number, heuristic drivers | seed 0 alone (2026-11-10) | ten-seed mean (lowest–highest seed) |
|---|---|---|
| BAU 2050 | 39.5 Gt | 37.8 Gt (31.9–42.1) |
| maximum-effort 2050 | 16.2 Gt | 22.1 Gt (15.0–27.6) |
| cut versus BAU | 59% | 41% (26–59%) |
| cumulative 2023–2050 at maximum effort, scaled to global | ≈765 Gt | ≈840 Gt (742–919) |
| warming avoided by maximum effort | 0.17 °C | 0.13 °C (0.10–0.17) |
| price of robustness (robust plan versus SSP2-tailored plan) | +22% (SSPs only) | +119% (SSPs and seeds) |

Seed 0 is the most optimistic of the ten. Every seed's maximum effort stays below its
BAU, and under every seed cumulative emissions exceed the 1.5 °C budget, so those
conclusions stand.

---

## What changed on 2026-11-10

Model selection is done. All 16 configurations in `eval/configs.py` were scored on the
validation folds at 4000 training steps, the length the deployed model trains for (2
seeds; `results/sweep_val/report.json`). Multiple shooting decided the first cut: the
ten configurations that use it beat the six that do not on each seed. The three leaders
(`ms_percap_damped` 11.05, `ms_full_shootonly` 11.29, `ms_full` 11.70) were closer than
the seed noise (one configuration's score moves by about 1.5 between seeds), so they
were compared on what made the original configuration fragile: whether the planning
checks hold for each of four training seeds (`scripts/seed_sensitivity.py --config`,
`results/tables/selection/`). `ms_full` was the only one that passed them on every seed
with smooth projections; `ms_percap_damped` failed effort monotonicity on one seed and
`ms_full_shootonly` oscillated. The test folds were scored once, after the choice.

| number | `orig` (the default until 2026-11-10) | `ms_full` (now) |
|---|---|---|
| in-sample CO₂ R² / MAPE | 0.9982 / 5.3% | 0.9997 / 3.3% |
| in-sample demand R² / MAPE | 0.9964 / 6.5% | 0.9994 / 4.4% |
| in-sample fuel-mix MAE | 1.8 pts | 0.8 pts |
| BAU 2050, heuristic drivers | 47.4 Gt | 39.5 Gt |
| maximum-effort 2050 | 25.4 Gt (46% below BAU) | 16.2 Gt (59% below BAU) |
| cumulative 2023–2050 at maximum effort, scaled to global | ≈959 Gt | ≈765 Gt |
| warming avoided by maximum effort | 0.17 °C (0.10–0.24) | 0.17 °C (0.10–0.23) |
| BAU 2050 across training seeds 0–3 | 47–64 Gt | 38.6–42.1 Gt |
| planning checks passed, seeds 0–3 | 3 of 4 seeds | 4 of 4 seeds |
| test folds, realised drivers: score / per-country MAPE | 19.96 / 14.0% | 9.40 / 7.6% |

The physics layer, the feasibility bounds, 2022 emissions, the carbon budgets and the
1.5 °C conclusion are unchanged. The test-fold, calibration and transfer tables and the
figures are new.

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

Drawn by `scripts/05_make_figures.py` from the selected configuration's checkpoint and
the tables written by `scripts/03_evaluate.py` and `scripts/04_plan.py`.

| | |
|---|---|
| `fig1_reconstruction.png` | Physics layer: reconstructed vs reported CO₂; emission factors |
| `fig2_insample.png` | Free-running 1990–2022 fit (global + five countries) |
| `fig3_ablation_validation.png` | Validation score of all 16 configurations, against the baselines |
| `fig4_test_and_china.png` | Test folds: per-country error by method, realised and ex-ante drivers; China's final-year error |
| `fig5_china_forecasts.png` | China from the observed 2015 state: selected and original configurations, trend-Kaya |
| `fig6_ssp_drivers.png` | SSP GDP and population drivers, 2014 and 2025 vintages |
| `fig7_ssp_pathways_robust.png` | BAU and maximum-effort pathways under SSP1–5; robust versus SSP2-tailored plan |
| `fig8_mix_ssp2.png` | Fuel-mix transition under SSP2, business-as-usual vs maximum feasible |
| `fig9_frontier_ssp2.png` | Emissions pathways to 2050 and the cost-of-abatement frontier under SSP2 |

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
emission-factor correction absorbs refining/petrostate residuals. The selected
configuration (`ms_full`) gives the network per-capita levels (energy and GDP per person)
rather than absolute ones, carries momentum as an exponential moving average inside a
damped-trend structure, uses no calendar-time inputs, and is trained both on one free
run from 1990 and on 8-year rollouts started from the observed state in every training
year (multiple shooting). Everything is written in JAX and trained end-to-end by
differentiable simulation (`optax` AdamW, about 2.5 minutes for 4000 steps on 4 CPU
cores).

Planning starts from business as usual. The network projects each country's natural path
of energy intensity and fuel mix, and a plan adds policy-driven change on top: an extra
annual fall in log intensity and an extra annual gain in the low-carbon share, per
country and year. Each is capped so that the total rate never exceeds its feasibility
limit, and the planned low-carbon share, split across sources in a fixed deployable
portfolio, displaces fossil fuels in proportion to their shares in business as usual
that year. The network is not run on the planned states: no country has been in them,
and training seeds respond to them differently ("Why the training seeds disagree").
Everything after business as usual is therefore closed-form and differentiable, and
emissions fall monotonically as effort rises, by construction. Plans are optimised by
gradient descent (Adam, with a learning rate decaying over 20,000 steps) against ten
training seeds and the five SSPs at once (`plan/ensemble.py`), in one batched
computation. Warming is obtained from cumulative emissions via the TCRE relationship
(IPCC AR6 central value 0.45 °C per 1000 GtCO₂).

| module | file | what it does |
|---|---|---|
| A — Physics | `model/emissions.py` | CO₂ = Σ EFₖ·energyₖ + process; factors fit by NNLS, held fixed |
| B — Dynamics | `model/dynamics.py`, `model/world_model.py` | learned differentiable rollout of demand + mix |
| — Training | `train/fit.py`, `train/losses.py` | fit across countries by differentiable simulation |
| — Evaluation | `eval/` | backtest, baselines, transfer, conformal calibration |
| C — Planning | `plan/optimize.py`, `plan/scenario.py`, `plan/ensemble.py` | plans on top of business as usual, up to the feasibility limits, against a seed ensemble |
| — Diagnostic | `plan/response.py` | why the network is kept on business as usual (the old closed loop) |
| D — Climate | `climate/tcre.py` | cumulative CO₂ → warming and carbon budgets |

---

## Reproduce

```bash
pip install -e .                     # numpy, pandas, jax, optax, scipy, matplotlib
bash scripts/download_data.sh        # fetch OWID CO2 + energy data (CC-BY)
python scripts/01_build_dataset.py   # build the 60-country panel (committed too)
python scripts/02_train.py           # train the selected configuration (~2.5 min, CPU)
python scripts/02b_train_ensemble.py # train seeds 1-9 for the planning ensemble (~25 min)
python scripts/03_backtest.py        # physics-layer and in-sample tables
python scripts/04_plan.py            # BAU / max effort / frontier / robust plans (~7 min)
python scripts/seed_sensitivity.py   # retrain with seeds 0-3: stability and spread
python scripts/control_response.py  # why the seeds disagree on maximum effort
python scripts/transfer_diagnostics.py  # where transfer error comes from (validation)
python scripts/stall_by_config.py    # which configurations stall the efficiency trend
python scripts/time_inputs.py        # time inputs in backtests and in planning (~15 min)
python scripts/stall_mechanism.py    # why early-cutoff models stall it (~5 min)
python scripts/plan_sensitivity.py   # maximum effort with the frontier's assumptions relaxed
                                     # (--costs: frontier and robust plans too, ~45 min)
pytest -q                            # 82 tests
```

Model selection and out-of-sample reporting: `scripts/06_sweep.py --phase val` scores
configurations on the validation folds (`scripts/slurm/` runs the same jobs as a SLURM
array), and `SELECTED_CONFIG` in `config.py` names the one that `default_config()`
returns. The test folds were then run with `scripts/06_sweep.py --phase test --configs
ms_full,orig --seeds 0,1,2,3,4 --ssp`, and `scripts/03_evaluate.py --config ms_full`
wrote the test-fold, calibration and transfer tables, from which
`scripts/05_make_figures.py` draws the figures. Job outputs and report entries record
how many steps each fit was trained for. The sweep merges into an existing report but
refuses to mix training lengths in one, and every script that loads the checkpoint
refuses it if `results/tables/config.json` shows it was trained with a configuration
other than the one the script runs it under. The processed panel
(`data/processed/panel.npz`, 125 KB) and a trained checkpoint are committed, so
`03_backtest.py` and `04_plan.py` run without re-downloading or retraining.

---

## Limitations (read these)

- **The selection rests on few seeds and one tie-break.** On the leak-free validation
  folds (2 seeds, `results/sweep_val/report.json`) the four leading configurations are
  within seed noise of each other (10.79–11.38). `ms_full`, fourth on score, is the only
  one whose planning checks held, with smooth projections, for each of four training
  seeds. A different tie-break could have chosen differently. The test folds were scored
  only after the choice: once from leaky fits and once, as a correction, leak-free, both
  times for the same two configurations. They must not be used to revisit the choice.
  The round-1 report (3000 training steps) is kept in
  `tests/data/report_round1_3000steps.json`.
  The tie-break's planning checks were run with the closed-loop planner. At the frontier
  they hold for every configuration by construction, and `ms_percap_damped`, best on
  score, failed only one of them (effort monotonicity, on one seed); its projections are
  as smooth as `ms_full`'s. With today's planner the tie-break would not have separated
  the two. They differ only in the calendar-time inputs, and `ms_percap_damped`'s lead
  on score depends on feeding calendar time past the cutoff, which planning never does.
  Held at the cutoff's value, as planning holds calendar time at 2022, it scores 12.57
  against `ms_full`'s 11.54 (seeds 0–3). Its planning business as usual also stalls
  efficiency gains (below). The sweep feeds every configuration with time inputs
  calendar time past the cutoff. The selection has not been revisited.
- **Training seeds disagree, and the ensemble covers only that.** The selected
  configuration trains without the loss spikes of the original one
  (`results/tables/seed_sensitivity.json`; the original's are in
  `results/tables/selection/seed_sensitivity_orig.json`), but across ten training seeds
  BAU 2050 spans 31.9–42.1 Gt and the maximum-effort cut 61–70% (26–59% with the
  closed-loop planner used until 2026-11-13). The planning numbers are therefore
  ten-seed means, and a plan robust to the seeds and the SSPs costs 62% more effort than
  one tuned to their mean under SSP2, and up to 257% more under relaxed assumptions
  ("What changed on 2026-11-14"). The ensemble captures the variation between training
  runs of one configuration only. It does not capture the uncertainty between
  configurations, in the model's structure, or in its feasibility bounds.
- **Planned change is assumed to persist, and maximum effort is generous.** The planner
  keeps the network on business as usual and adds policy-driven change on top. It
  therefore assumes no rebound in energy use after efficiency gains, no crowding out of
  business as usual's own low-carbon growth, and fossil fuels displaced in proportion to
  business as usual's mix. Letting the network respond to planned states instead (the
  planner until 2026-11-13) relies on extrapolation that training seeds disagree on,
  with cuts of 26–59% ("Why the training seeds disagree"). Maximum effort runs all 60
  countries at the feasibility limits for 28 years at once, which no group of countries
  has done: each limit is a rate that individual countries sustained for a decade.
  Relaxing these (`scripts/plan_sensitivity.py`) lowers the 67% cut to 63% with a
  quarter of the efficiency gains taken back by rebound, to 60% with a quarter of the
  low-carbon gain crowded out, to 50% with the limits at three quarters, and to 40% with
  all three. They also raise the effort any carbon budget needs, put the tightest out of
  reach, and raise the price of robustness from +62% to as much as +257% ("What changed
  on 2026-11-14").
- **Short-horizon forecasting is not where this model wins.** Given the true drivers,
  the selected configuration is slightly ahead of trend-Kaya extrapolation on the test
  folds (score 8.33 against 9.49), but behind it on the validation folds (11.38 against
  9.85): level, within seed noise. Given the drivers that were available at the time,
  baselines that ignore drivers beat both (test score 11.15 against 13.68). CWM's value
  is being a *structural, differentiable* simulator that supports counterfactual and
  planning queries no extrapolative baseline can perform — not point forecasting.
- **Transfer to unseen countries is worse than trend extrapolation.** Fitted on 48
  countries up to 2015 and adapted to 12 held-out ones from their own data up to 2015, the
  model forecasts their 2016–2022 emissions with 11.9% error averaged over five seeds
  (9.1–15.4%), against 7.6% for trend-Kaya (`results/tables/transfer.json`). By country
  the error ranges from 6% (Germany, Turkey) to 18% (United Kingdom, Kuwait). The model
  that trained on all 60 countries forecasts the same 12 over the same years with 7.9%
  error per seed (6.1% for the five-seed mean), so transfer costs about 4 points on the
  test period. On the validation period it costs about 1 point (12.6% against 11.6%
  in-training, trend-Kaya 9.6%; `scripts/transfer_diagnostics.py`), and most of the gap
  to trend-Kaya there comes from the bias in the next item. One transfer seed is not a
  reliable measurement: on the test period the seeds differ by up to 6.4 points. The
  earlier 21% came from the leaky loss and an optimiser that was unstable for this
  configuration. Nothing after the cutoff is used, but the held-out countries' data up to
  the cutoff do inform the shared emission factors, normalisation and static features.
- **Models trained on data up to 2005–2009 stall mature economies' efficiency gains.**
  For 12 mature economies (the United Kingdom, United States, Germany, France, Italy,
  Japan and six smaller European ones), the validation models forecast an
  energy-intensity trend of −0.8 to +0.6%/yr. Realised trends were −1.8 to −2.5%/yr, and
  trends before the cutoff −2.4 to −2.8%/yr, at every cutoff and seed. So these models
  over-predict those countries' final-year emissions by 15–30% per seed (trend-Kaya by
  1–2%). These 12 of the 60 countries make up most of the models' per-country gap to
  trend-Kaya on the validation folds: 63–126% of it by cutoff and seed, where more than
  100% means the other 48 together beat trend-Kaya. At the 2009 cutoff the damped
  momentum carries only about −1.1%/yr, and the network's correction adds +0.2 to
  +0.35%/yr where it subtracts for every other country
  (`results/tables/transfer_diagnostics.json`).
  - Step by step, that correction is a few single-year swings of +2.6 to +6.4%/yr
    (`scripts/stall_mechanism.py`, `results/tables/stall_mechanism.json`).
  - The 2005 and 2007 models swing at 2008–2009. They treat the recession as if energy
    use were sticky, when in fact it fell faster than GDP.
  - The 2009 models swing at 2010–2011, through the fuel-share momentum inputs.
  - Each swing enters the momentum, the model's only trend memory. The damping learned
    from so few years (0.58–0.78, against 0.87 on full data) cannot rebuild it within six
    years.
  - Three fixes aimed at this failed their rules written down beforehand (`CLAUDE.md`):
    - one common input scale for all fuels' momenta;
    - momentum that network corrections cannot enter;
    - a network blind to same-year GDP growth and the share momenta, whose swings move to
      the level inputs instead.

  Configurations with calendar-time inputs avoid the stall in these backtests, but only
  by extrapolating calendar time, so they do not fix it.
  - In the backtests, all seven configurations without the inputs forecast these
    economies' trend at −0.2 to −1.1%/yr, and all nine with them at −1.5 to −2.7%/yr,
    against −2.1%/yr realised (`scripts/stall_by_config.py`, which reads the validation
    outputs and fits nothing; `results/tables/stall_by_config.json`, means over the
    cutoffs and two seeds).
  - Calendar time is the year's index over the panel's span: 0 in 1990, 1 in 2022. A
    backtest feeds it past the cutoff, up to 1.3–1.4 times the largest value the model was
    trained on; planning holds it at its 2022 value.
  - Held at the cutoff's value in the backtests too, `ms_percap_damped` (the selected
    configuration plus time inputs, seeds 0–3) forecasts these economies' trend at
    −0.7%/yr, like `ms_full`'s −0.6%/yr; with calendar time running on it is −1.9%/yr.
    Held, it scores 12.57 against `ms_full`'s 11.54 (10.70 running on;
    `scripts/time_inputs.py`, `results/tables/time_inputs.json`, four-seed sweep outputs
    in `results/sweep_val_4seeds/`).
  - In planning, fitted on all the data, its business as usual lets energy intensity fall
    by only 0.5%/yr in these economies and 0.7%/yr in the other 48 over 2022–2032. The
    selected configuration: 2.0 and 1.1%/yr; observed 2012–2022: 2.2 and 1.3%/yr. Its
    business as usual reaches 55.8 Gt in 2050, against 39.9 Gt.
  - With calendar time running on to 2050, up to 1.875, the trend returns, but business
    as usual falls to 28.0 Gt.

  The time inputs tie the trend to the calendar rather than to the state of each
  economy. The bias fades with more data:
  - The 2015 test-fold models forecast the 12 transfer countries, four of them mature
    economies, about as well as trend-Kaya (7.9% against 7.6%).
  - The planning ensemble's BAU keeps most of these economies' trend: −1.9%/yr for
    2022–2032, against −2.2%/yr in 2012–2022.
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
