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
python -m pytest -q   # 2026-11-10: 55 pass (python -m: a standalone `pytest` may use another interpreter)
```

## State on 2026-11-10

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
- Sweep caches (`results/sweep_*/*/*.npz`) are git-ignored; only `report.json` files are committed.
- Branch `claude/clever-fermi-rm4429` holds a separate Space Solar World Model (`src/space_solar_world_model/`).
  It split from this branch at `ed4c64f` and has not been merged.

## Where the next steps are

README "Limitations (read these)", especially the first two: a way to constrain or test how the dynamics
respond to controls, and checks of long-run levels (not just smoothness). Untried: tuning w_stab beyond
{10, 100}, more seeds for the top configs, and why shoot-only training transfers worse.
