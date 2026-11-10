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
python -m pytest -q   # 2026-11-10: 51 pass (python -m: a standalone `pytest` may use another interpreter)
```

## State on 2026-11-10

- All tests pass. The checkpoint `results/model/params_full.pkl` is the selected config `ms_percap_damped`
  (`eval/configs.SELECTED`, chosen on validation folds only), and `params_full.json` records that. Scripts
  rebuild the model from that record (`train.fit.load_trained`). Retrain with `scripts/02_train.py`
  whenever the dynamics change; `tests/test_checkpoint.py` fails otherwise.
- README numbers and `results/tables/` were regenerated on 2026-11-10 from the leak-free protocol.
  Trend-Kaya still beats the model on aggregate CO2 (validation, test and transfer).
- Main open problem: planning results depend on structure the data do not identify. Configs tied on
  validation disagree widely beyond 6 years (`scripts/07_structural.py`, `results/tables/structural.json`),
  the fitted dynamics partly undo the planner's controls (`plan/diagnostics.py`; open-loop counterfactual
  in `plan/optimize.max_ambition_open_loop`), and the selected model's free run shows a learned ~12-year
  oscillation in China's intensity. Report ranges, not single-model planning numbers.
- Sweep caches (`results/sweep_*/*/*.npz`) are git-ignored; only `report.json` files are committed.
- Branch `claude/clever-fermi-rm4429` holds a separate Space Solar World Model (`src/space_solar_world_model/`).
  It split from this branch at `ed4c64f` and has not been merged.

## Where the next steps are

README "Limitations (read these)", especially the first two: long-horizon stability/plausibility
constraints in training, and a way to constrain how the dynamics respond to controls.
