# Carbon World Model (CWM)

A differentiable, mechanistic-neural **world model of the global energy–emissions
system**, calibrated to real data for 60 countries, built to answer the question
that ordinary forecasts cannot:

> *Given how energy systems actually evolve, what is the cheapest feasible path to
> net-zero — and how much warming does it avoid?*

CWM couples a **physics-grounded emissions layer** (CO₂ reconstructed from per-fuel
energy use via emission factors) with a **learned dynamics model** of how national
energy demand and fuel mix evolve. Because the whole simulator is differentiable, we
can *plan inside it*: optimise decarbonisation pathways by gradient descent, subject
to feasibility limits taken from the fastest transitions humanity has ever achieved.

> **Status:** research prototype. See `README` "Results" and `results/` for what is
> validated and what is not. This is a transparent scientific model, not a forecast
> of the future or policy advice.

Data: [Our World in Data](https://github.com/owid/co2-data) CO₂ and energy datasets
(CC-BY). Run `bash scripts/download_data.sh` to fetch them.

Full documentation, methods, and results are filled in as the model is built.
