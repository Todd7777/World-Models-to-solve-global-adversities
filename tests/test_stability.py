"""Stability term: definition, probe drivers, and that it is wired into training."""
import dataclasses

import jax.numpy as jnp
import numpy as np

from carbon_world_model.eval.configs import make_config
from carbon_world_model.model.dynamics import rollout
from carbon_world_model.model.world_model import WorldModel
from carbon_world_model.train.fit import (
    STAB_GDPPC_LONGRUN, fit, stability_batch, stability_loss,
)


def test_stability_loss_zero_for_constant_rates_positive_for_cycles():
    T, F = 31, 9
    const = {"dI_nat": jnp.full((4, T), -0.03), "ds_nat": jnp.full((4, T, F), 0.001)}
    assert float(stability_loss(const)) == 0.0
    t = jnp.arange(T)
    cyc = {"dI_nat": jnp.tile(0.05 * jnp.sin(2 * jnp.pi * t / 12), (4, 1)),
           "ds_nat": jnp.zeros((4, T, F))}
    assert float(stability_loss(cyc)) > 1e-4


def test_stability_batch_shapes_and_driver_path(panel, cfg):
    k = panel.n_years - 1
    wm = WorldModel.build(panel, cfg)
    b = stability_batch(wm, panel, k, H=30)
    C = panel.n_countries
    gdp, pop = np.asarray(b["drivers"]["gdp"]), np.asarray(b["drivers"]["pop"])
    assert gdp.shape == pop.shape == (C, 31) and np.isfinite(gdp).all()
    # step 0 is the observed cutoff year
    np.testing.assert_allclose(gdp[:, 0], panel.gdp[:, k], rtol=1e-6)
    np.testing.assert_allclose(pop[:, 0], panel.population[:, k], rtol=1e-6)
    # by the horizon end, per-capita growth has tapered to the long-run rate
    g_pc_end = np.log(gdp[:, -1] / pop[:, -1]) - np.log(gdp[:, -2] / pop[:, -2])
    np.testing.assert_allclose(g_pc_end, STAB_GDPPC_LONGRUN, atol=1e-6)
    assert np.allclose(np.asarray(b["tfrac"]), wm.tfrac(k))


def test_stability_term_is_reported_and_lowers_the_penalty(panel):
    base = make_config("ms_percap_damped", steps=60)
    stab = dataclasses.replace(base, train=dataclasses.replace(base.train, w_stab=100.0))
    r0 = fit(panel, base, seed=0, verbose=False)
    r1 = fit(panel, stab, seed=0, verbose=False)
    assert "stab" not in r0.history[-1] and "stab" in r1.history[-1]
    b = stability_batch(r0.wm, panel, panel.n_years - 1, H=30)

    def penalty(r):
        wm = r.wm
        tr = rollout(r.params, jnp.asarray(wm.static), wm.norm, jnp.asarray(wm.ef),
                     b["inits"], b["drivers"], None, wm.hp, tfrac=b["tfrac"])
        return float(stability_loss(tr))

    assert penalty(r1) < penalty(r0)
