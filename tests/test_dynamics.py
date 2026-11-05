"""World-model rollout invariants and differentiability."""
import jax
import jax.numpy as jnp
import numpy as np


def test_rollout_shapes_and_finite(wm, panel, init_params):
    tr = wm.rollout(init_params, panel)
    C, T, F = panel.energy.shape
    assert tr["co2"].shape == (C, T)
    assert tr["shares"].shape == (C, T, F)
    for v in tr.values():
        assert bool(jnp.isfinite(v).all())


def test_shares_stay_on_simplex(wm, panel, init_params):
    tr = wm.rollout(init_params, panel)
    s = jnp.sum(tr["shares"], axis=2)
    assert np.allclose(np.asarray(s), 1.0, atol=1e-4)
    assert float(jnp.min(tr["shares"])) >= -1e-6


def test_initial_state_reconstructs_panel(wm, panel, init_params):
    tr = wm.rollout(init_params, panel)
    assert np.allclose(np.asarray(tr["energy"][:, 0, :]), panel.energy[:, 0, :], rtol=1e-3)
    assert np.allclose(np.asarray(tr["D"][:, 0]), panel.primary_energy[:, 0], rtol=1e-4)


def test_gradients_finite(wm, panel, init_params):
    co2 = jnp.asarray(panel.co2)

    def loss(p):
        return jnp.mean((wm.rollout(p, panel)["co2"] - co2) ** 2)

    g = jax.grad(loss)(init_params)
    leaves = jax.tree_util.tree_leaves(g)
    assert all(bool(jnp.isfinite(x).all()) for x in leaves)
    assert sum(float(jnp.sum(x ** 2)) for x in leaves) > 0


def test_trained_insample_accuracy(wm, panel, params):
    tr = wm.rollout(params, panel)
    m = panel.mask.astype(bool)
    pred, true = np.asarray(tr["co2"])[m], panel.co2[m]
    r2 = 1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2)
    assert r2 > 0.98  # trained simulator reproduces history
