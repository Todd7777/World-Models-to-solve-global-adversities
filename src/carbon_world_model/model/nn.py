"""Minimal JAX MLP and parameter initialisation (no flax dependency).

Parameters are a plain nested dict (a JAX pytree), so optax and jax.grad handle them
directly and they serialise trivially.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _glorot(key, shape):
    fan_in, fan_out = shape[0], shape[1]
    lim = np.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)


def init_mlp(key, sizes: list[int]):
    """List of (W, b) for a fully-connected stack with the given layer sizes."""
    layers = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, din, dout in zip(keys, sizes[:-1], sizes[1:]):
        layers.append({"w": _glorot(k, (din, dout)), "b": jnp.zeros(dout)})
    return layers


def mlp_apply(layers, x, activation=jax.nn.silu):
    """Apply an MLP; activation on all but the last layer."""
    h = x
    for i, layer in enumerate(layers):
        h = h @ layer["w"] + layer["b"]
        if i < len(layers) - 1:
            h = activation(h)
    return h
