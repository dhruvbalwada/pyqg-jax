# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import pytest
import numpy as np
import jax
import jax.numpy as jnp
from pyqg_jax.particles import GriddedParticleStepper, ParticleState

L = 1.0e6
NX = 32


def _grid_centers(n, length):
    return (np.arange(n) + 0.5) * (length / n)


def test_interpolate_constant_field():
    stepper = GriddedParticleStepper(L=L)
    field = jnp.full((NX, NX), 3.5)
    state = stepper.initialize_state(
        jnp.array([1.0e5, 7.3e5]), jnp.array([2.0e5, 5.1e5])
    )
    out = stepper.interpolate(field, state)
    assert np.allclose(np.asarray(out), 3.5)


def test_interpolate_at_cell_centers():
    # sampling exactly at cell centers returns the grid values
    stepper = GriddedParticleStepper(L=L)
    rng = np.random.RandomState(0)
    field = jnp.asarray(rng.randn(NX, NX))
    xc = _grid_centers(NX, L)
    # pick a few cell centers
    ii = np.array([3, 10, 25])
    jj = np.array([1, 17, 30])
    state = stepper.initialize_state(jnp.asarray(xc[ii]), jnp.asarray(xc[jj]))
    out = np.asarray(stepper.interpolate(field, state))
    assert np.allclose(out, np.asarray(field)[jj, ii], atol=1e-10)


def test_interpolate_linear_field_exact():
    # bilinear interpolation is exact for a field linear in x
    stepper = GriddedParticleStepper(L=L)
    xc = _grid_centers(NX, L)
    field = jnp.asarray(np.broadcast_to(xc, (NX, NX)).copy())
    px = jnp.array([1.234e5, 5.0e5, 8.9e5])
    py = jnp.array([2.0e5, 6.1e5, 3.3e5])
    state = stepper.initialize_state(px, py)
    out = np.asarray(stepper.interpolate(field, state))
    # away from the periodic seam the linear field reproduces x exactly
    assert np.allclose(out, np.asarray(px), atol=1e-6)


def test_uniform_flow_advection():
    # a spatially uniform velocity advects particles by u * (n * dt) exactly
    stepper = GriddedParticleStepper(L=L)
    u = jnp.full((NX, NX), 0.2)
    v = jnp.full((NX, NX), -0.1)
    state = stepper.initialize_state(jnp.array([5.0e5]), jnp.array([5.0e5]))
    dt = 100.0
    nsteps = 20
    for _ in range(nsteps):
        state = stepper.step(state, u, v, u, v, dt)
    exp_x = (5.0e5 + 0.2 * dt * nsteps) % L
    exp_y = (5.0e5 - 0.1 * dt * nsteps) % L
    assert np.allclose(float(state.x[0]), exp_x, atol=1e-4)
    assert np.allclose(float(state.y[0]), exp_y, atol=1e-4)


def test_periodic_wrapping():
    stepper = GriddedParticleStepper(L=L)
    state = stepper.initialize_state(jnp.array([-1.0e5]), jnp.array([1.1e6]))
    # positions are wrapped into [0, L) on initialization
    assert 0 <= float(state.x[0]) < L
    assert 0 <= float(state.y[0]) < L
    assert np.allclose(float(state.x[0]), L - 1.0e5)
    assert np.allclose(float(state.y[0]), 1.1e6 - L)


def test_match_mainline_trajectory():
    pytest.importorskip("pyqg.particles")
    from pyqg.particles import GriddedLagrangianParticleArray2D

    rng = np.random.RandomState(3)
    X, Y = np.meshgrid(_grid_centers(NX, L), _grid_centers(NX, L))

    def smooth(seed):
        f = np.zeros((NX, NX))
        r = np.random.RandomState(seed)
        for _ in range(5):
            kx = r.randint(1, 4) * 2 * np.pi / L
            ky = r.randint(1, 4) * 2 * np.pi / L
            f += r.uniform(-1, 1) * np.sin(kx * X + ky * Y + r.uniform(0, 6.28))
        return 0.1 * f / np.abs(f).max()

    U0, V0, U1, V1 = smooth(1), smooth(2), smooth(3), smooth(4)
    x0 = rng.uniform(0, L, size=64)
    y0 = rng.uniform(0, L, size=64)
    dt = 3600.0
    nsteps = 30

    parts = GriddedLagrangianParticleArray2D(
        x0,
        y0,
        NX,
        NX,
        periodic_in_x=True,
        periodic_in_y=True,
        xmin=0.0,
        xmax=L,
        ymin=0.0,
        ymax=L,
    )
    for _ in range(nsteps):
        parts.step_forward_with_gridded_uv(U0, V0, U1, V1, dt, order=1)

    stepper = GriddedParticleStepper(L=L)
    state = stepper.initialize_state(jnp.asarray(x0), jnp.asarray(y0))
    for _ in range(nsteps):
        state = stepper.step(
            state,
            jnp.asarray(U0),
            jnp.asarray(V0),
            jnp.asarray(U1),
            jnp.asarray(V1),
            dt,
        )

    assert np.allclose(np.asarray(state.x), parts.x, atol=1e-6)
    assert np.allclose(np.asarray(state.y), parts.y, atol=1e-6)


def test_grad_and_vmap():
    stepper = GriddedParticleStepper(L=L)
    rng = np.random.RandomState(1)
    u = jnp.asarray(rng.randn(NX, NX) * 0.1)
    v = jnp.asarray(rng.randn(NX, NX) * 0.1)
    y0 = jnp.asarray(rng.uniform(0, L, size=16))

    def final_x_sum(x0):
        state = stepper.initialize_state(x0, y0)
        state = stepper.step(state, u, v, u, v, 3600.0)
        return jnp.sum(state.x)

    g = jax.grad(final_x_sum)(jnp.asarray(rng.uniform(0, L, size=16)))
    assert np.all(np.isfinite(np.asarray(g)))

    # vmap over an ensemble of particle sets
    x0s = jnp.asarray(rng.uniform(0, L, size=(8, 16)))
    y0s = jnp.asarray(rng.uniform(0, L, size=(8, 16)))

    def run(x0, y0):
        state = stepper.initialize_state(x0, y0)
        state = stepper.step(state, u, v, u, v, 3600.0)
        return state.x

    out = jax.vmap(run)(x0s, y0s)
    assert out.shape == (8, 16)
    assert np.all(np.isfinite(np.asarray(out)))


def test_tree_flatten_roundtrip():
    stepper = GriddedParticleStepper(L=L, W=2 * L, periodic_in_x=False)
    leaves, treedef = jax.tree_util.tree_flatten(stepper)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.L == stepper.L
    assert restored.W == stepper.W
    assert restored.periodic_in_x == stepper.periodic_in_x
    assert restored.periodic_in_y == stepper.periodic_in_y


def test_state_update_rejects_unknown():
    state = ParticleState(x=jnp.zeros(3), y=jnp.zeros(3))
    with pytest.raises(ValueError, match="invalid state updates"):
        state.update(z=jnp.zeros(3))
