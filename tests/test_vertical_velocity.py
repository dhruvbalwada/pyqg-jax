# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import pytest
import numpy as np
import jax
import jax.numpy as jnp
import pyqg_jax
from pyqg_jax import diagnostics


def _model(nz=3, **kwargs):
    H = jnp.linspace(300.0, 1000.0, nz)
    base = {
        "nx": 48,
        "nz": nz,
        "f": 1e-4,
        "beta": 1.5e-11,
        "rek": 0.0,
        "U": jnp.linspace(0.06, 0.0, nz),
        "H": H,
        "rho": 1025.0 + jnp.linspace(0.0, 0.9, nz),
        "precision": pyqg_jax.state.Precision.DOUBLE,
    }
    if nz == 2:
        # the two-layer stretching matrix is built from rd/delta, so delta
        # must be consistent with the layer thicknesses for mass conservation
        base["delta"] = float(H[0] / H[1])
    base.update(kwargs)
    return pyqg_jax.layered_model.LayeredModel(**base)


def _random_state(model, seed=3, amp=1e-5):
    key = jax.random.key(seed)
    q = amp * jax.random.normal(key, (model.nz, model.ny, model.nx))
    q = q - q.mean(axis=(-2, -1), keepdims=True)
    return model.create_initial_state(key).update(q=q)


def test_shape_and_finite():
    model = _model(nz=3)
    full = model.get_full_state(_random_state(model))
    w = diagnostics.vertical_velocity(model, full)
    assert w.shape == (model.nz - 1, model.ny, model.nx)
    assert np.all(np.isfinite(np.asarray(w)))


@pytest.mark.parametrize("nz", [2, 3, 4])
def test_closes_at_bottom_boundary(nz):
    # the depth-integrated omega balance must close so that w = 0 at the
    # bottom; recompute the would-be bottom value and check it is ~0 vs |w|
    model = _model(nz=nz)
    state = _random_state(model)
    full = model.get_full_state(state)
    w = np.asarray(diagnostics.vertical_velocity(model, full))
    # reconstruct the bottom-boundary residual from the layer sources
    ph = full.ph.astype(jnp.complex128)
    dph = model.get_full_state(model.get_updates(state)).ph.astype(jnp.complex128)
    S = jnp.asarray(model.S, dtype=jnp.float64)
    Hi = np.asarray(model.Hi)
    Ubg = np.asarray(model.Ubg)
    f0 = model.f

    def irfft(a):
        return np.asarray(jnp.fft.irfftn(a, axes=(-2, -1), s=(model.ny, model.nx)))

    str_h = jnp.einsum("ij,jlk->ilk", S, ph)
    dstr_h = jnp.einsum("ij,jlk->ilk", S, dph)
    s_ubg = np.asarray(S @ Ubg)
    u = np.asarray(full.u)
    v = np.asarray(full.v)
    str_x = irfft(1j * model.k * str_h)
    str_y = irfft(1j * model.l * str_h)
    dstr = irfft(dstr_h)
    d_str = (
        dstr + (u + Ubg[:, None, None]) * str_x + v * str_y - v * s_ubg[:, None, None]
    )
    rhs = -d_str
    # full vertical integral (over all layers) should return to ~0 at the bottom
    bottom = -np.cumsum(Hi[:, None, None] * rhs / f0, axis=0)[-1]
    assert np.abs(bottom).std() / np.abs(w).std() < 1e-10


@pytest.mark.parametrize("nz", [2, 3, 4])
def test_mass_conservation(nz):
    # no net vertical mass flux: depth-integrated source vanishes
    model = _model(nz=nz)
    state = _random_state(model)
    full = model.get_full_state(state)
    # the diagnostic closing at the bottom (above) is the operative check;
    # here also confirm each interface w has ~zero horizontal mean
    w = np.asarray(diagnostics.vertical_velocity(model, full))
    assert np.abs(w.mean(axis=(-2, -1))).max() < 1e-12 * np.abs(w).max()


def test_barotropic_state_zero():
    # a depth-independent streamfunction has no stretching -> no w
    # (requires no background flow: barotropic eddies over a sloping
    # background interface would otherwise produce w)
    model = _model(nz=3, U=jnp.zeros(3))
    key = jax.random.key(1)
    psi = 1e-2 * jax.random.normal(key, (model.ny, model.nx))
    psi_h = jnp.fft.rfftn(psi, axes=(-2, -1))
    # q = (-wv2) psi in every layer (stretching cancels for uniform psi)
    qh = jnp.stack([-model.wv2 * psi_h] * model.nz)
    q = jnp.fft.irfftn(qh, axes=(-2, -1), s=(model.ny, model.nx))
    state = model.create_initial_state(key).update(q=q)
    full = model.get_full_state(state)
    # confirm the streamfunction really is depth-independent
    assert float(jnp.abs(full.ph - full.ph[0]).max()) < 1e-10
    w = diagnostics.vertical_velocity(model, full)
    assert float(jnp.abs(w).max()) < 1e-12


def test_grad_through_vertical_velocity():
    def loss(beta):
        model = _model(nz=3, beta=beta)
        full = model.get_full_state(_random_state(model))
        return jnp.sum(diagnostics.vertical_velocity(model, full) ** 2)

    g = jax.grad(loss)(1.5e-11)
    assert np.isfinite(float(g))


def test_requires_stretching_matrix():
    model = pyqg_jax.sqg_model.SQGModel(precision=pyqg_jax.state.Precision.DOUBLE)
    full = model.get_full_state(model.create_initial_state(jax.random.key(0)))
    with pytest.raises(TypeError, match="stretching matrix"):
        diagnostics.vertical_velocity(model, full)
