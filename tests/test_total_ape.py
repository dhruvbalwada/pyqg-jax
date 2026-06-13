# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import numpy as np
import jax
import jax.numpy as jnp
import pyqg_jax
from pyqg_jax import diagnostics


def _model(nz=3):
    return pyqg_jax.layered_model.LayeredModel(
        nx=48,
        nz=nz,
        f=1e-4,
        beta=1.5e-11,
        U=jnp.linspace(0.05, 0.0, nz),
        H=jnp.linspace(400.0, 1000.0, nz),
        rho=1025.0 + jnp.linspace(0.0, 1.2, nz),
        precision=pyqg_jax.state.Precision.DOUBLE,
    )


def _state(model, seed=0, amp=1e-5):
    key = jax.random.key(seed)
    q = amp * jax.random.normal(key, (model.nz, model.ny, model.nx))
    q = q - q.mean(axis=(-2, -1), keepdims=True)
    return model.create_initial_state(key).update(q=q)


def test_matches_realspace_definition():
    model = _model()
    full = model.get_full_state(_state(model))
    ape = float(diagnostics.total_ape(full, model.get_grid(), model.f, model.gpi))
    # independent real-space computation of the interface-displacement PE
    p = np.asarray(full.p)
    gpi = np.asarray(model.gpi)
    H = float(model.get_grid().H)
    hand = (model.f**2 / (2 * H)) * sum(
        ((p[k] - p[k + 1]) ** 2).mean() / gpi[k] for k in range(model.nz - 1)
    )
    assert np.isclose(ape, hand)


def test_matches_spectral_parseval():
    # the real-space variance must equal the spectral sum (Parseval),
    # accounting for the rfft layout (double the interior columns)
    model = _model()
    full = model.get_full_state(_state(model))
    ape = float(diagnostics.total_ape(full, model.get_grid(), model.f, model.gpi))
    ph = np.asarray(full.ph)
    gpi = np.asarray(model.gpi)
    H = float(model.get_grid().H)
    M2 = (model.nx * model.ny) ** 2
    mult = np.ones((model.nl, model.nk))
    mult[:, 1:-1] = 2.0
    spectral = 0.0
    for k in range(model.nz - 1):
        var = (np.abs(ph[k] - ph[k + 1]) ** 2 * mult).sum() / M2
        spectral += var / gpi[k]
    spectral *= model.f**2 / (2 * H)
    assert np.isclose(ape, spectral, rtol=1e-8)


def test_positive():
    model = _model()
    full = model.get_full_state(_state(model))
    assert float(diagnostics.total_ape(full, model.get_grid(), model.f, model.gpi)) > 0


def test_zero_for_barotropic():
    # depth-independent streamfunction -> no interface displacement -> no APE
    model = _model()
    key = jax.random.key(1)
    psi = 1e-2 * jax.random.normal(key, (model.ny, model.nx))
    psi_h = jnp.fft.rfftn(psi, axes=(-2, -1))
    qh = jnp.stack([-model.wv2 * psi_h] * model.nz)
    q = jnp.fft.irfftn(qh, axes=(-2, -1), s=(model.ny, model.nx))
    full = model.get_full_state(model.create_initial_state(key).update(q=q))
    ape = float(diagnostics.total_ape(full, model.get_grid(), model.f, model.gpi))
    assert abs(ape) < 1e-18


def test_vmap_over_batch():
    model = _model()
    keys = jax.random.split(jax.random.key(2), 5)
    states = jax.vmap(lambda k: _state_from_key(model, k))(keys)
    fulls = jax.vmap(model.get_full_state)(states)
    apes = jax.vmap(
        lambda fs: diagnostics.total_ape(fs, model.get_grid(), model.f, model.gpi)
    )(fulls)
    assert apes.shape == (5,)
    assert np.all(np.asarray(apes) > 0)


def _state_from_key(model, key):
    q = 1e-5 * jax.random.normal(key, (model.nz, model.ny, model.nx))
    q = q - q.mean(axis=(-2, -1), keepdims=True)
    return model.create_initial_state(key).update(q=q)
