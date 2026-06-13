# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import warnings
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import pyqg_jax
from pyqg_jax import diagnostics

CONFIG = {
    "nx": 48,
    "nz": 3,
    "f": 1e-4,
    "beta": 1.5e-11,
    "rd": 15000.0,
    "U": [0.05, 0.025, 0.0],
    "H": [400.0, 600.0, 1000.0],
    "rho": [1025.0, 1025.4, 1026.2],
}


def _model():
    kw = dict(CONFIG)
    for key in ("U", "H", "rho"):
        kw[key] = jnp.asarray(kw[key])
    return pyqg_jax.layered_model.LayeredModel(
        precision=pyqg_jax.state.Precision.DOUBLE, **kw
    )


def _band_limited_state(model, kmax=8, seed=0, amp=1e-5):
    key = jax.random.key(seed)
    q = amp * jax.random.normal(key, (model.nz, model.ny, model.nx))
    qh = jnp.fft.rfftn(q, axes=(-2, -1))
    ky = np.abs(np.fft.fftfreq(model.nx) * model.nx)
    kh = np.hypot(ky[:, None], np.arange(model.nx // 2 + 1)[None, :])
    qh = jnp.where(jnp.asarray(kh) > kmax, 0.0, qh)
    q = jnp.fft.irfftn(qh, axes=(-2, -1), s=(model.ny, model.nx))
    q = q - q.mean(axis=(-2, -1), keepdims=True)
    return model.create_initial_state(key).update(q=q)


def _depth_weighted_total(model, vals):
    Hi = np.asarray(model.Hi)
    H = float(model.get_grid().H)
    return (Hi[:, None, None] * np.asarray(vals)).sum(0) / H


def test_shapes_and_finite():
    model = _model()
    full = model.get_full_state(_band_limited_state(model))
    for fn in (diagnostics.ke_flux_spec_vals, diagnostics.ape_flux_spec_vals):
        out = fn(model, full)
        assert out.shape == (model.nz, model.nl, model.nk)
        assert np.all(np.isfinite(np.asarray(out)))


@pytest.mark.parametrize(
    "fn", [diagnostics.ke_flux_spec_vals, diagnostics.ape_flux_spec_vals]
)
def test_transfer_conserves_energy(fn):
    # for alias-free (band-limited) fields the nonlinear transfer is
    # conservative: the depth-integrated spectral transfer sums to zero
    model = _model()
    full = model.get_full_state(_band_limited_state(model, kmax=8))
    flux = _depth_weighted_total(model, fn(model, full))
    mult = np.ones((model.nl, model.nk))
    mult[:, 1:-1] = 2.0  # rfft: count the full annulus (conjugate half)
    total = abs((mult * flux).sum())
    assert total / np.abs(flux).max() < 1e-9


def test_ape_flux_requires_stretching():
    model = pyqg_jax.sqg_model.SQGModel(precision=pyqg_jax.state.Precision.DOUBLE)
    full = model.get_full_state(model.create_initial_state(jax.random.key(0)))
    with pytest.raises(TypeError, match="stretching"):
        diagnostics.ape_flux_spec_vals(model, full)


def test_works_with_calc_ispec():
    model = _model()
    full = model.get_full_state(_band_limited_state(model))
    vals = diagnostics.ke_flux_spec_vals(model, full)
    spec = diagnostics.calc_ispec(vals, model.get_grid())
    assert spec.shape[0] == model.nz
    assert np.all(np.isfinite(np.asarray(spec)))


def test_matches_pyqg():
    pyqg = pytest.importorskip("pyqg")
    model = _model()
    state = _band_limited_state(model, kmax=12)
    full = model.get_full_state(state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        orig = pyqg.LayeredModel(log_level=0, **CONFIG)
    orig.set_q(np.asarray(state.q).astype(np.float64))
    orig._invert()
    xi = orig.ifft(-orig.wv2 * orig.ph)
    jpxi = orig._advect(xi, orig.u, orig.v)
    sp = orig.ifft(np.einsum("ij,jkl->ikl", orig.S, orig.ph))
    jsp = orig._advect(sp, orig.u, orig.v)
    ke_ref = (
        (orig.Hi[:, None, None] * (orig.ph.conj() * jpxi).real).sum(0)
        / orig.H
        / orig.M**2
    )
    ape_ref = (
        (orig.Hi[:, None, None] * (orig.ph.conj() * jsp).real).sum(0)
        / orig.H
        / orig.M**2
    )
    ke = _depth_weighted_total(model, diagnostics.ke_flux_spec_vals(model, full))
    ape = _depth_weighted_total(model, diagnostics.ape_flux_spec_vals(model, full))
    assert np.allclose(ke, ke_ref, atol=0, rtol=1e-8)
    assert np.allclose(ape, ape_ref, atol=0, rtol=1e-8)
