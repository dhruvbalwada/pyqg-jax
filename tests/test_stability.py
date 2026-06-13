# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import math
import warnings
import pytest
import numpy as np
import pyqg_jax


def test_shapes():
    model = pyqg_jax.layered_model.LayeredModel(
        nx=32, nz=3, f=1e-4, precision=pyqg_jax.state.Precision.DOUBLE
    )
    omega, evec = model.stability_analysis()
    assert omega.shape == (model.nl, model.nk)
    assert evec.shape == (model.nz, model.nl, model.nk)
    assert np.all(np.isfinite(np.asarray(omega)))


def test_callies_matches_paper():
    # Callies et al. (2016) Fig 5a: mixed-layer instability peaks near
    # 0.31 f Lambda_m / N_m ~ 1.55e-6 1/s at ~9 km wavelength
    model = pyqg_jax.callies_model.CalliesTwoEady(
        nx=256, L=5e5, dt=1.0, precision=pyqg_jax.state.Precision.DOUBLE
    )
    omega, _ = model.stability_analysis()
    sigma = np.asarray(omega).imag
    ll = np.asarray(model.l)
    kk = np.asarray(model.k)
    # along l = 0
    mask = ll == 0
    sig0 = sigma[mask]
    k0 = kk[mask]
    peak = np.argmax(sig0)
    growth = sig0[peak]
    wavelength_km = 2 * np.pi / k0[peak] / 1e3
    eady = 0.31 * model.f * model.Sm / model.Nm
    assert math.isclose(growth, eady, rel_tol=0.15)
    assert 6.0 < wavelength_km < 13.0


def test_single_layer_no_instability():
    # a one-layer SQG model has no baroclinic instability
    model = pyqg_jax.sqg_model.SQGModel(precision=pyqg_jax.state.Precision.DOUBLE)
    omega, _ = model.stability_analysis()
    assert np.abs(np.asarray(omega).imag).max() < 1e-20


def test_matches_pyqg_qg():
    pyqg = pytest.importorskip("pyqg")
    jax_model = pyqg_jax.qg_model.QGModel(
        nx=64, precision=pyqg_jax.state.Precision.DOUBLE
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        orig = pyqg.QGModel(nx=64, log_level=0)
    ev = orig.stability_analysis()
    o_ref = (ev[0] if isinstance(ev, tuple) else ev).imag
    omega, _ = jax_model.stability_analysis()
    sj = np.asarray(omega).imag
    assert np.allclose(sj, o_ref, atol=0, rtol=1e-8)
