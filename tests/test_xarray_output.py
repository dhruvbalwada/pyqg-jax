# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import pytest
import numpy as np
import jax
import pyqg_jax

pytest.importorskip("xarray")


def _model():
    return pyqg_jax.qg_model.QGModel(
        nx=32, ny=24, precision=pyqg_jax.state.Precision.DOUBLE
    )


def test_partial_state_dataset():
    model = _model()
    state = model.create_initial_state(jax.random.key(0))
    ds = pyqg_jax.xarray_output.state_to_dataset(state, model)
    # partial state has q and qh
    assert "q" in ds
    assert ds["q"].dims == ("lev", "y", "x")
    assert ds["q"].shape == (model.nz, model.ny, model.nx)
    np.testing.assert_array_equal(np.asarray(ds["q"].values), np.asarray(state.q))
    assert "qh" in ds
    assert ds["qh"].dims == ("lev", "l", "k")


def test_full_state_dataset():
    model = _model()
    state = model.create_initial_state(jax.random.key(0))
    full = model.get_full_state(state)
    ds = pyqg_jax.xarray_output.state_to_dataset(full, model)
    for name in ("q", "u", "v", "p"):
        assert name in ds
        assert ds[name].dims == ("lev", "y", "x")
    for name in ("qh", "ph"):
        assert name in ds
        assert ds[name].dims == ("lev", "l", "k")
    np.testing.assert_allclose(np.asarray(ds["u"].values), np.asarray(full.u))


def test_coordinates_and_attrs():
    model = _model()
    full = model.get_full_state(model.create_initial_state(jax.random.key(0)))
    ds = pyqg_jax.xarray_output.state_to_dataset(full, model)
    assert ds.sizes["x"] == model.nx
    assert ds.sizes["y"] == model.ny
    assert ds.sizes["lev"] == model.nz
    # cell-centered x coordinate
    np.testing.assert_allclose(
        ds["x"].values, (np.arange(model.nx) + 0.5) / model.nx * model.L
    )
    np.testing.assert_allclose(ds["k"].values, np.asarray(model.kk))
    assert ds.attrs["nx"] == model.nx
    assert ds.attrs["L"] == model.L


def test_layered_model_dataset():
    model = pyqg_jax.layered_model.LayeredModel(
        nx=16, nz=3, f=1e-4, precision=pyqg_jax.state.Precision.DOUBLE
    )
    full = model.get_full_state(model.create_initial_state(jax.random.key(0)))
    ds = pyqg_jax.xarray_output.state_to_dataset(full, model)
    assert ds["q"].shape == (3, model.ny, model.nx)
    assert ds.attrs["nz"] == 3
