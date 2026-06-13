# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


"""Convert model states into labeled :class:`xarray.Dataset` objects.

This module provides :func:`state_to_dataset`, which packages the
fields of a model state together with spatial and spectral coordinates
into an :class:`xarray.Dataset` for analysis and serialization.

This module requires the optional dependency `xarray
<https://docs.xarray.dev/>`__.
"""

__all__ = ["state_to_dataset"]


import contextlib
import numpy as np


def _require_xarray():
    try:
        import xarray as xr
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "state_to_dataset requires the optional dependency 'xarray'"
            " (install it with `pip install xarray`)"
        ) from e
    return xr


def state_to_dataset(state, model):
    r"""Package a model state into an :class:`xarray.Dataset`.

    Real-space fields are given dimensions :pycode:`("lev", "y", "x")`
    and spectral fields :pycode:`("lev", "l", "k")`, with matching
    coordinates derived from the model grid. Whichever of the fields
    ``q``, ``qh``, ``u``, ``v``, ``p``, ``ph`` are present on `state`
    are included (so both a :class:`~pyqg_jax.state.PseudoSpectralState`
    and a :class:`~pyqg_jax.state.FullPseudoSpectralState` are
    accepted).

    .. versionadded:: 0.9.0

    Parameters
    ----------
    state : PseudoSpectralState or FullPseudoSpectralState
        The state to convert. This function operates on a single time
        step; to build a dataset for a trajectory, convert snapshots
        and combine them (for example with :func:`xarray.concat`).

    model
        The model that produced `state`, used for the grid coordinates
        and model parameters (stored as dataset attributes).

    Returns
    -------
    xarray.Dataset
        The state fields with labeled spatial/spectral coordinates.
    """
    xr = _require_xarray()
    grid = model.get_grid()
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    x = (np.arange(nx) + 0.5) / nx * float(grid.L)
    y = (np.arange(ny) + 0.5) / ny * float(grid.W)
    lev = np.arange(nz)
    kk = np.asarray(model.kk)
    ll = np.asarray(model.ll)

    coords = {
        "lev": ("lev", lev),
        "y": ("y", y),
        "x": ("x", x),
        "l": ("l", ll),
        "k": ("k", kk),
    }
    real_dims = ("lev", "y", "x")
    spec_dims = ("lev", "l", "k")
    real_fields = ("q", "u", "v", "p")
    spec_fields = ("qh", "ph")

    data_vars = {}
    for name in real_fields:
        val = getattr(state, name, None)
        if val is not None:
            arr = np.asarray(val)
            if arr.shape == (nz, ny, nx):
                data_vars[name] = (real_dims, arr)
    for name in spec_fields:
        val = getattr(state, name, None)
        if val is not None:
            arr = np.asarray(val)
            if arr.shape == (nz, grid.nl, grid.nk):
                data_vars[name] = (spec_dims, arr)

    attrs = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "L": float(grid.L),
        "W": float(grid.W),
    }
    for name in ("f", "g", "beta", "rek"):
        val = getattr(model, name, None)
        if val is not None:
            with contextlib.suppress(TypeError, ValueError):
                attrs[name] = float(val)

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
