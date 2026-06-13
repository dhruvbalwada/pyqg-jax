# Copyright 2023 Karl Otness, pyqg developers
# SPDX-License-Identifier: MIT


r"""Stochastic ring forcing of potential vorticity.

Forces PV with vertically-uniform white noise confined to a ring of
wavenumbers :math:`k_{in} < \sqrt{k_x^2 + k_y^2} < k_{out}`, decorrelated
in time, as in `Uchida et al. (2023) <https://doi.org/10.31223/X5C063>`__.

See also: :class:`pyqg.RingForcing`
"""

__all__ = [
    "apply_parameterization",
    "param_func",
    "init_param_aux_func",
]


import jax
import jax.numpy as jnp
from . import _defs, _parameterized_model
from .. import state as _state


def apply_parameterization(
    model, *, k_in_forc=0.0, k_out_forc=0.0, mag_noise_forc=0.0, layers="all", key=None
):
    """Apply stochastic ring forcing to `model`.

    See also: :class:`pyqg.RingForcing`

    Parameters
    ----------
    model
        The inner model to wrap in the forcing.

    k_in_forc : float, optional
        Inner wavenumber of the forcing ring.

    k_out_forc : float, optional
        Outer wavenumber of the forcing ring.

    mag_noise_forc : float, optional
        Amplitude of the forcing.

    layers : {"all", "surf", "bottom"}, optional
        Which layers to force. Defaults to forcing every layer equally.

    key : jax.random.key, optional
        PRNG key seeding the stochastic forcing. Defaults to
        :pycode:`jax.random.key(0)`.

    Returns
    -------
    ParameterizedModel
        `model` wrapped in the forcing.
    """
    if layers not in ("all", "surf", "bottom"):
        raise ValueError("layers must be one of 'all', 'surf', 'bottom'")
    if key is None:
        key = jax.random.key(0)
    return _parameterized_model.ParameterizedModel(
        model=model,
        param_func=jax.tree_util.Partial(
            param_func,
            k_in_forc=k_in_forc,
            k_out_forc=k_out_forc,
            mag_noise_forc=mag_noise_forc,
            layers=layers,
        ),
        init_param_aux_func=jax.tree_util.Partial(init_param_aux_func, key=key),
    )


@_defs.q_parameterization
def param_func(
    state, param_aux, model, *, k_in_forc, k_out_forc, mag_noise_forc, layers="all"
):
    key, subkey = jax.random.split(param_aux)
    grid = model.get_grid()
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    wvx = jnp.sqrt(model.k**2 + model.l**2)
    mask = jnp.where((wvx > k_in_forc) & (wvx <= k_out_forc), 1.0, 0.0)
    ka, kb = jax.random.split(subkey)
    ring_hat = mask * (
        jax.random.normal(ka, wvx.shape, dtype=model.precision.dtype_real)
        + 1j * jax.random.normal(kb, wvx.shape, dtype=model.precision.dtype_real)
    )
    ring = _state._generic_irfftn(
        jnp.expand_dims(ring_hat, 0), shape=grid.real_state_shape
    )  # (1, ny, nx)
    ring = ring - jnp.mean(ring, axis=(-1, -2), keepdims=True)
    denom = jnp.mean(jnp.abs(ring), axis=(-1, -2), keepdims=True)
    ring = ring / jnp.where(denom == 0, 1, denom)
    forcing = mag_noise_forc * ring[0]  # (ny, nx)
    if layers == "all":
        dq = jnp.broadcast_to(forcing, (nz, ny, nx))
    elif layers == "surf":
        dq = jnp.zeros((nz, ny, nx), dtype=forcing.dtype).at[0].set(forcing)
    else:  # "bottom"
        dq = jnp.zeros((nz, ny, nx), dtype=forcing.dtype).at[-1].set(forcing)
    return dq, key


def init_param_aux_func(state, model, *, key=None):
    if key is None:
        key = jax.random.key(0)
    return key
