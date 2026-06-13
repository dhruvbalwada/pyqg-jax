# Copyright 2024 Karl Otness
# SPDX-License-Identifier: MIT


"""Functions for computing diagnostics of simulation states.

The functions in this module can be used to compute diagnostic
quantities such as kinetic energy or various spectra. See
:doc:`examples.diagnostics` for examples of how to use them and how to
plot the results.
"""

import jax.numpy as jnp
from . import _spectral, state as _state

__all__ = [
    "total_ke",
    "total_ape",
    "cfl",
    "ke_spec_vals",
    "ens_spec_vals",
    "ispec_grid",
    "calc_ispec",
    "vertical_velocity",
    "ke_flux_spec_vals",
    "ape_flux_spec_vals",
]


def _getattr_shape_check(full_state, attr, grid):
    if attr in {"q", "p", "u", "v", "dqdt"}:
        corr_shape = grid.real_state_shape
    else:
        corr_shape = grid.spectral_state_shape
    corr_dims = len(corr_shape)
    arr = getattr(full_state, attr)
    shape = jnp.shape(arr)
    dims = len(shape)
    if dims != corr_dims:
        vmap_msg = " (use jax.vmap)" if dims > corr_dims else ""
        raise ValueError(
            f"{attr} has {dims} dimensions but, should have {corr_dims}{vmap_msg}"
        )
    if shape != corr_shape:
        raise ValueError(f"{attr} has wrong shape {shape}, should be {corr_shape}")
    return arr


def _grid_shape_check(grid, attr):
    val = getattr(grid, attr)
    if attr in {
        "nz",
        "ny",
        "nx",
        "nl",
        "nk",
        "real_state_shape",
        "spectral_state_shape",
    }:
        # Static attributes couldn't be JAX-transformed
        return val
    shape = jnp.shape(val)
    if attr == "Hi":
        if shape != (grid.nz,):
            # Check Hi shape
            raise ValueError(
                f"grid.Hi should be a 1D sequence of length {grid.nz} "
                f"but had shape {shape}"
            )
    elif len(shape) != 0:
        # Everything else must be a scalar
        raise ValueError(
            f"grid.{attr} has {len(shape)} dimensions, but should have 0 (use jax.vmap)"
        )
    return val


def total_ke(full_state, grid):
    """Compute the total kinetic energy in a single snapshot.

    The density in the KE calculation is taken such that the entire
    model grid space has a mass of one unit. To use a different
    density value, multiply the result of this calculation by the
    total mass of the full space.

    .. versionadded:: 0.8.0

    Parameters
    ----------
    full_state : FullPseudoSpectralState
        The state for which the kinetic energy is to be computed. This
        argument should be retrieved from a model, for example from
        :meth:`~pyqg_jax.qg_model.QGModel.get_full_state`.

        This function only operates on a single time step. To apply it
        to a trajectory use :func:`jax.vmap`.

    grid : Grid
        Information on the spatial grid for `full_state`. This should
        be retrieved from a model, for example from
        :meth:`~pyqg_jax.qg_model.QGModel.get_grid`.

    Returns
    -------
    float
        The total kinetic energy for the provided snapshot.
    """
    u = _getattr_shape_check(full_state, "u", grid)
    v = _getattr_shape_check(full_state, "v", grid)
    ke = (u**2 + v**2) / 2
    H = _grid_shape_check(grid, "H")
    Hi = _grid_shape_check(grid, "Hi")
    h_weights = jnp.expand_dims(Hi / H, axis=(-1, -2))
    return jnp.mean(jnp.sum(ke * h_weights, axis=-3), axis=(-1, -2))


def total_ape(full_state, grid, f, gpi):
    r"""Compute the total available potential energy in a single snapshot.

    For a layered model the available potential energy is stored in the
    displacement of the interfaces between layers,

    .. math::

       \mathrm{APE} = \frac{1}{2H} \sum_k \frac{f_0^2}{g'_k}
       \left\langle (\psi_k - \psi_{k+1})^2 \right\rangle,

    where :math:`g'_k` is the reduced gravity at interface `k`,
    :math:`\langle\cdot\rangle` is a horizontal average, and the
    normalization (per total depth :math:`H`, with the factor of one
    half) matches :func:`total_ke`, so :pycode:`total_ke + total_ape` is
    the total energy per unit mass.

    .. versionadded:: 0.9.0

    Parameters
    ----------
    full_state : FullPseudoSpectralState
        The state for which the APE is to be computed, for example from
        :meth:`~pyqg_jax.layered_model.LayeredModel.get_full_state`.
        This function operates on a single time step; use
        :func:`jax.vmap` for a trajectory.

    grid : Grid
        Information on the spatial grid for `full_state`, for example
        from :meth:`~pyqg_jax.layered_model.LayeredModel.get_grid`.

    f : float
        The Coriolis parameter :math:`f_0`.

    gpi : array-like
        The reduced gravities :math:`g'_k` at each of the ``nz - 1``
        interfaces, for example from
        :attr:`~pyqg_jax.layered_model.LayeredModel.gpi`.

    Returns
    -------
    float
        The total available potential energy for the provided snapshot.
    """
    p = _getattr_shape_check(full_state, "p", grid)
    H = _grid_shape_check(grid, "H")
    gpi = jnp.asarray(gpi)
    dpsi = p[..., :-1, :, :] - p[..., 1:, :, :]
    layer_ape = jnp.mean(dpsi**2, axis=(-1, -2)) / gpi
    return (f**2 / (2 * H)) * jnp.sum(layer_ape, axis=-1)


def cfl(full_state, grid, ubg, dt):
    """Calculate the CFL condition value for a single snapshot.

    This computes the `CFL
    <https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition>`__
    condition value at each grid point in a given state. To report the
    worst value across the full state, aggregate the values using
    :func:`jnp.max <jax.numpy.max>`.

    .. versionadded:: 0.8.0

    Parameters
    ----------
    full_state : FullPseudoSpectralState
        The state for which the CFL condition is to be checked. This
        argument should be retrieved from a model, for example from
        :meth:`~pyqg_jax.qg_model.QGModel.get_full_state`.

        This function only operates on a single time step. To apply it
        to a trajectory use :func:`jax.vmap`.

    grid : Grid
        Information on the spatial grid for `full_state`. This should
        be retrieved from a model, for example from
        :meth:`~pyqg_jax.qg_model.QGModel.get_grid`.

    ubg : jax.Array
        The model's background velocity. Retrieve it from the same
        model as `full_state`, for example from
        :attr:`~pyqg_jax.qg_model.QGModel.Ubg`.

    dt : float
        The time step size. This should be retrieved from the relevant
        time stepper, for example from
        :attr:`~pyqg_jax.steppers.AB3Stepper.dt`.

    Returns
    -------
    jax.Array
        The CFL condition value at each spatial grid location. These
        may optionally be aggregated with :func:`jnp.max
        <jax.numpy.max>`.
    """
    u = jnp.abs(
        _getattr_shape_check(full_state, "u", grid)
        + jnp.expand_dims(ubg, axis=(-1, -2))
    ) / _grid_shape_check(grid, "dy")
    v = jnp.abs(_getattr_shape_check(full_state, "v", grid)) / _grid_shape_check(
        grid, "dx"
    )
    return dt * (u + v)


def ke_spec_vals(full_state, grid):
    """Calculate the kinetic energy spectrum values for a snapshot.

    The values produced by this function should be further processed
    by :func:`calc_ispec` to produce the kinetic energy spectrum.

    .. versionadded:: 0.8.0

    Parameters
    ----------
    full_state : FullPseudoSpectralState
        The state for which the KE spectrum values should be computed.
        This argument should be retrieved from a model, for example
        from :meth:`~pyqg_jax.qg_model.QGModel.get_full_state`.

        This function only operates on a single time step. To apply it
        to a trajectory use :func:`jax.vmap`.

    grid : Grid
        Information on the spatial grid for `full_state`. This should
        be retrieved from a model, for example from
        :meth:`~pyqg_jax.qg_model.QGModel.get_grid`.

    Returns
    -------
    jax.Array
        The KE spectrum values for the provided time step.

    Note
    ----
    The returned array should be treated as opaque. Values should only
    be averaged over any vmapped time dimensions, then passed to
    :func:`calc_ispec`.
    """
    ph = _getattr_shape_check(full_state, "ph", grid)
    M = _grid_shape_check(grid, "nx") * _grid_shape_check(grid, "ny")
    abs_ph = jnp.abs(ph)
    kappa = grid.get_kappa(abs_ph.dtype)
    if kappa.shape != grid.spectral_state_shape[1:]:
        raise ValueError(
            f"grid kappa array has unexpected shape {kappa.shape}, "
            f"should be {grid.spectral_state_shape[1:]}"
        )
    return kappa**2 * abs_ph**2 / M**2


def ens_spec_vals(full_state, grid):
    """Calculate the enstrophy spectrum values for a snapshot.

    The values produced by this function should be further processed
    by :func:`calc_ispec` to produce the kinetic energy spectrum.

    .. versionadded:: 0.9.0

    Parameters
    ----------
    full_state : FullPseudoSpectralState
        The state for which the enstropy spectrum values should be
        computed. This argument should be retrieved from a model, for
        example from :meth:`~pyqg_jax.qg_model.QGModel.get_full_state`.

        This function only operates on a single time step. To apply it
        to a trajectory use :func:`jax.vmap`.

    grid : Grid
        Information on the spatial grid for `full_state`. This should
        be retrieved from a model, for example from
        :meth:`~pyqg_jax.qg_model.QGModel.get_grid`.

    Returns
    -------
    jax.Array
        The enstropy spectrum values for the provided time step.

    Note
    ----
    The returned array should be treated as opaque. Values should only
    be averaged over any vmapped time dimensions, then passed to
    :func:`calc_ispec`.
    """
    qh = _getattr_shape_check(full_state, "qh", grid)
    M = _grid_shape_check(grid, "nx") * _grid_shape_check(grid, "ny")
    return jnp.abs(qh) ** 2 / M**2


def vertical_velocity(model, full_state):
    r"""Diagnose the quasigeostrophic vertical velocity at layer interfaces.

    In quasigeostrophy the vertical velocity does not appear in the
    leading-order dynamics; it is the next-order diagnostic given by the
    QG omega balance. This function computes it from the vortex-stretching
    (vorticity) form of the omega equation,

    .. math::

       f_0 \frac{\partial w}{\partial z} = -\frac{D_g}{Dt}(\mathsf{S}\psi),

    where :math:`\mathsf{S}\psi` is the vortex stretching (the part of the
    potential vorticity that is not relative vorticity) and :math:`D_g/Dt`
    is the material derivative following the full geostrophic flow
    (eddy plus background). Discretized over layers with a rigid lid and
    flat bottom (:math:`w = 0` at the top and bottom boundaries) this is
    integrated vertically to give the vertical velocity at each interior
    interface.

    This formulation uses the model's own stretching operator, so the
    depth-integrated balance closes to roundoff and the recovered
    :math:`w` vanishes at the bottom boundary (no spurious net vertical
    mass flux). It is the route ``(a)`` quantity for estimating
    submesoscale vertical velocity. It is implemented for layered models
    that expose a stretching matrix (:class:`~pyqg_jax.layered_model.LayeredModel`).

    .. versionadded:: 0.9.0

    Parameters
    ----------
    model
        The model that produced `full_state`. It must expose the
        stretching matrix as ``S`` (shape ``(nz, nz)``), the background
        velocity ``Ubg``, the Coriolis parameter ``f``, and the spectral
        wavenumber grids ``k`` and ``l`` (as
        :class:`~pyqg_jax.layered_model.LayeredModel` does).

    full_state : FullPseudoSpectralState
        The expanded state, for example from
        :meth:`~pyqg_jax.layered_model.LayeredModel.get_full_state`.
        This function operates on a single time step; use
        :func:`jax.vmap` for a trajectory.

    Returns
    -------
    jax.Array
        The vertical velocity at each of the ``nz - 1`` interior
        interfaces, in real space, with shape :pycode:`(nz - 1, ny, nx)`.
    """
    S = getattr(model, "S", None)
    if S is None:
        raise TypeError(
            "vertical_velocity requires a layered model exposing a stretching"
            " matrix as `S` (e.g. LayeredModel)"
        )
    grid = model.get_grid()
    ny, nx = grid.ny, grid.nx
    ph = full_state.ph.astype(jnp.complex128)
    # streamfunction tendency: invert the model's (advective) PV tendency.
    # The inversion is linear, so this is the streamfunction's d/dt.
    dph = model.get_full_state(model.get_updates(full_state.state)).ph.astype(
        jnp.complex128
    )
    S = jnp.asarray(S, dtype=jnp.float64)
    f0 = jnp.float64(model.f)
    Hi = jnp.asarray(grid.Hi, dtype=jnp.float64)
    Ubg = jnp.asarray(model.Ubg, dtype=jnp.float64)
    ik = jnp.expand_dims(1j * model.k, 0)
    il = jnp.expand_dims(1j * model.l, 0)

    def irfft(field_h):
        return jnp.fft.irfftn(field_h, axes=(-2, -1), s=(ny, nx))

    # vortex stretching (spectral) and its tendency, via the stretching matrix
    str_h = jnp.einsum("ij,jlk->ilk", S, ph)
    dstr_h = jnp.einsum("ij,jlk->ilk", S, dph)
    s_ubg = jnp.expand_dims(S @ Ubg, (-1, -2))  # background stretching gradient term
    u = full_state.u  # eddy velocities
    v = full_state.v
    str_x = irfft(ik * str_h)
    str_y = irfft(il * str_h)
    dstr = irfft(dstr_h)
    # material derivative of the (total) stretching following the full flow:
    #   D_g(str)/Dt = d_t str + (u + Ubg) str_x + v str_y - v (S Ubg)
    d_str_dt = (
        dstr + (u + jnp.expand_dims(Ubg, (-1, -2))) * str_x + v * str_y - v * s_ubg
    )
    # f0 dw/dz = -D_g(str)/Dt; integrate from the top (w = 0 at top & bottom)
    contrib = jnp.expand_dims(Hi, (-1, -2)) * (-d_str_dt) / f0
    w = -jnp.cumsum(contrib, axis=0)[:-1]  # interior interfaces; last (bottom) ~ 0
    return w.astype(model.precision.dtype_real)


def ke_flux_spec_vals(model, full_state):
    r"""Kinetic-energy spectral transfer (flux) values for a snapshot.

    Returns the per-layer spectral kinetic-energy transfer by the
    nonlinear advection of relative vorticity,

    .. math::

       \mathrm{Re}\!\left[\hat\psi^{*}\,
       \widehat{\nabla\cdot(\mathbf{u}\,\nabla^2\psi)}\right] / M^2,

    one of the terms in the spectral energy budget. As with the other
    spectral diagnostics, average the result over any vmapped time
    dimension and process it with :func:`calc_ispec`; for the total
    (depth-integrated) flux, weight the per-layer isotropic spectra by
    :pycode:`Hi / H` and sum over layers.

    .. versionadded:: 0.9.0

    Parameters
    ----------
    model
        The model that produced `full_state` (used for the wavenumber
        grids and the FFTs).

    full_state : FullPseudoSpectralState
        The expanded state, for example from
        :meth:`~pyqg_jax.layered_model.LayeredModel.get_full_state`.
        Operates on a single time step; use :func:`jax.vmap` for a
        trajectory.

    Returns
    -------
    jax.Array
        The (signed) per-layer KE transfer values, shape
        :pycode:`(nz, nl, nk)`.
    """
    grid = model.get_grid()
    ph = full_state.ph
    u = full_state.u
    v = full_state.v
    ik = jnp.expand_dims(1j * model.k, 0)
    il = jnp.expand_dims(1j * model.l, 0)
    M = grid.nx * grid.ny
    xi = _state._generic_irfftn(-model.wv2 * ph, shape=grid.real_state_shape)
    adv = ik * _state._generic_rfftn(u * xi) + il * _state._generic_rfftn(v * xi)
    return (jnp.conj(ph) * adv).real / M**2


def ape_flux_spec_vals(model, full_state):
    r"""Available-potential-energy spectral transfer (flux) values.

    Returns the per-layer spectral available-potential-energy transfer
    by the nonlinear advection of the vortex stretching
    :math:`\mathsf{S}\psi`,

    .. math::

       \mathrm{Re}\!\left[\hat\psi^{*}\,
       \widehat{\nabla\cdot(\mathbf{u}\,\mathsf{S}\psi)}\right] / M^2,

    the companion of :func:`ke_flux_spec_vals` in the spectral energy
    budget. Process with :func:`calc_ispec` and combine across layers
    with the :pycode:`Hi / H` weighting as for the KE flux.

    Requires a model exposing a stretching matrix ``S``
    (:class:`~pyqg_jax.layered_model.LayeredModel`).

    .. versionadded:: 0.9.0

    Parameters
    ----------
    model
        The model that produced `full_state`. Must expose the stretching
        matrix as ``S``.

    full_state : FullPseudoSpectralState
        The expanded state. Operates on a single time step; use
        :func:`jax.vmap` for a trajectory.

    Returns
    -------
    jax.Array
        The (signed) per-layer APE transfer values, shape
        :pycode:`(nz, nl, nk)`.
    """
    S = getattr(model, "S", None)
    if S is None:
        raise TypeError(
            "ape_flux_spec_vals requires a layered model exposing a stretching"
            " matrix as `S` (e.g. LayeredModel)"
        )
    grid = model.get_grid()
    ph = full_state.ph
    u = full_state.u
    v = full_state.v
    ik = jnp.expand_dims(1j * model.k, 0)
    il = jnp.expand_dims(1j * model.l, 0)
    M = grid.nx * grid.ny
    sph = jnp.einsum("ij,jlk->ilk", jnp.asarray(S, dtype=ph.dtype), ph)
    sp = _state._generic_irfftn(sph, shape=grid.real_state_shape)
    adv = ik * _state._generic_rfftn(u * sp) + il * _state._generic_rfftn(v * sp)
    return (jnp.conj(ph) * adv).real / M**2


def ispec_grid(grid):
    """Information on the spacing of values in an isotropic spectrum.

    This function produces two results: `iso_k` and `keep`. The values
    `iso_k` are the isotropic wavenumbers for each entry in the result
    of `calc_ispec`. The result `keep` is an integer which should be
    used to slice the result of `calc_ispec`. Only the first `keep`
    entries should be interpreted.

    The values computed by this function are useful when plotting the
    result of :func:`calc_ispec`.

    .. versionadded:: 0.8.0

    Parameters
    ----------
    grid : Grid
        The spatial grid over which the base values were defined. This
        should be retrieved from a model, for example from
        :meth:`~pyqg_jax.qg_model.QGModel.get_grid`.

    Returns
    -------
    iso_k : jax.Array
        The isotropic wavenumbers for each spectrum entry.

    keep : int
        An integer indicating how many of the first spectrum entries
        should be interpreted or plotted.
    """
    iso_k, keep = _spectral.get_plot_kr(grid, truncate=True)
    return iso_k, keep


def calc_ispec(spec_vals, grid):
    """Compute the isotropic spectrum from the given values.

    The array `spec_vals` should have been computed by one of the
    spectral diagnostics functions--for example :func:`ke_spec_vals`.

    To correctly plot or interpret the spectrum computed by this
    function, use the result of :func:`ispec_grid`.

    .. versionadded:: 0.8.0

    Parameters
    ----------
    spec_vals : jax.Array
        The input values which should be processed into an isotropic
        spectrum. These values should be a squared modulus of the
        Fourier coefficients.

    grid : Grid
        The spatial grid over which the base values were defined. This
        should be retrieved from a model, for example from
        :meth:`~pyqg_jax.qg_model.QGModel.get_grid`.

    Returns
    -------
    jax.Array
        A one-dimensional array providing the isotropic spectrum of
        `spec_vals`.
    """
    shape = spec_vals.shape
    corr_shape = grid.spectral_state_shape
    if shape != corr_shape:
        raise ValueError(
            f"mismatched shape for calc_ispec, expected {corr_shape} but got {shape}"
        )
    return _spectral.calc_ispec(spec_vals, grid, averaging=True, truncate=True)
