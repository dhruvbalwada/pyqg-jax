# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


"""Lagrangian particle advection on the model grid.

This module provides tools to advect passive Lagrangian particles
using a gridded velocity field, as in :class:`pyqg.LagrangianParticleArray2D`
and :class:`pyqg.GriddedLagrangianParticleArray2D`.

The velocities are assumed to live at cell centers (an Arakawa "A"
grid), matching the real-space velocities produced by the models in
this package. Interpolation is bilinear and the particles are advanced
with the same two-time-level Runge-Kutta scheme used by PyQG.

Everything here is written in pure JAX, so the advection is
differentiable, :func:`jax.jit`-compatible, and can be mapped over
ensembles of particle sets with :func:`jax.vmap`.
"""

__all__ = ["ParticleState", "GriddedParticleStepper"]


import dataclasses
import jax
import jax.numpy as jnp
from . import _utils


@_utils.register_pytree_dataclass
@dataclasses.dataclass(frozen=True)
class ParticleState:
    """The positions of a set of Lagrangian particles.

    Warning
    -------
    You should not usually construct this class yourself. Instead
    obtain instances from :meth:`GriddedParticleStepper.initialize_state`.

    Attributes
    ----------
    x : jax.Array
        Particle positions along the `x` axis. Units: :math:`\\mathrm{m}`.

    y : jax.Array
        Particle positions along the `y` axis. Units: :math:`\\mathrm{m}`.
    """

    x: jax.Array
    y: jax.Array

    def update(self, **kwargs):
        """Produce a *new* state with the specified values replaced.

        The keyword arguments may be `x` or `y`.

        Returns
        -------
        ParticleState
            A copy of this object with the specified values replaced.
        """
        if extra_attrs := (kwargs.keys() - {"x", "y"}):
            extra_attr_str = ", ".join(extra_attrs)
            raise ValueError(
                f"invalid state updates, can only update x and y (not {extra_attr_str})"
            )
        return dataclasses.replace(self, **kwargs)

    def __repr__(self):
        x_summary = _utils.summarize_object(self.x)
        y_summary = _utils.summarize_object(self.y)
        return f"ParticleState(x={x_summary}, y={y_summary})"


@_utils.register_pytree_class_attrs(
    children=["L", "W"],
    static_attrs=["periodic_in_x", "periodic_in_y"],
)
class GriddedParticleStepper:
    r"""Advect Lagrangian particles using a gridded velocity field.

    The velocity fields are assumed to be defined at cell centers on a
    regular grid spanning :math:`[0, L) \times [0, W)`, matching the
    real-space velocities of the models in this package. Interpolation
    to the particle positions is bilinear.

    Parameters
    ----------
    L : float
        Domain length in the `x` direction. Units: :math:`\mathrm{m}`.

    W : float, optional
        Domain length in the `y` direction. Defaults to `L`.
        Units: :math:`\mathrm{m}`.

    periodic_in_x : bool, optional
        Whether the domain wraps in the `x` direction. Defaults to
        :pycode:`True`.

    periodic_in_y : bool, optional
        Whether the domain wraps in the `y` direction. Defaults to
        :pycode:`True`.

    Attributes
    ----------
    L, W : float
        Domain lengths.
    """

    def __init__(self, *, L, W=None, periodic_in_x=True, periodic_in_y=True):
        self.L = L
        self.W = W if W is not None else L
        self.periodic_in_x = periodic_in_x
        self.periodic_in_y = periodic_in_y

    def initialize_state(self, x0, y0) -> ParticleState:
        """Wrap initial particle positions in a :class:`ParticleState`.

        Parameters
        ----------
        x0, y0 : array-like
            Two arrays (broadcast to the same shape) of initial
            particle positions.

        Returns
        -------
        ParticleState
            The new particle state.
        """
        x = jnp.asarray(x0)
        y = jnp.asarray(y0)
        x, y = jnp.broadcast_arrays(x, y)
        return ParticleState(x=self._wrap_x(x), y=self._wrap_y(y))

    def _wrap_x(self, x):
        if self.periodic_in_x:
            return jnp.mod(x, self.L)
        return x

    def _wrap_y(self, y):
        if self.periodic_in_y:
            return jnp.mod(y, self.W)
        return y

    def interpolate(self, field, state: ParticleState) -> jax.Array:
        """Bilinearly interpolate a gridded `field` to the particles.

        Parameters
        ----------
        field : jax.Array
            A real-space field with its final two axes corresponding to
            the `y` and `x` grid axes (shape :pycode:`(..., ny, nx)`).

        state : ParticleState
            The particle positions at which to sample `field`.

        Returns
        -------
        jax.Array
            The interpolated values, of shape
            :pycode:`field.shape[:-2] + state.x.shape`.
        """
        return self._interpolate(field, state.x, state.y)

    def _interpolate(self, field, x, y):
        ny, nx = field.shape[-2:]
        # cell centers sit at (i + 0.5) * dx, so the fractional index of
        # a point is x / dx - 0.5
        fi = (x / self.L) * nx - 0.5
        fj = (y / self.W) * ny - 0.5
        i0 = jnp.floor(fi)
        j0 = jnp.floor(fj)
        wi = fi - i0
        wj = fj - j0
        i0 = i0.astype(jnp.int32)
        j0 = j0.astype(jnp.int32)
        i1 = i0 + 1
        j1 = j0 + 1
        if self.periodic_in_x:
            i0 = jnp.mod(i0, nx)
            i1 = jnp.mod(i1, nx)
        else:
            i0 = jnp.clip(i0, 0, nx - 1)
            i1 = jnp.clip(i1, 0, nx - 1)
        if self.periodic_in_y:
            j0 = jnp.mod(j0, ny)
            j1 = jnp.mod(j1, ny)
        else:
            j0 = jnp.clip(j0, 0, ny - 1)
            j1 = jnp.clip(j1, 0, ny - 1)
        f00 = field[..., j0, i0]
        f01 = field[..., j0, i1]
        f10 = field[..., j1, i0]
        f11 = field[..., j1, i1]
        return (
            f00 * (1 - wi) * (1 - wj)
            + f01 * wi * (1 - wj)
            + f10 * (1 - wi) * wj
            + f11 * wi * wj
        )

    def step(self, state: ParticleState, u0, v0, u1, v1, dt) -> ParticleState:
        """Advance particles one step with a two-time-level RK4 scheme.

        This matches the time stepping of
        :meth:`pyqg.GriddedLagrangianParticleArray2D.step_forward_with_gridded_uv`:
        the first Runge-Kutta stage uses the velocity at the current
        time, and the remaining three stages use the velocity at the
        next time.

        Parameters
        ----------
        state : ParticleState
            The current particle positions.

        u0, v0 : jax.Array
            Gridded velocity components at the current time, shape
            :pycode:`(ny, nx)`.

        u1, v1 : jax.Array
            Gridded velocity components at the next time (a step `dt`
            later), shape :pycode:`(ny, nx)`.

        dt : float
            Time step. Units: :math:`\\mathrm{sec}`.

        Returns
        -------
        ParticleState
            The advected particle positions, a new object.
        """
        x, y = state.x, state.y

        def uv0(px, py):
            return self._interpolate(u0, px, py), self._interpolate(v0, px, py)

        def uv1(px, py):
            return self._interpolate(u1, px, py), self._interpolate(v1, px, py)

        ku0, kv0 = uv0(x, y)
        k1u, k1v = dt * ku0, dt * kv0
        ku1, kv1 = uv1(self._wrap_x(x + 0.5 * k1u), self._wrap_y(y + 0.5 * k1v))
        k2u, k2v = dt * ku1, dt * kv1
        ku2, kv2 = uv1(self._wrap_x(x + 0.5 * k2u), self._wrap_y(y + 0.5 * k2v))
        k3u, k3v = dt * ku2, dt * kv2
        ku3, kv3 = uv1(self._wrap_x(x + k3u), self._wrap_y(y + k3v))
        k4u, k4v = dt * ku3, dt * kv3
        dx = (k1u + 2 * k2u + 2 * k3u + k4u) / 6
        dy = (k1v + 2 * k2v + 2 * k3v + k4v) / 6
        return state.update(x=self._wrap_x(x + dx), y=self._wrap_y(y + dy))

    def __repr__(self):
        return _utils.auto_repr(self)
