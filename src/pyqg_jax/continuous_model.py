# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


"""A continuous-stratification quasigeostrophic model.

Unlike the layered models, this model resolves a continuous vertical
coordinate with a general stratification profile :math:`N^2(z)` using
Chebyshev collocation. The prognostic state holds the interior
potential vorticity at the interior collocation points together with
the surface and bottom buoyancy at the boundary points; the inversion
solves the vertical boundary-value problem at each horizontal
wavenumber.
"""

__all__ = ["ContinuousQGModel"]


import jax
import jax.numpy as jnp
from . import _model, _utils, _chebyshev, state as _state


@_utils.register_pytree_class_attrs(
    children=["beta", "U", "N2"],
    static_attrs=["H"],
)
class ContinuousQGModel(_model.Model):
    r"""Continuous-stratification quasigeostrophic model.

    The vertical is discretized with Chebyshev collocation on
    ``z in [-H, 0]``. The prognostic field ``q`` has shape
    ``(nz, ny, nx)``: the interior entries (``q[1:-1]``) are the interior
    potential vorticity and the boundary entries (``q[0]``, ``q[-1]``)
    are the surface and bottom buoyancy. The inversion solves

    .. math::

       \left[\partial_z\!\left(\tfrac{f_0^2}{N^2}\partial_z\right)
       - \kappa^2\right]\hat\psi = \hat q \quad(\text{interior}),
       \qquad f_0\,\partial_z\hat\psi = \hat b \quad(\text{boundaries}),

    per horizontal wavenumber :math:`\kappa`.

    .. versionadded:: 0.9.0

    Parameters
    ----------
    nx : int, optional
        Number of grid points in the `x` direction.

    ny : int, optional
        Number of grid points in the `y` direction. Defaults to `nx`.

    nz : int, optional
        Number of vertical Chebyshev collocation points (``>= 3``).

    L : float, optional
        Domain length in `x`. Units: :math:`\mathrm{m}`.

    W : float, optional
        Domain length in `y`. Defaults to `L`.

    H : float, optional
        Total depth. The vertical grid spans ``z in [-H, 0]``. This is a
        static attribute.

    f : float, optional
        Coriolis parameter.

    beta : float, optional
        Gradient of the Coriolis parameter.

    N2 : float or array, optional
        Buoyancy frequency squared :math:`N^2(z)`. A scalar (uniform
        stratification) or an array of length `nz` sampled at the
        collocation points.

    U : float or array, optional
        Background zonal velocity profile :math:`U(z)`, length `nz`.
        Defaults to zero.

    rek : float, optional
        Linear bottom drag.

    filterfac : float, optional
        Spectral cutoff filter amplitude.

    precision : Precision, optional
        Computation precision.

    Note
    ----
    The inversion is performed in 64-bit precision regardless of
    `precision` (the vertical operator can be poorly conditioned).
    Enable JAX 64-bit support.
    """

    def __init__(
        self,
        *,
        nx=64,
        ny=None,
        nz=24,
        L=1e6,
        W=None,
        H=4000.0,
        f=1e-4,
        g=9.81,
        beta=0.0,
        N2=1e-4,
        U=None,
        rek=0.0,
        filterfac=23.6,
        precision=_state.Precision.SINGLE,
    ):
        if nz < 3:
            raise ValueError("ContinuousQGModel requires nz >= 3")
        super().__init__(
            nz=nz,
            nx=nx,
            ny=ny,
            L=L,
            W=W,
            rek=rek,
            filterfac=filterfac,
            f=f,
            g=g,
            precision=precision,
        )
        self.H = H
        self.beta = beta
        dtype = self.precision.dtype_real
        self.N2 = jnp.broadcast_to(jnp.asarray(N2, dtype=dtype), (nz,))
        if U is None:
            U = jnp.zeros((nz,), dtype=dtype)
        self.U = jnp.broadcast_to(jnp.asarray(U, dtype=dtype), (nz,))

    @property
    def z(self):
        """The vertical collocation points (descending from 0 to -H)."""
        return _chebyshev.cheb_grid_dz(self.nz, self.H)[0]

    @property
    def _Dz(self):
        return _chebyshev.cheb_grid_dz(self.nz, self.H)[1]

    @property
    def _interior_mask(self):
        m = jnp.ones((self.nz,), dtype=jnp.float64)
        return m.at[0].set(0.0).at[-1].set(0.0)

    def _Lvert64(self):
        # d_z((f^2/N^2) d_z) on the Chebyshev grid, in float64
        Dz = self._Dz.astype(jnp.float64)
        f2n2 = (jnp.float64(self.f) ** 2) / self.N2.astype(jnp.float64)
        return Dz @ (f2n2[:, jnp.newaxis] * Dz)

    @property
    def Hi(self):
        # trapezoidal vertical weights on the (descending) grid; sum == H
        z = self.z.astype(self.precision.dtype_real)
        edges = (z[:-1] + z[1:]) / 2  # midpoints, length nz-1
        upper = jnp.concatenate([z[:1], edges])
        lower = jnp.concatenate([edges, z[-1:]])
        return upper - lower  # positive, sums to H

    @property
    def Ubg(self):
        return self.U.astype(self.precision.dtype_real)

    @property
    def Qy(self):
        # generalized background gradient: interior beta - d_z((f^2/N^2)d_z)U;
        # boundaries -f dU/dz (the background buoyancy gradient, thermal wind)
        Lvert = self._Lvert64()
        Dz = self._Dz.astype(jnp.float64)
        U = self.U.astype(jnp.float64)
        f0 = jnp.float64(self.f)
        interior = jnp.float64(self.beta) - (Lvert @ U)
        boundary = -f0 * (Dz @ U)
        mask = self._interior_mask
        qy = mask * interior + (1 - mask) * boundary
        return qy.astype(self.precision.dtype_real)

    @property
    def ikQy(self):
        return jnp.expand_dims(self.Qy, (-1, -2)) * 1j * self.k

    @property
    def ilQx(self):
        return 0.0

    def create_initial_state(self, key):
        """Create a new state with small random initialization.

        Parameters
        ----------
        key : jax.random.key
            PRNG key for the random initialization.

        Returns
        -------
        PseudoSpectralState
        """
        q = 1e-7 * jax.random.normal(
            key,
            shape=(self.nz, self.ny, self.nx),
            dtype=self.precision.dtype_real,
        )
        q = q - q.mean(axis=(-2, -1), keepdims=True)
        return super().create_initial_state().update(q=q)

    def _apply_a_ph(self, state):
        n = self.nz
        qh = state.qh.astype(jnp.complex128)
        f0 = jnp.float64(self.f)
        Dz = self._Dz.astype(jnp.float64)
        Lvert = self._Lvert64()
        # base operator with Neumann buoyancy boundary rows (kappa-independent)
        base = Lvert.at[0, :].set(f0 * Dz[0, :]).at[-1, :].set(f0 * Dz[-1, :])
        eye_int = jnp.diag(self._interior_mask)
        wv2 = self.wv2.astype(jnp.float64)  # (nl, nk)
        # L[l, k] = base - kappa^2 * (identity on interior rows)
        L = base - jnp.expand_dims(wv2, (-1, -2)) * eye_int
        singular = wv2 == 0  # mean mode: replace with identity, zero result
        L = jnp.where(
            jnp.expand_dims(singular, (-1, -2)), jnp.eye(n, dtype=jnp.float64), L
        ).astype(jnp.complex128)
        rhs = jnp.expand_dims(jnp.moveaxis(qh, 0, -1), -1)  # (nl, nk, nz, 1)
        ph = jnp.linalg.solve(L, rhs)[..., 0]  # (nl, nk, nz)
        ph = jnp.where(jnp.expand_dims(singular, -1), 0.0, ph)
        ph = jnp.moveaxis(ph, -1, 0)  # (nz, nl, nk)
        return ph.astype(self.precision.dtype_complex)
