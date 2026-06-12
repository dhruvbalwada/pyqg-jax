# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


"""An implementation of :class:`pyqg.LayeredModel`."""

__all__ = ["LayeredModel"]


import jax
import jax.numpy as jnp
from . import _model, _utils, state as _state


@_utils.register_pytree_class_attrs(
    children=["beta", "rd", "delta", "U", "H", "rho"],
    static_attrs=[],
)
class LayeredModel(_model.Model):
    r"""Layered quasi-geostrophic model with general number of layers.

    See also :class:`pyqg.LayeredModel`.

    .. versionadded:: 0.9.0

    The potential vorticity anomalies :math:`q_i` relate to the
    streamfunctions :math:`\psi_i` through a tridiagonal stretching
    matrix :math:`\mathsf{S}`,

    .. math::

       \hat{q} = (\mathsf{S} - \kappa^2 \mathsf{I})\,\hat\psi,

    where :math:`\kappa^2` is the horizontal wavenumber magnitude
    squared. The stretching matrix is built from the layer thicknesses
    :math:`H_i` and the reduced gravities :math:`g'_i = g
    (\rho_{i+1} - \rho_i) / \rho_i`. The meridional background PV
    gradient is :math:`\mathsf{Q}_y = \beta - \mathsf{S}\,\mathsf{U}`.

    Parameters
    ----------
    nx : int, optional
        Number of grid points in the `x` direction.

    ny : int, optional
        Number of grid points in the `y` direction. Defaults to `nx`.

    nz : int, optional
        Number of layers (must be ``>= 2``).

    L : float, optional
        Domain length in the `x` direction. Units: :math:`\mathrm{m}`.

    W : float, optional
        Domain length in the `y` direction. Defaults to `L`.
        Units: :math:`\mathrm{m}`.

    rek : float, optional
        Linear drag in the bottom layer. Units: :math:`\mathrm{sec}^{-1}`.

    filterfac : float, optional
        Amplitude of the spectral spherical filter.

    f : float, optional
        Coriolis parameter. Units: :math:`\mathrm{sec}^{-1}`.

    g : float, optional
        Acceleration due to gravity. Units:
        :math:`\mathrm{m}\ \mathrm{sec}^{-2}`.

    beta : float, optional
        Gradient of the Coriolis parameter. Units:
        :math:`\mathrm{m}^{-1}\ \mathrm{sec}^{-1}`.

    rd : float, optional
        Deformation radius. Units: :math:`\mathrm{m}`. Only used in the
        two-layer (``nz == 2``) case.

    delta : float, optional
        Layer thickness ratio :math:`H_1 / H_2`. Unitless. Only used in
        the two-layer (``nz == 2``) case.

    U : array of size nz, optional
        Background zonal velocity in each layer. Units:
        :math:`\mathrm{m}\ \mathrm{sec}^{-1}`.

    H : array of size nz, optional
        Layer thicknesses. Units: :math:`\mathrm{m}`.

    rho : array of size nz, optional
        Layer densities. Units: :math:`\mathrm{kg}\ \mathrm{m}^{-3}`.
        Used for ``nz >= 3`` to build the stretching matrix.

    precision : Precision, optional
        Precision of model computation. Selects dtype of state values.

    Attributes
    ----------
    Ubg : jax.Array
        The background velocity for this model.

    Note
    ----
    Only a zonal background flow is supported (no meridional background
    ``V``), matching the pyqg-jax pseudo-spectral kernel, which carries
    only the :math:`\mathrm{i} k \mathsf{Q}_y` advection term.

    This model internally uses 64-bit floating point values for the
    inversion *regardless* of the chosen :class:`precision
    <pyqg_jax.state.Precision>`. Make sure that JAX has `64-bit
    precision enabled
    <https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision>`__.
    """

    def __init__(
        self,
        *,
        # grid size parameters
        nx=64,
        ny=None,
        nz=2,
        L=1e6,
        W=None,
        # friction parameters
        rek=5.787e-7,
        filterfac=23.6,
        # constants
        f=None,
        g=9.81,
        # Additional model parameters
        beta=1.5e-11,
        rd=15000.0,
        delta=0.25,
        U=None,
        H=None,
        rho=None,
        # Precision choice
        precision=_state.Precision.SINGLE,
    ):
        if nz < 2:
            raise ValueError("LayeredModel requires nz >= 2")
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
        self.beta = beta
        self.rd = rd
        self.delta = delta
        dtype_real = self.precision.dtype_real
        if U is None:
            U = (jnp.arange(nz) * 0.025)[::-1]
        if H is None:
            H = jnp.asarray([500.0] + [1750.0] * (nz - 1))
        if rho is None:
            rho = jnp.arange(nz) * 0.3 + 1025.0
        self.U = jnp.asarray(U, dtype=dtype_real)
        self.H = jnp.asarray(H, dtype=dtype_real)
        self.rho = jnp.asarray(rho, dtype=dtype_real)
        if self.U.shape != (nz,):
            raise ValueError(f"U must have shape ({nz},), got {self.U.shape}")
        if self.H.shape != (nz,):
            raise ValueError(f"H must have shape ({nz},), got {self.H.shape}")
        if self.rho.shape != (nz,):
            raise ValueError(f"rho must have shape ({nz},), got {self.rho.shape}")

    def create_initial_state(self, key):
        """Create a new initial state with random initialization.

        Parameters
        ----------
        key : jax.random.key
            The PRNG state used as the random key for initialization.

        Returns
        -------
        PseudoSpectralState
            The new state with random initialization.
        """
        q = 1e-7 * jax.random.normal(
            key,
            shape=(self.nz, self.ny, self.nx),
            dtype=self.precision.dtype_real,
        )
        return super().create_initial_state().update(q=q)

    @property
    def Ubg(self):
        return self.U.astype(self.precision.dtype_real)

    @property
    def Hi(self):
        return self.H.astype(self.precision.dtype_real)

    @property
    def rhoi(self):
        return self.rho.astype(self.precision.dtype_real)

    @property
    def Htot(self):
        """Total depth, summed over layers."""
        return self.get_grid().H

    @property
    def gpi(self):
        # reduced gravities (buoyancy jumps) between adjacent layers
        rho = self.rho.astype(jnp.float64)
        return jnp.float64(self.g) * (rho[1:] - rho[:-1]) / rho[:-1]

    @property
    def S(self):
        # stretching matrix (nz, nz), computed in float64
        f2 = jnp.float64(self.f) ** 2
        if self.nz == 2:
            f64_rd = jnp.float64(self.rd)
            f64_delta = jnp.float64(self.delta)
            F1 = f64_rd**-2 / (1.0 + f64_delta)
            F2 = f64_delta * F1
            return jnp.array([[-F1, F1], [F2, -F2]], dtype=jnp.float64)
        hi = self.Hi.astype(jnp.float64)
        gpi = self.gpi
        # S[i, i+1] = f^2 / (H_i g'_i); S[i+1, i] = f^2 / (H_{i+1} g'_i)
        a_up = f2 / hi[:-1] / gpi
        a_lo = f2 / hi[1:] / gpi
        zero = jnp.zeros((1,), dtype=jnp.float64)
        diag = -jnp.concatenate([a_up, zero]) - jnp.concatenate([zero, a_lo])
        return jnp.diag(diag) + jnp.diag(a_up, 1) + jnp.diag(a_lo, -1)

    @property
    def Qy(self):
        # meridional background PV gradient: beta - S @ Ubg
        qy = jnp.float64(self.beta) - self.S @ self.Ubg.astype(jnp.float64)
        return qy.astype(self.precision.dtype_real)

    @property
    def ikQy(self):
        return jnp.expand_dims(self.Qy, (-1, -2)) * 1j * self.k

    @property
    def ilQx(self):
        return 0.0

    def _apply_a_ph(self, state):
        f64_wv2 = self.wv2.astype(jnp.float64)
        qh = state.qh.astype(jnp.complex128)
        S = self.S
        if self.nz == 2:
            s00, s01 = S[0, 0], S[0, 1]
            s10, s11 = S[1, 0], S[1, 1]
            det = (s00 - f64_wv2) * (s11 - f64_wv2) - s01 * s10
            det1 = jnp.where(det == 0, 1.0, det)
            ph = jnp.where(
                det == 0,
                0.0,
                jnp.stack(
                    [
                        ((s11 - f64_wv2) * qh[0] - s01 * qh[1]) / det1,
                        (-s10 * qh[0] + (s00 - f64_wv2) * qh[1]) / det1,
                    ]
                ),
            )
            return ph.astype(self.precision.dtype_complex)
        # general nz: solve (S - wv2 I) ph = qh per wavenumber. S is
        # tridiagonal, so use a Thomas sweep, unrolled over the (static)
        # nz layers and vectorized over wavenumbers: O(nz) per mode
        # rather than the O(nz^3) of a batched matrix inverse, and it
        # stays differentiable in S. The matrix is strictly diagonally
        # dominant for wv2 > 0 (|S_ii| + wv2 > |S_i,i-1| + |S_i,i+1|),
        # so no pivoting is needed. The mean mode (wv2 == 0, where S
        # alone is singular) is computed with a placeholder wv2 of 1 and
        # zeroed afterward.
        nz = self.nz
        wv2_safe = jnp.where(f64_wv2 == 0, 1.0, f64_wv2)
        diag = [S[i, i] - wv2_safe for i in range(nz)]
        upper = [S[i, i + 1] for i in range(nz - 1)]
        lower = [S[i + 1, i] for i in range(nz - 1)]
        # forward elimination
        cp = [None] * (nz - 1)
        rp = [None] * nz
        cp[0] = upper[0] / diag[0]
        rp[0] = qh[0] / diag[0]
        for i in range(1, nz):
            denom = diag[i] - lower[i - 1] * cp[i - 1]
            if i < nz - 1:
                cp[i] = upper[i] / denom
            rp[i] = (qh[i] - lower[i - 1] * rp[i - 1]) / denom
        # back substitution
        sol = [None] * nz
        sol[nz - 1] = rp[nz - 1]
        for i in range(nz - 2, -1, -1):
            sol[i] = rp[i] - cp[i] * sol[i + 1]
        ph = jnp.stack(sol)
        ph = jnp.where(f64_wv2 == 0, 0.0, ph)
        return ph.astype(self.precision.dtype_complex)
