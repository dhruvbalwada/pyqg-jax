# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import abc
import numpy as np
import jax
import jax.numpy as jnp
from . import _kernel, _utils, state


def _grid_xy(nx, ny, L, W, dtype_real):
    x, y = jnp.meshgrid(
        (jnp.arange(0.5, nx, 1.0, dtype=dtype_real) / nx) * L,
        (jnp.arange(0.5, ny, 1.0, dtype=dtype_real) / ny) * W,
    )
    return x, y


def _grid_kl(kk, ll):
    k, l = jnp.meshgrid(kk, ll)
    return k, l


@_utils.register_pytree_class_attrs(
    children=["L", "W", "filterfac", "g", "f"],
    static_attrs=[],
)
class Model(_kernel.PseudoSpectralKernel):
    def __init__(
        self,
        *,
        # grid size parameters
        nz=1,
        ny=None,
        nx=64,
        L=1e6,
        W=None,
        # friction parameters
        rek=5.787e-7,
        filterfac=23.6,
        # constants
        f=None,
        g=9.81,
        precision=state.Precision.SINGLE,
    ):
        super().__init__(
            nz=nz,
            ny=ny if ny is not None else nx,
            nx=nx,
            rek=rek,
            precision=precision,
        )
        self.L = L
        self.W = W if W is not None else L
        self.filterfac = filterfac
        self.g = g
        self.f = f

    def get_full_state(
        self, state: state.PseudoSpectralState
    ) -> state.FullPseudoSpectralState:
        """Expand a partial state into a full state with all computed values.

        Parameters
        ----------
        state : PseudoSpectralState
            The partial state to be expanded.

        Returns
        -------
        FullPseudoSpectralState
            New state object with all computed fields derived from `state`.
        """
        full_state = super().get_full_state(state)
        full_state = self._do_external_forcing(full_state)
        return full_state

    def get_grid(self) -> state.Grid:
        """Retrieve information on the model grid.

        .. versionadded:: 0.8.0

        Returns
        -------
        Grid
            A grid instance with attributes giving information on the
            spatial and spectral model grids.
        """
        return state.Grid(
            nz=self.nz,
            ny=self.ny,
            nx=self.nx,
            L=self.L,
            W=self.W,
            Hi=self.Hi,
        )

    def _do_external_forcing(
        self, state: state.FullPseudoSpectralState
    ) -> state.FullPseudoSpectralState:
        return state

    def stability_analysis(self):
        r"""Linear stability analysis of the background state.

        Solves, at every wavenumber, the eigenvalue problem for normal
        modes :math:`\hat\theta \propto e^{-\mathrm{i}\omega t}` of the
        linearized dynamics,

        .. math::

           k\,(\mathsf{U} + \mathsf{Q}_y\,\mathsf{a})\,\tilde\theta
           = \omega\,\tilde\theta,

        where :math:`\mathsf{U} = \mathrm{diag}(\mathtt{Ubg})`,
        :math:`\mathsf{Q}_y = \mathrm{diag}(\mathtt{Qy})`, and
        :math:`\mathsf{a}` is the model's inversion matrix
        (:math:`\hat\psi = \mathsf{a}\,\hat q`). The growth rate at each
        wavenumber is :math:`\mathrm{Im}(\omega)`.

        .. versionadded:: 0.9.0

        Returns
        -------
        omega : jax.Array
            The eigenvalue with the largest growth rate (imaginary part)
            at each wavenumber, shape :pycode:`(nl, nk)`. The growth rate
            is :pycode:`omega.imag` and the phase speed is
            :pycode:`omega.real / k`.

        evec : jax.Array
            The corresponding eigenvector (the vertical structure of the
            most unstable mode) at each wavenumber, shape
            :pycode:`(nz, nl, nk)`.

        Note
        ----
        The eigenvalue problem is solved on the host with NumPy, so this
        is a host-side diagnostic (not differentiable or jit-compatible).
        It works regardless of the active JAX platform.
        """
        nz, nl, nk = self.nz, self.nl, self.nk
        real_shape = self.get_grid().real_state_shape[-2:]
        # inversion matrix a[i, j] (per wavenumber) from unit-basis columns
        cols = []
        for j in range(nz):
            unit = (
                jnp.zeros((nz, nl, nk), dtype=self.precision.dtype_complex)
                .at[j]
                .set(1.0)
            )
            col = self._apply_a_ph(
                state.PseudoSpectralState(qh=unit, _q_shape=real_shape)
            )
            cols.append(col)
        a = jnp.stack(cols, axis=1).real  # (nz, nz, nl, nk): a[i, j, l, k]
        a = jnp.moveaxis(a, (0, 1), (-2, -1))  # (nl, nk, nz, nz)
        ubg = self.Ubg.astype(a.dtype)
        qy = self.Qy.astype(a.dtype)
        eye = jnp.eye(nz, dtype=a.dtype)
        # M = k * (diag(Ubg) + diag(Qy) @ a)
        inner = jnp.expand_dims(ubg, -1) * eye + jnp.expand_dims(qy, -1) * a
        kk = jnp.expand_dims(self.k, (-1, -2))  # (nl, nk, 1, 1)
        # eigensolve on the host (jnp.linalg.eig is CPU-only); this is a
        # host-side diagnostic, not part of the differentiable model.
        mat = np.asarray(kk * inner)  # (nl, nk, nz, nz)
        evals, evecs = np.linalg.eig(mat)  # (nl, nk, nz), (nl, nk, nz, nz)
        imax = evals.imag.argmax(axis=-1)  # (nl, nk)
        ii, jj = np.indices(imax.shape)
        omega = evals[ii, jj, imax]  # (nl, nk)
        evec = np.moveaxis(evecs[ii, jj, :, imax], -1, 0)  # (nz, nl, nk)
        return jnp.asarray(omega), jnp.asarray(evec)

    @property
    def f2(self):
        if self.f is not None:
            return self.f**2
        else:
            return None

    @property
    def dk(self):
        return self.get_grid().dk

    @property
    def dl(self):
        return self.get_grid().dl

    @property
    def dx(self):
        return self.get_grid().dx

    @property
    def dy(self):
        return self.get_grid().dy

    @property
    def M(self):
        return self.nx * self.ny

    @property
    @abc.abstractmethod
    def Hi(self) -> jax.Array:
        pass

    @property
    def x(self):
        return _grid_xy(
            nx=self.nx,
            ny=self.ny,
            L=self.L,
            W=self.W,
            dtype_real=self.precision.dtype_real,
        )[0]

    @property
    def y(self):
        return _grid_xy(
            nx=self.nx,
            ny=self.ny,
            L=self.L,
            W=self.W,
            dtype_real=self.precision.dtype_real,
        )[1]

    @property
    def ll(self):
        return jnp.fft.fftfreq(
            self.ny,
            d=(self.W / (2 * jnp.pi * self.ny)),
            dtype=self.precision.dtype_real,
        )

    @property
    def kk(self):
        return jnp.fft.rfftfreq(
            self.nx,
            d=(self.L / (2 * jnp.pi * self.nx)),
            dtype=self.precision.dtype_real,
        )

    @property
    def k(self):
        return _grid_kl(kk=self.kk, ll=self.ll)[0]

    @property
    def l(self):
        return _grid_kl(kk=self.kk, ll=self.ll)[1]

    @property
    def ik(self):
        return 1j * self.k

    @property
    def il(self):
        return 1j * self.l

    @property
    def wv2(self):
        return self.wv**2

    @property
    def wv(self):
        return self.get_grid().get_kappa(self.precision)

    @property
    def wv2i(self):
        return jnp.where((self.wv2 != 0), jnp.power(self.wv2, -1), self.wv2)

    @property
    def filtr(self):
        cphi = 0.65 * jnp.pi
        wvx = jnp.sqrt((self.k * self.dx) ** 2 + (self.l * self.dy) ** 2)
        filtr = jnp.exp(-self.filterfac * (wvx - cphi) ** 4)
        return jnp.where(wvx <= cphi, 1, filtr)
