# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


"""A next-order-in-Rossby (QG+1) balanced model.

This model extends the continuous-stratification quasigeostrophic model
to next order in the Rossby number following Dù, Smith & Bühler (2024)
(clean-room from arXiv:2408.03422). The leading-order dynamics are
ordinary continuous QG; the next-order corrections add ageostrophic
velocities, a self-consistent vertical velocity, and the
cyclone/anticyclone asymmetry that leading-order QG cannot represent.

The leading-order inversion is inherited verbatim from
:class:`~pyqg_jax.continuous_model.ContinuousQGModel`; the next-order
corrections are computed by :mod:`pyqg_jax._qgplus1`.
"""

__all__ = ["QGPlus1Model"]


import jax
import jax.numpy as jnp
from . import _qgplus1, _utils, continuous_model, state as _state


@_utils.register_pytree_class_attrs(
    children=["beta", "U", "N2"],
    static_attrs=["H"],
)
class QGPlus1Model(continuous_model.ContinuousQGModel):
    r"""Next-order-in-Rossby (QG+1) balanced model.

    Same construction and prognostic state as
    :class:`~pyqg_jax.continuous_model.ContinuousQGModel` (interior
    potential vorticity plus surface/bottom buoyancy on a Chebyshev
    vertical grid), but the diagnostics and (optionally) the dynamics
    carry the first-order Rossby corrections of Dù, Smith & Bühler
    (2024).

    The background mean state is the Eady profile ``U = Lambda * z``; the
    shear :math:`\Lambda` is read from the supplied ``U``. A uniform
    :math:`N^2` is assumed (the next-order operator
    :math:`\mathcal{L} = N^2\nabla^2 + f^2\partial_{zz}` reduces to the
    inherited inversion operator only for constant :math:`N^2`).

    .. versionadded:: 0.9.0

    See Also
    --------
    pyqg_jax.continuous_model.ContinuousQGModel
    """

    @property
    def _shear(self):
        # Eady mean shear Lambda = dU/dz (constant for the Eady profile)
        dz = self._Dz.astype(jnp.float64)
        return jnp.mean(dz @ self.U.astype(jnp.float64))

    @property
    def _n2_scalar(self):
        return self.N2.astype(jnp.float64)[0]

    def next_order(self, state):
        r"""Compute the next-order (QG+1) corrections for a state.

        Parameters
        ----------
        state : PseudoSpectralState
            The prognostic state to expand.

        Returns
        -------
        dict
            Spectral fields ``phi1h``, ``f1h``, ``g1h`` (the first-order
            potential corrections) and ``uh``, ``vh``, ``wh``, ``bh``
            (the velocities and buoyancy reconstructed to first order in
            the Rossby number), each shape ``(nz, nl, nk)``.
        """
        ph = self._apply_a_ph(state).astype(jnp.complex128)
        return _qgplus1.next_order(
            ph,
            self.k,
            self.l,
            self._Lvert64(),
            self._Dz,
            self._n2_scalar,
            jnp.float64(self.f),
            self._shear,
            (self.ny, self.nx),
        )

    def get_updates(self, state):
        """Compute the QG+1 time-stepping tendency.

        The prognostic potential vorticity is advected by the **full**
        next-order three-dimensional velocity (Eq. 29a of Dù, Smith &
        Bühler 2024), including the vertical advection :math:`w\\,q_z` on
        the interior, while the surface/bottom buoyancy is advected by
        the full horizontal velocity. The background mean-state terms
        (``Ubg`` and ``ikQy``) are inherited unchanged from the
        leading-order model.

        Parameters
        ----------
        state : PseudoSpectralState
            The state to be time stepped.

        Returns
        -------
        PseudoSpectralState
            A state object whose ``qh`` holds the tendency update.
        """
        shape = (self.ny, self.nx)
        qh = state.qh.astype(jnp.complex128)
        ph = self._apply_a_ph(state).astype(jnp.complex128)
        res = _qgplus1.next_order(
            ph,
            self.k,
            self.l,
            self._Lvert64(),
            self._Dz,
            self._n2_scalar,
            jnp.float64(self.f),
            self._shear,
            shape,
        )
        u = _state._generic_irfftn(res["uh"], shape)
        v = _state._generic_irfftn(res["vh"], shape)
        w = _state._generic_irfftn(res["wh"], shape)
        q = _state._generic_irfftn(qh, shape)

        # horizontal advection by the full next-order velocity + background U
        ubg = jnp.expand_dims(self.Ubg.astype(jnp.float64), (-1, -2))
        uqh = _state._generic_rfftn((u + ubg) * q)
        vqh = _state._generic_rfftn(v * q)
        dqhdt = jnp.negative(
            jnp.expand_dims(self._ik, (0, 1)) * uqh
            + jnp.expand_dims(self._il, (0, -1)) * vqh
            + jnp.expand_dims(self._ikQy[: self.nz], 1) * ph
        )

        # vertical advection -w q_z (interior potential vorticity only; w = 0
        # at the boundaries so the buoyancy rows are unaffected)
        qz = _state._generic_irfftn(
            jnp.tensordot(self._Dz.astype(jnp.float64), qh, axes=([1], [0])), shape
        )
        mask = jnp.expand_dims(self._interior_mask, (-1, -2))
        dqhdt = dqhdt - _state._generic_rfftn(mask * w * qz)

        # bottom drag (Beckmann friction), matching the kernel
        def with_friction(dqhdt):
            return jnp.concatenate(
                [
                    dqhdt[:-1],
                    jnp.expand_dims(dqhdt[-1] + (self.rek * self._k2l2 * ph[-1]), 0),
                ],
                axis=0,
            )

        dqhdt = jax.lax.cond(self.rek != 0, with_friction, lambda d: d, dqhdt)
        return _state.PseudoSpectralState(
            qh=dqhdt.astype(self.precision.dtype_complex),
            _q_shape=shape,
        )

    def vertical_velocity(self, state):
        r"""Next-order vertical velocity :math:`w` in physical space.

        Parameters
        ----------
        state : PseudoSpectralState
            The prognostic state.

        Returns
        -------
        jax.Array
            The vertical velocity with shape ``(nz, ny, nx)``. It
            vanishes (to roundoff) at the rigid top and bottom
            boundaries.
        """
        wh = self.next_order(state)["wh"]
        w = _state._generic_irfftn(wh, (self.ny, self.nx))
        return w.astype(self.precision.dtype_real)
