# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


"""Next-order (QG+1) balanced-model corrections.

Implements the next-order-in-Rossby balanced model of Dù, Smith &
Bühler (2024), clean-room from the paper's equations (arXiv:2408.03422).
Given the leading-order streamfunction :math:`\\Phi^0` (the ordinary
continuous-QG inversion), three further Poisson problems give the
first-order corrections :math:`\\Phi^1, F^1, G^1`, from which the
next-order velocities (including a self-consistent vertical velocity
:math:`w`) and buoyancy are reconstructed.

All operators share the same elliptic left-hand side
:math:`\\mathcal{L} = N^2\\nabla^2 + f^2\\partial_{zz}`. With uniform
:math:`N^2` this is :math:`N^2` times the operator inverted by
:class:`~pyqg_jax.continuous_model.ContinuousQGModel`, so each
correction is obtained by feeding the (scaled) right-hand side to the
shared :func:`~pyqg_jax._chebyshev.solve_vertical_bvp`. The mean-shear
(Eady) terms follow the paper's Eqs. (50)-(51); the constant pieces
(``C_q`` and ``-Lambda**2``) are dropped because they only add a
horizontally constant function to :math:`\\Phi^1` and do not affect the
reconstructed velocities.
"""

import jax.numpy as jnp
from . import _chebyshev, state as _state


def _zderiv(dz, field):
    # apply the (nz, nz) d/dz matrix along the leading (vertical) axis
    return jnp.tensordot(dz, field, axes=([1], [0]))


def next_order(ph, k, l, lvert, dz, n2, f0, lam, grid_shape):
    r"""Compute the QG+1 corrections and reconstructed fields.

    Parameters
    ----------
    ph : jax.Array
        Leading-order streamfunction :math:`\Phi^0`, spectral, shape
        ``(nz, nl, nk)``.
    k, l : jax.Array
        Horizontal wavenumbers, shape ``(nl, nk)``.
    lvert : jax.Array
        The ``(nz, nz)`` operator
        :math:`\partial_z\!\left(\tfrac{f^2}{N^2}\partial_z\right)`.
    dz : jax.Array
        The ``(nz, nz)`` first-derivative matrix.
    n2 : float
        Uniform buoyancy frequency squared :math:`N^2`.
    f0 : float
        Coriolis parameter.
    lam : float
        Eady mean shear :math:`\Lambda` (``dU/dz``); pass ``0`` for no
        background mean state.
    grid_shape : tuple
        The physical grid shape ``(ny, nx)`` for the FFTs.

    Returns
    -------
    dict
        Spectral fields ``phi1h``, ``f1h``, ``g1h`` (the corrections) and
        ``uh``, ``vh``, ``wh``, ``bh`` (the next-order reconstruction),
        each shape ``(nz, nl, nk)``.
    """
    ph = ph.astype(jnp.complex128)
    dz = dz.astype(jnp.float64)
    lvert = lvert.astype(jnp.float64)
    k = k.astype(jnp.float64)
    l = l.astype(jnp.float64)
    n2 = jnp.float64(n2)
    f0 = jnp.float64(f0)
    lam = jnp.float64(lam)
    ik = 1j * k
    il = 1j * l
    wv2 = k**2 + l**2

    phz = _zderiv(dz, ph)  # Phi0_z
    phzz = _zderiv(dz, phz)  # Phi0_zz

    def to_phys(fh):
        return _state._generic_irfftn(fh, grid_shape)

    def to_spec(fp):
        return _state._generic_rfftn(fp)

    # second-derivative fields needed for the quadratic sources (physical)
    phzx = to_phys(ik * phz)
    phzy = to_phys(il * phz)
    phxx = to_phys(-(k**2) * ph)
    phyy = to_phys(-(l**2) * ph)
    phxy = to_phys(-(k * l) * ph)
    phzz_p = to_phys(phzz)
    phyz = phzy  # Phi0_yz == Phi0_zy

    # right-hand sides L(.) = source (Eady Eqs. 50-51), dropping mean-only
    # constants C_q and -Lambda^2 (they do not affect velocities)
    src_f = 2 * f0 * (phzx * phxy - phzy * phxx + lam * phxx)
    src_g = 2 * f0 * (phzx * phyy - phzy * phxy + lam * phxy)
    src_p = -f0 * (-(f0**2 / n2) * phzz_p**2 - (phzx**2 + phzy**2) + 2 * lam * phyz)

    # L = N^2 * A (uniform N^2), so solve A x = source / N^2
    def homog_bdy(rh):
        return rh.at[0].set(0.0).at[-1].set(0.0)

    rhs_f = homog_bdy(to_spec(src_f) / n2)
    rhs_g = homog_bdy(to_spec(src_g) / n2)
    rhs_p = homog_bdy(to_spec(src_p) / n2)

    f1h = _chebyshev.solve_vertical_bvp(lvert, dz, wv2, f0, rhs_f, "dirichlet")
    g1h = _chebyshev.solve_vertical_bvp(lvert, dz, wv2, f0, rhs_g, "dirichlet")
    phi1h = _chebyshev.solve_vertical_bvp(lvert, dz, wv2, f0, rhs_p, "neumann")

    # reconstruction to O(eps) (Eq. 31)
    f1z = _zderiv(dz, f1h)
    g1z = _zderiv(dz, g1h)
    p1z = _zderiv(dz, phi1h)
    uh = -il * ph - (il * phi1h + f1z)
    vh = ik * ph + (ik * phi1h - g1z)
    wh = ik * f1h + il * g1h
    bh = f0 * phz + f0 * p1z + (n2 / f0) * (ik * g1h - il * f1h)
    return {
        "phi1h": phi1h,
        "f1h": f1h,
        "g1h": g1h,
        "uh": uh,
        "vh": vh,
        "wh": wh,
        "bh": bh,
    }
