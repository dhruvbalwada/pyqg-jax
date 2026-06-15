# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


"""Chebyshev collocation machinery for continuous vertical structure.

Provides the Chebyshev-Gauss-Lobatto grid and differentiation matrix
used by the continuous-stratification quasigeostrophic model. The grid
and matrices depend only on the (static) number of vertical points and
the domain depth, so they are constants for a given model.
"""

import numpy as np
import jax.numpy as jnp


def cheb_grid_dz(n, H):
    """Chebyshev-Gauss-Lobatto grid on ``[-H, 0]`` and the ``d/dz`` matrix.

    Parameters
    ----------
    n : int
        Number of vertical collocation points (``n >= 2``).

    H : float
        Total depth. The grid spans ``z`` in ``[-H, 0]``.

    Returns
    -------
    z : jax.Array
        The ``n`` collocation points, descending from ``0`` (surface) to
        ``-H`` (bottom).

    dz : jax.Array
        The ``(n, n)`` first-derivative matrix in ``z``.
    """
    if n < 2:
        raise ValueError("cheb_grid_dz requires n >= 2")
    N = n - 1
    j = np.arange(n)
    x = np.cos(np.pi * j / N)  # Gauss-Lobatto points on [-1, 1], descending
    c = np.ones(n)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** j)
    X = np.tile(x, (n, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n))
    D = D - np.diag(D.sum(axis=1))
    # map x in [-1, 1] -> z in [-H, 0] via z = H * (x - 1) / 2
    z = H * (x - 1.0) / 2.0
    dz = (2.0 / H) * D
    return jnp.asarray(z), jnp.asarray(dz)


def solve_vertical_bvp(lvert, dz, wv2, f0, rhs, bc):
    r"""Solve the vertical boundary-value problem at every wavenumber.

    Solves, per horizontal wavenumber :math:`\kappa` (with
    :math:`\kappa^2 =` ``wv2``),

    .. math::

       \left[\mathsf{L}_z - \kappa^2\right]\hat\psi = \hat r
       \quad(\text{interior rows}),

    where :math:`\mathsf{L}_z =` ``lvert`` is the discretized
    :math:`\partial_z\!\left(\tfrac{f_0^2}{N^2}\partial_z\right)`
    operator. The two boundary rows enforce, depending on `bc`:

    - ``"neumann"``: :math:`f_0\,\partial_z\hat\psi = \hat r` at the
      boundary (a buoyancy condition);
    - ``"dirichlet"``: :math:`\hat\psi = \hat r` at the boundary.

    The boundary entries of `rhs` (``rhs[0]`` and ``rhs[-1]``) carry the
    boundary-condition values; the interior entries carry the interior
    forcing.

    Parameters
    ----------
    lvert : jax.Array
        The ``(nz, nz)`` vertical operator, float64.
    dz : jax.Array
        The ``(nz, nz)`` first-derivative matrix, float64.
    wv2 : jax.Array
        Horizontal wavenumber squared, shape ``(nl, nk)``, float64.
    f0 : float
        Coriolis parameter (used by the Neumann boundary rows).
    rhs : jax.Array
        Right-hand side, shape ``(nz, nl, nk)``, complex128.
    bc : str
        Either ``"neumann"`` or ``"dirichlet"``.

    Returns
    -------
    jax.Array
        The solution ``psi`` with shape ``(nz, nl, nk)``, complex128.

    Note
    ----
    For ``"neumann"`` the :math:`\kappa = 0` (domain-mean) mode is
    singular (the pure-Neumann operator annihilates constants); it is
    replaced by the identity and its solution set to zero. The
    ``"dirichlet"`` problem is well posed at :math:`\kappa = 0` and is
    solved normally.
    """
    if bc not in ("neumann", "dirichlet"):
        raise ValueError(f"bc must be 'neumann' or 'dirichlet' (got {bc!r})")
    n = lvert.shape[0]
    lvert = lvert.astype(jnp.float64)
    dz = dz.astype(jnp.float64)
    wv2 = wv2.astype(jnp.float64)
    eye = jnp.eye(n, dtype=jnp.float64)
    if bc == "neumann":
        base = lvert.at[0, :].set(f0 * dz[0, :]).at[-1, :].set(f0 * dz[-1, :])
    else:
        base = lvert.at[0, :].set(eye[0, :]).at[-1, :].set(eye[-1, :])
    interior_mask = jnp.ones((n,), dtype=jnp.float64).at[0].set(0.0).at[-1].set(0.0)
    eye_int = jnp.diag(interior_mask)
    # L[l, k] = base - kappa^2 * (identity on interior rows)
    op = base - jnp.expand_dims(wv2, (-1, -2)) * eye_int
    singular = (wv2 == 0) if bc == "neumann" else jnp.zeros_like(wv2, dtype=bool)
    op = jnp.where(jnp.expand_dims(singular, (-1, -2)), eye, op).astype(jnp.complex128)
    rhs_c = jnp.expand_dims(jnp.moveaxis(rhs.astype(jnp.complex128), 0, -1), -1)
    psi = jnp.linalg.solve(op, rhs_c)[..., 0]  # (nl, nk, nz)
    psi = jnp.where(jnp.expand_dims(singular, -1), 0.0, psi)
    return jnp.moveaxis(psi, -1, 0)  # (nz, nl, nk)
