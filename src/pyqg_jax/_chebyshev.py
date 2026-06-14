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
