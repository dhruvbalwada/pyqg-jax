# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import functools
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import pyqg_jax
from pyqg_jax import _chebyshev
from pyqg_jax.qgplus1_model import QGPlus1Model

requires_x64 = pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="need float64 enabled"
)

DOUBLE = pyqg_jax.state.Precision.DOUBLE


# ---------------------------------------------------------------------------
# Q1A: vertical BVP inversion kit (Neumann + Dirichlet), manufactured solution
# ---------------------------------------------------------------------------

# uniform stratification so the operator has a known analytic form
H = 1000.0
F0 = 1e-4
N2 = 1e-6  # N = 1e-3
F2N2 = F0**2 / N2


def _operator(n):
    z, dz = _chebyshev.cheb_grid_dz(n, H)
    z = np.asarray(z)
    dz64 = np.asarray(dz)
    lvert = dz64 @ (F2N2 * dz64)  # d_z((f^2/N^2) d_z), uniform N^2
    return z, jnp.asarray(dz), jnp.asarray(lvert)


def _manufactured(z, wv2):
    # psi(z) = cos(pi z / H): zero normal derivative at both boundaries
    # (so the Neumann BC values are homogeneous, matching the Phi^1 case)
    m = np.pi / H
    psi = np.cos(m * z)  # (nz,)
    psi_z = -m * np.sin(m * z)
    psi_zz = -(m**2) * np.cos(m * z)
    # interior forcing (Lvert - kappa^2) psi, per column
    interior = (F2N2 * psi_zz)[:, None] - wv2[None, :] * psi[:, None]  # (nz, ncol)
    return psi, psi_z, interior


@requires_x64
def test_bvp_neumann_recovers_manufactured():
    n = 24
    z, dz, lvert = _operator(n)
    # kappa^2 values (exclude 0: the Neumann mean mode is singular by design)
    wv2 = np.array([1e-8, 1e-6, 1e-4])
    psi, psi_z, interior = _manufactured(z, wv2)
    rhs = np.array(interior, dtype=np.complex128)  # (nz, ncol)
    rhs[0, :] = F0 * psi_z[0]  # Neumann BC value at surface (= 0 here)
    rhs[-1, :] = F0 * psi_z[-1]  # Neumann BC value at bottom (= 0 here)
    rhs = jnp.asarray(rhs)[:, None, :]  # (nz, nl=1, nk=ncol)
    out = np.asarray(
        _chebyshev.solve_vertical_bvp(
            lvert, dz, jnp.asarray(wv2)[None, :], F0, rhs, "neumann"
        )
    )[:, 0, :]
    for j in range(wv2.size):
        assert np.allclose(out[:, j].real, psi, atol=1e-8, rtol=1e-6)
        assert np.abs(out[:, j].imag).max() < 1e-12


@requires_x64
def test_bvp_dirichlet_recovers_manufactured():
    n = 24
    z, dz, lvert = _operator(n)
    wv2 = np.array([0.0, 1e-6, 1e-4])  # Dirichlet is well posed at kappa=0
    psi, _psi_z, interior = _manufactured(z, wv2)
    rhs = np.array(interior, dtype=np.complex128)
    rhs[0, :] = psi[0]  # Dirichlet BC value at surface
    rhs[-1, :] = psi[-1]  # Dirichlet BC value at bottom
    rhs = jnp.asarray(rhs)[:, None, :]
    out = np.asarray(
        _chebyshev.solve_vertical_bvp(
            lvert, dz, jnp.asarray(wv2)[None, :], F0, rhs, "dirichlet"
        )
    )[:, 0, :]
    for j in range(wv2.size):
        assert np.allclose(out[:, j].real, psi, atol=1e-8, rtol=1e-6)


@requires_x64
def test_bvp_neumann_mean_mode_is_zero():
    n = 16
    _z, dz, lvert = _operator(n)
    wv2 = jnp.array([[0.0]])
    rhs = jnp.ones((n, 1, 1), dtype=jnp.complex128)
    out = _chebyshev.solve_vertical_bvp(lvert, dz, wv2, F0, rhs, "neumann")
    assert np.allclose(np.asarray(out), 0.0)


@requires_x64
def test_bvp_dirichlet_homogeneous_zero_source_is_zero():
    # F1/G1 case: homogeneous Dirichlet with zero interior forcing -> zero
    n = 16
    _z, dz, lvert = _operator(n)
    wv2 = jnp.array([[1e-6, 1e-4]])
    rhs = jnp.zeros((n, 1, 2), dtype=jnp.complex128)
    out = _chebyshev.solve_vertical_bvp(lvert, dz, wv2, F0, rhs, "dirichlet")
    assert np.allclose(np.asarray(out), 0.0)


@requires_x64
def test_bvp_rejects_bad_bc():
    n = 8
    _z, dz, lvert = _operator(n)
    with pytest.raises(ValueError, match=r"neumann.*dirichlet"):
        _chebyshev.solve_vertical_bvp(
            lvert,
            dz,
            jnp.array([[1e-6]]),
            F0,
            jnp.zeros((n, 1, 1), jnp.complex128),
            "x",
        )


# ---------------------------------------------------------------------------
# Q1B: next-order corrections + reconstruction
# ---------------------------------------------------------------------------


def _qgp1_model(nx=48, nz=20, L=4e5, depth=1000.0, shear=0.0):
    grid = QGPlus1Model(
        nx=nx, nz=nz, L=L, H=depth, f=F0, beta=0.0, N2=N2, precision=DOUBLE
    )
    return QGPlus1Model(
        nx=nx,
        nz=nz,
        L=L,
        H=depth,
        f=F0,
        beta=0.0,
        N2=N2,
        U=shear * np.asarray(grid.z),
        precision=DOUBLE,
    )


def _random_state(model, amp, seed=0):
    key = jax.random.key(seed)
    q = amp * jax.random.normal(key, (model.nz, model.ny, model.nx))
    q = q - q.mean(axis=(-2, -1), keepdims=True)
    return model.create_initial_state(key).update(q=q)


@requires_x64
def test_corrections_are_finite_and_nonzero():
    model = _qgp1_model(shear=8e-4)
    state = _random_state(model, amp=1e-3)
    res = model.next_order(state)
    for name, field in res.items():
        arr = np.asarray(field)
        assert np.all(np.isfinite(arr)), name
    # the corrections must actually do something
    assert np.abs(np.asarray(res["f1h"])).max() > 0
    assert np.abs(np.asarray(res["wh"])).max() > 0


@requires_x64
def test_vertical_velocity_vanishes_at_boundaries():
    # F1 = G1 = 0 (homogeneous Dirichlet) -> w = F1_x + G1_y = 0 at z = 0, -H
    model = _qgp1_model(shear=8e-4)
    state = _random_state(model, amp=1e-3)
    w = np.asarray(model.vertical_velocity(state))
    scale = np.abs(w).max()
    assert np.abs(w[0]).max() < 1e-10 * scale
    assert np.abs(w[-1]).max() < 1e-10 * scale


@requires_x64
def test_eps_to_zero_reduces_to_qg():
    # corrections are quadratic in the field amplitude; the leading-order
    # velocity is linear -> the relative size of the correction scales like
    # the amplitude (the Rossby number). Halving by 10x must shrink it ~10x.
    model = _qgp1_model(shear=0.0)  # no mean state: pure quadratic corrections

    def rel_correction(amp):
        state = model.next_order(_random_state(model, amp=amp))
        ph = model._apply_a_ph(_random_state(model, amp=amp))
        lead = np.asarray(-1j * model.l * ph)  # u^0 = -Phi0_y
        corr = np.asarray(state["uh"]) - lead
        return np.linalg.norm(corr) / np.linalg.norm(lead)

    r_big = rel_correction(1e-2)
    r_small = rel_correction(1e-3)
    ratio = r_big / r_small
    assert 8.0 < ratio < 12.0  # linear in amplitude (Rossby number)


@requires_x64
def test_symmetry_breaking_quadratic_invariance():
    # Eq. 44: with no mean state, q -> -q gives Phi0 -> -Phi0 but the
    # corrections Phi1, F1, G1 are invariant (their sources are quadratic).
    model = _qgp1_model(shear=0.0)
    state = _random_state(model, amp=1e-3)
    neg = state.update(q=-state.q)
    a = model.next_order(state)
    b = model.next_order(neg)
    for name in ("phi1h", "f1h", "g1h"):
        fa = np.asarray(a[name])
        fb = np.asarray(b[name])
        assert np.allclose(fa, fb, atol=1e-14 * (np.abs(fa).max() + 1e-30))


@requires_x64
def test_mean_state_breaks_the_symmetry():
    # with the Eady mean shear the linear Lambda terms break the q -> -q
    # invariance -- this is the cyclone/anticyclone asymmetry
    model = _qgp1_model(shear=8e-4)
    state = _random_state(model, amp=1e-3)
    neg = state.update(q=-state.q)
    a = np.asarray(model.next_order(state)["f1h"])
    b = np.asarray(model.next_order(neg)["f1h"])
    assert not np.allclose(a, b, atol=1e-10 * np.abs(a).max())


@requires_x64
def test_w_matches_independent_omega_equation():
    # the QG+1 vertical velocity must equal the standard QG omega-equation
    # w, N^2 grad^2 w + f^2 w_zz = 2 div Q, with the Q-vector built from the
    # geostrophic velocities and buoyancy by an INDEPENDENT formula
    # Q1 = -(u_gx b_x + v_gx b_y), Q2 = -(u_gy b_x + v_gy b_y).
    from pyqg_jax import _chebyshev, state as _state

    model = QGPlus1Model(nx=96, nz=24, L=4e5, H=1000.0, f=F0, N2=N2, precision=DOUBLE)
    ny, nx, nz = model.ny, model.nx, model.nz
    z = np.asarray(model.z)
    X, Y = np.meshgrid(np.arange(nx) / nx * 4e5, np.arange(ny) / ny * 4e5)
    q = np.zeros((nz, ny, nx))
    for kx, ky, p in [(1, 1, 0.0), (2, 1, 0.7), (1, 2, 1.3)]:
        horiz = 1e-4 * np.cos(2 * np.pi * (kx * X / 4e5 + ky * Y / 4e5) + p)
        q += horiz[None] * np.cos(np.pi * z / 1000.0)[:, None, None]
    q -= q.mean(axis=(-2, -1), keepdims=True)
    state = model.create_initial_state(jax.random.key(0)).update(q=jnp.asarray(q))

    w_model = np.asarray(model.vertical_velocity(state))

    ph = model._apply_a_ph(state).astype(np.complex128)
    k = jnp.asarray(model.k)
    l = jnp.asarray(model.l)
    ik, il = 1j * k, 1j * l
    dz = model._Dz.astype(jnp.float64)

    def to_p(fh):
        return _state._generic_irfftn(fh, (ny, nx))

    def to_s(fp):
        return _state._generic_rfftn(fp)

    phz = jnp.tensordot(dz, ph, axes=([1], [0]))
    ug, vg, b = to_p(-il * ph), to_p(ik * ph), to_p(F0 * phz)
    ugx, ugy = to_p(ik * to_s(ug)), to_p(il * to_s(ug))
    vgx, vgy = to_p(ik * to_s(vg)), to_p(il * to_s(vg))
    bx, by = to_p(ik * to_s(b)), to_p(il * to_s(b))
    q1 = -(ugx * bx + vgx * by)
    q2 = -(ugy * bx + vgy * by)
    rhs = (2 * (ik * to_s(q1) + il * to_s(q2)) / N2).at[0].set(0.0).at[-1].set(0.0)
    wh = _chebyshev.solve_vertical_bvp(
        model._Lvert64(), dz, model.wv2, F0, rhs, "dirichlet"
    )
    w_omega = np.asarray(to_p(wh))
    assert np.linalg.norm(w_model - w_omega) < 1e-10 * np.linalg.norm(w_omega)


@requires_x64
def test_next_order_grad_jit_vmap():
    model = _qgp1_model(nx=24, nz=12, shear=5e-4)

    def loss(amp):
        state = _random_state(model, amp=amp)
        return jnp.sum(jnp.abs(model.next_order(state)["wh"]) ** 2).real

    g = jax.grad(loss)(1e-3)
    assert np.isfinite(float(g))

    @jax.jit
    def w_of(state):
        return model.vertical_velocity(state)

    states = jax.vmap(lambda s: _random_state(model, amp=1e-3, seed=s))(jnp.arange(3))
    out = jax.vmap(w_of)(states)
    assert out.shape == (3, model.nz, model.ny, model.nx)
    assert np.all(np.isfinite(np.asarray(out)))


# ---------------------------------------------------------------------------
# Q1C: dynamical tendency (full corrected advection + vertical advection)
# ---------------------------------------------------------------------------


def _rollout(model, amp, nsteps, dt, seed=0):
    sm = pyqg_jax.steppers.SteppedModel(model, pyqg_jax.steppers.AB3Stepper(dt=dt))
    key = jax.random.key(seed)
    q = amp * jax.random.normal(key, (model.nz, model.ny, model.nx))
    q = q - q.mean(axis=(-2, -1), keepdims=True)
    st = sm.initialize_stepper_state(model.create_initial_state(key).update(q=q))

    @functools.partial(jax.jit, static_argnames=["n"])
    def go(s, n):
        return jax.lax.scan(lambda c, _: (sm.step_model(c), None), s, None, length=n)[0]

    return go(st, nsteps)


@requires_x64
def test_dynamics_reduce_to_continuous_qg():
    # as the amplitude -> 0 the QG+1 trajectory must approach the leading-order
    # continuous-QG trajectory, with the difference linear in the amplitude
    from pyqg_jax.continuous_model import ContinuousQGModel

    shear = 8e-4

    def both(amp):
        kw = {
            "nx": 32,
            "nz": 16,
            "L": 4e5,
            "H": 1000.0,
            "f": F0,
            "N2": N2,
            "precision": DOUBLE,
        }
        gz = np.asarray(ContinuousQGModel(**kw).z)
        mp1 = QGPlus1Model(**kw, U=shear * gz)
        mqg = ContinuousQGModel(**kw, U=shear * gz)
        qp = np.asarray(_rollout(mp1, amp, 200, 200.0).state.q)
        qq = np.asarray(_rollout(mqg, amp, 200, 200.0).state.q)
        return np.linalg.norm(qp - qq) / np.linalg.norm(qq)

    r_big = both(1e-5)
    r_small = both(1e-6)
    assert np.isfinite(r_big)
    assert 7.0 < r_big / r_small < 13.0  # linear in amplitude


@requires_x64
def test_develops_cyclonic_skewness():
    # the dynamical headline: under the Eady mean state QG+1 develops a
    # long cyclonic (positive) vorticity tail that leading-order QG does not
    from pyqg_jax.continuous_model import ContinuousQGModel

    shear = 8e-4
    kw = {
        "nx": 32,
        "nz": 16,
        "L": 4e5,
        "H": 1000.0,
        "f": F0,
        "N2": N2,
        "precision": DOUBLE,
    }
    gz = np.asarray(ContinuousQGModel(**kw).z)
    mp1 = QGPlus1Model(**kw, U=shear * gz)
    mqg = ContinuousQGModel(**kw, U=shear * gz)
    nsteps = int(4.0 * 86400 / 200)
    sp1 = _rollout(mp1, 1e-6, nsteps, 200.0, seed=3).state
    sqg = _rollout(mqg, 1e-6, nsteps, 200.0, seed=3).state

    def skew(model, state, *, next_order=False):
        if next_order:
            res = model.next_order(state)
            zh = 1j * model.k * res["vh"][0] - 1j * model.l * res["uh"][0]
        else:
            ph = model._apply_a_ph(state)
            zh = -model.wv2 * ph[0]
        z = np.asarray(jnp.fft.irfftn(zh, s=(model.ny, model.nx)))
        return ((z - z.mean()) ** 3).mean() / z.std() ** 3

    skew_p1 = skew(mp1, sp1, next_order=True)
    skew_qg = skew(mqg, sqg, next_order=False)
    assert np.all(np.isfinite(np.asarray(sp1.q)))
    assert skew_p1 > 0.5  # strong cyclonic asymmetry
    assert skew_p1 > skew_qg + 0.2  # markedly more than leading-order QG


@requires_x64
def test_reaches_bounded_equilibrium():
    # the Eady-forced run saturates into a statistical equilibrium (the
    # spectral filter dissipates at small scales) rather than blowing up
    model = _qgp1_model(nx=24, nz=12, shear=8e-4)
    state = _rollout(model, 1e-6, int(8.0 * 86400 / 150), 150.0, seed=1).state
    q = np.asarray(state.q)
    assert np.all(np.isfinite(q))
    ph = model._apply_a_ph(state)
    zeta = np.asarray(jnp.fft.irfftn(-model.wv2 * ph[0], s=(model.ny, model.nx))) / F0
    ro = zeta.std()
    assert 0.05 < ro < 3.0  # developed turbulence, but bounded (no runaway)


@requires_x64
def test_dynamics_grad_jit_vmap():
    model = _qgp1_model(nx=24, nz=12, shear=5e-4)
    dt = 200.0

    def loss(amp):
        st = _rollout(model, amp, 5, dt)
        return jnp.sum(jnp.abs(st.state.qh) ** 2).real

    g = jax.grad(loss)(1e-4)
    assert np.isfinite(float(g))

    sm = pyqg_jax.steppers.SteppedModel(model, pyqg_jax.steppers.AB3Stepper(dt=dt))

    def from_key(key):
        q = 1e-4 * jax.random.normal(key, (model.nz, model.ny, model.nx))
        q = q - q.mean(axis=(-2, -1), keepdims=True)
        st = sm.initialize_stepper_state(model.create_initial_state(key).update(q=q))
        return jax.lax.scan(lambda c, _: (sm.step_model(c), None), st, None, length=5)[
            0
        ]

    out = jax.jit(jax.vmap(from_key))(jax.random.split(jax.random.key(0), 3))
    assert np.all(np.isfinite(np.asarray(out.state.qh)))
