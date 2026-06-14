# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import math
import functools
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import pyqg_jax
from pyqg_jax.continuous_model import ContinuousQGModel

# uniform-stratification Eady configuration with an exact dispersion relation
H = 1000.0
F = 1e-4
NBV = 1e-3  # buoyancy frequency
LAM = 1e-3  # vertical shear, U = LAM * z


def _eady_model(nx=64, nz=32, L=1e6):
    base = ContinuousQGModel(
        nx=nx,
        nz=nz,
        L=L,
        H=H,
        f=F,
        beta=0.0,
        N2=NBV**2,
        precision=pyqg_jax.state.Precision.DOUBLE,
    )
    return ContinuousQGModel(
        nx=nx,
        nz=nz,
        L=L,
        H=H,
        f=F,
        beta=0.0,
        N2=NBV**2,
        U=LAM * np.asarray(base.z),
        precision=pyqg_jax.state.Precision.DOUBLE,
    )


def test_requires_min_levels():
    with pytest.raises(ValueError, match="nz >= 3"):
        ContinuousQGModel(nz=2, f=1e-4)


def test_grid_and_state():
    model = _eady_model(nz=16)
    z = np.asarray(model.z)
    assert math.isclose(z[0], 0.0, abs_tol=1e-9)
    assert math.isclose(z[-1], -H)
    # Hi (vertical weights) sums to the depth
    assert math.isclose(float(np.asarray(model.Hi).sum()), H, rel_tol=1e-10)
    state = model.create_initial_state(jax.random.key(0))
    assert state.q.shape == (model.nz, model.ny, model.nx)
    assert np.abs(np.asarray(state.q).mean(axis=(-2, -1))).max() < 1e-18


def test_inversion_matches_analytic_edge():
    # interior PV = 0, surface buoyancy = 1, bottom buoyancy = 0: the
    # inversion must match the analytic finite-depth edge solution
    model = _eady_model(nz=40)
    z = np.asarray(model.z)
    nl, nk = model.nl, model.nk
    kk = np.asarray(model.k)
    # pick a non-mean wavenumber column
    il, ik = 0, 5
    kappa = kk[il, ik]
    qh = np.zeros((model.nz, nl, nk), dtype=np.complex128)
    qh[0, il, ik] = 1.0  # surface buoyancy
    ph = np.asarray(
        model._apply_a_ph(
            pyqg_jax.state.PseudoSpectralState(
                qh=jnp.asarray(qh), _q_shape=(model.ny, model.nx)
            )
        )
    )[:, il, ik]
    m = NBV * kappa / F
    B = 1.0 / (F * m * (np.exp(2 * m * H) - 1))
    A = B * np.exp(2 * m * H)
    psi_a = A * np.exp(m * z) + B * np.exp(-m * z)
    assert np.allclose(ph.real, psi_a, rtol=1e-8)
    assert np.abs(ph.imag).max() < 1e-12 * np.abs(ph.real).max()


def test_stability_reproduces_eady():
    model = _eady_model(nx=64, nz=48, L=1e6)
    omega, _ = model.stability_analysis()
    sigma = np.asarray(omega).imag
    ll = np.asarray(model.l)
    kk = np.asarray(model.k)
    mask = ll == 0
    sig0 = sigma[mask]
    k0 = kk[mask]
    peak = np.argmax(sig0)
    eady = 0.31 * F * LAM / NBV
    mu = NBV * H * k0[peak] / F
    assert math.isclose(sig0[peak], eady, rel_tol=0.1)
    assert 1.2 < mu < 2.1  # Eady peak near mu ~ 1.6


def test_charney_surface_trapped_no_cutoff():
    # with beta (interior PV gradient) the surface buoyancy gradient gives
    # the Charney instability: a surface-trapped mode and no short-wave cutoff
    f = 1e-4
    nbv = 1e-3
    beta = 1.5e-11
    depth = 4000.0
    shear = -1.2e-6  # sign selecting surface trapping
    grid = ContinuousQGModel(
        nx=64,
        nz=24,
        L=3e5,
        H=depth,
        f=f,
        beta=beta,
        N2=nbv**2,
        precision=pyqg_jax.state.Precision.DOUBLE,
    )
    model = ContinuousQGModel(
        nx=64,
        nz=24,
        L=3e5,
        H=depth,
        f=f,
        beta=beta,
        N2=nbv**2,
        U=shear * grid.z,
        precision=pyqg_jax.state.Precision.DOUBLE,
    )
    omega, evec = model.stability_analysis()
    sigma = np.asarray(omega).imag
    ll = np.asarray(model.l)
    kk = np.asarray(model.k)
    mask = ll == 0
    sig0 = sigma[mask]
    k0 = kk[mask]
    peak = np.argmax(sig0)
    assert sig0[peak] > 0  # unstable

    # most-unstable mode is surface-trapped (decays away from z = 0)
    z = np.asarray(model.z)
    ev = np.abs(np.asarray(evec)[:, 0, np.argmax(sig0)])
    ev = ev / ev.max()
    surf = ev[np.argmin(np.abs(z))]
    mid = ev[np.argmin(np.abs(z + depth / 2))]
    assert surf > 5 * mid

    # no short-wave cutoff: short waves remain unstable (Eady would not)
    wl_km = 2 * np.pi / np.where(k0 > 0, k0, np.inf) / 1e3
    short = (wl_km < 25) & (k0 > 0)
    assert np.any(sig0[short] > 1e-10)


def _integrate(model, dt, nsteps, amp=1e-9, seed=0):
    sm = pyqg_jax.steppers.SteppedModel(model, pyqg_jax.steppers.AB3Stepper(dt=dt))
    key = jax.random.key(seed)
    q = amp * jax.random.normal(key, (model.nz, model.ny, model.nx))
    q = q - q.mean(axis=(-2, -1), keepdims=True)
    state = sm.initialize_stepper_state(model.create_initial_state(key).update(q=q))

    @functools.partial(jax.jit, static_argnames=["n"])
    def run(s, n):
        return jax.lax.scan(lambda c, _: (sm.step_model(c), None), s, None, length=n)[0]

    final = run(state, nsteps)

    def ke(s):
        return float(jnp.sum(jnp.abs(sm.get_full_state(s).ph) ** 2).real)

    return ke(state), ke(final), np.asarray(final.state.q)


def test_eady_grows_and_stays_finite():
    model = _eady_model(nx=48, nz=20, L=4e5)
    ke0, ke1, q = _integrate(model, dt=300.0, nsteps=int(2 * 86400 / 300))
    assert np.all(np.isfinite(q))
    assert ke1 > 10 * ke0  # baroclinic instability amplifies energy


def test_no_shear_does_not_grow():
    # with no background shear there is no instability: energy must not grow
    model = ContinuousQGModel(
        nx=48,
        nz=20,
        L=4e5,
        H=H,
        f=F,
        beta=0.0,
        N2=NBV**2,
        precision=pyqg_jax.state.Precision.DOUBLE,
    )
    ke0, ke1, q = _integrate(model, dt=300.0, nsteps=int(2 * 86400 / 300))
    assert np.all(np.isfinite(q))
    assert ke1 < 2.0 * ke0


def test_grad_and_vmap():
    def loss(lam):
        base = ContinuousQGModel(
            nx=24,
            nz=12,
            L=4e5,
            H=H,
            f=F,
            N2=NBV**2,
            precision=pyqg_jax.state.Precision.DOUBLE,
        )
        model = ContinuousQGModel(
            nx=24,
            nz=12,
            L=4e5,
            H=H,
            f=F,
            N2=NBV**2,
            U=lam * base.z,
            precision=pyqg_jax.state.Precision.DOUBLE,
        )
        sm = pyqg_jax.steppers.SteppedModel(
            model, pyqg_jax.steppers.AB3Stepper(dt=600.0)
        )
        s = sm.create_initial_state(jax.random.key(0))
        s = jax.lax.scan(lambda c, _: (sm.step_model(c), None), s, None, length=6)[0]
        return jnp.sum(jnp.abs(sm.get_full_state(s).ph) ** 2).real

    g = jax.grad(loss)(1e-3)
    assert np.isfinite(float(g))

    model = ContinuousQGModel(
        nx=24,
        nz=12,
        L=4e5,
        H=H,
        f=F,
        N2=NBV**2,
        precision=pyqg_jax.state.Precision.DOUBLE,
    )
    sm = pyqg_jax.steppers.SteppedModel(model, pyqg_jax.steppers.AB3Stepper(dt=600.0))

    def run(key):
        s = sm.create_initial_state(key)
        return jax.lax.scan(lambda c, _: (sm.step_model(c), None), s, None, length=4)[
            0
        ].state.qh

    out = jax.jit(jax.vmap(run))(jax.random.split(jax.random.key(0), 4))
    assert out.shape == (4, model.nz, model.nl, model.nk)
    assert np.all(np.isfinite(np.asarray(out)))
