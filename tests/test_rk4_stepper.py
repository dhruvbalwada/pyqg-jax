# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import pytest
import numpy as np
import jax
import jax.numpy as jnp
import pyqg_jax


def _model(**kwargs):
    return pyqg_jax.callies_model.CalliesTwoEady(
        nx=32, L=5e5, dt=1.0, precision=pyqg_jax.state.Precision.DOUBLE, **kwargs
    )


def _band_limited_ic(model, amp=1e-3, kmax=4):
    key = jax.random.key(0)
    q = amp * jax.random.normal(key, (model.nz, model.ny, model.nx))
    qh = jnp.fft.rfftn(q, axes=(-2, -1))
    ky = np.abs(np.fft.fftfreq(model.nx) * model.nx)
    kh = np.hypot(ky[:, None], np.arange(model.nx // 2 + 1)[None, :])
    qh = jnp.where(jnp.asarray(kh) > kmax, 0.0, qh)
    q = jnp.fft.irfftn(qh, s=(model.ny, model.nx), axes=(-2, -1))
    q = q - q.mean(axis=(-2, -1), keepdims=True)
    return model.create_initial_state(key).update(q=q)


def _integrate(model, stepper, ic, dt, nsteps):
    sm = pyqg_jax.steppers.SteppedModel(model, stepper)
    s = sm.initialize_stepper_state(ic)

    @jax.jit
    def go(s):
        return jax.lax.scan(
            lambda c, _: (sm.step_model(c), None), s, None, length=nsteps
        )[0]

    return np.asarray(go(s).state.qh)


def test_rk4_steps_finite():
    model = _model()
    ic = model.create_initial_state(jax.random.key(0))
    out = _integrate(model, pyqg_jax.steppers.RK4Stepper(dt=600.0), ic, 600.0, 20)
    assert np.all(np.isfinite(out))


def test_rk4_more_accurate_than_ab3():
    # filterless (pure advection): RK4 should be far more accurate than AB3
    model = _model(nu=0.0, hypodiff=0.0, use_dealias_filter=False)
    ic = _band_limited_ic(model)
    T = 4000.0
    ref = _integrate(model, pyqg_jax.steppers.RK4Stepper(dt=2.5), ic, 2.5, int(T / 2.5))
    err_rk4 = np.linalg.norm(
        _integrate(model, pyqg_jax.steppers.RK4Stepper(dt=80.0), ic, 80.0, int(T / 80))
        - ref
    )
    err_ab3 = np.linalg.norm(
        _integrate(model, pyqg_jax.steppers.AB3Stepper(dt=80.0), ic, 80.0, int(T / 80))
        - ref
    )
    assert err_rk4 < err_ab3


def test_rk4_grad_finite():
    def loss(beta):
        model = pyqg_jax.layered_model.LayeredModel(
            nx=24, nz=3, f=1e-4, beta=beta, precision=pyqg_jax.state.Precision.DOUBLE
        )
        sm = pyqg_jax.steppers.SteppedModel(
            model, pyqg_jax.steppers.RK4Stepper(dt=600.0)
        )
        s = sm.create_initial_state(jax.random.key(0))
        s, _ = jax.lax.scan(lambda c, _: (sm.step_model(c), None), s, None, length=8)
        return jnp.sum(jnp.abs(sm.get_full_state(s).ph) ** 2).real

    g = jax.grad(loss)(1.5e-11)
    assert np.isfinite(float(g))


def test_rk4_vmap():
    model = _model()
    sm = pyqg_jax.steppers.SteppedModel(model, pyqg_jax.steppers.RK4Stepper(dt=600.0))
    keys = jax.random.split(jax.random.key(1), 6)

    def run(k):
        s = sm.create_initial_state(k)
        return jax.lax.scan(lambda c, _: (sm.step_model(c), None), s, None, length=8)[
            0
        ].state.qh

    out = jax.jit(jax.vmap(run))(keys)
    assert out.shape == (6, 3, model.nl, model.nk)
    assert np.all(np.isfinite(np.asarray(out)))


def test_rk4_rectangular():
    model = pyqg_jax.layered_model.LayeredModel(
        nx=16, ny=32, nz=2, f=1e-4, precision=pyqg_jax.state.Precision.DOUBLE
    )
    out = _integrate(
        model,
        pyqg_jax.steppers.RK4Stepper(dt=3600.0),
        model.create_initial_state(jax.random.key(0)),
        3600.0,
        3,
    )
    assert np.all(np.isfinite(out))


def test_rk4_tree_flatten_roundtrip():
    stepper = pyqg_jax.steppers.RK4Stepper(dt=600.0)
    leaves, treedef = jax.tree_util.tree_flatten(stepper)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.dt == stepper.dt


def test_rk4_apply_updates_raises():
    # RK4 cannot be driven with a single precomputed update
    model = _model()
    stepper = pyqg_jax.steppers.RK4Stepper(dt=600.0)
    sstate = stepper.initialize_stepper_state(
        model.create_initial_state(jax.random.key(0))
    )
    with pytest.raises(NotImplementedError, match="intermediate stages"):
        stepper.apply_updates(sstate, model.get_updates(sstate.state))
