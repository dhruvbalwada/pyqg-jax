# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import warnings
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import pyqg_jax

# nz=3 layered configuration shared between comparison tests
LAYERED3 = {
    "nx": 48,
    "nz": 3,
    "L": 1e6,
    "f": 1e-4,
    "beta": 1.5e-11,
    "rek": 5.787e-7,
    "U": [0.06, 0.03, 0.0],
    "H": [300.0, 700.0, 1000.0],
    "rho": [1025.0, 1025.3, 1025.9],
}
LAYERED2 = {
    "nx": 48,
    "nz": 2,
    "L": 1e6,
    "f": 1e-4,
    "beta": 1.5e-11,
    "rek": 5.787e-7,
    "rd": 15000.0,
    "delta": 0.25,
    "U": [0.05, 0.0],
    "H": [500.0, 1500.0],
    "rho": [1025.0, 1025.6],
}
DT = 3600.0


def _make_pyqg(pyqg, cfg):
    kwargs = dict(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        return pyqg.LayeredModel(dt=DT, log_level=0, **kwargs)


def _make_jax(cfg):
    kwargs = dict(cfg)
    for k in ("U", "H", "rho"):
        if k in kwargs:
            kwargs[k] = jnp.asarray(kwargs[k])
    return pyqg_jax.layered_model.LayeredModel(
        precision=pyqg_jax.state.Precision.DOUBLE, **kwargs
    )


@pytest.mark.parametrize("cfg", [LAYERED2, LAYERED3], ids=["nz2", "nz3"])
@pytest.mark.parametrize("param", ["S", "Qy", "Ubg", "Hi", "filtr", "gpi"])
def test_parameters_match(cfg, param):
    pyqg = pytest.importorskip("pyqg")
    jax_model = _make_jax(cfg)
    orig_model = _make_pyqg(pyqg, cfg)
    jax_param = np.asarray(getattr(jax_model, param))
    orig_param = np.asarray(getattr(orig_model, param))
    assert jax_param.shape == orig_param.shape
    assert np.allclose(jax_param, orig_param, atol=0, rtol=1e-10)


@pytest.mark.parametrize("cfg", [LAYERED2, LAYERED3], ids=["nz2", "nz3"])
def test_match_final_step(cfg):
    pyqg = pytest.importorskip("pyqg")
    num_steps = 100
    jax_model = _make_jax(cfg)
    start_jax_state = jax_model.create_initial_state(jax.random.key(0))
    orig_model = _make_pyqg(pyqg, cfg)
    orig_model.set_q(np.asarray(start_jax_state.q).copy().astype(np.float64))

    @jax.jit
    def do_jax_steps(init_state):
        stepper = pyqg_jax.steppers.AB3Stepper(dt=DT)
        stepped_model = pyqg_jax.steppers.SteppedModel(model=jax_model, stepper=stepper)
        final_state, _ = jax.lax.scan(
            lambda carry, _: (stepped_model.step_model(carry), None),
            stepped_model.initialize_stepper_state(init_state),
            None,
            length=num_steps,
        )
        return final_state

    final_jax_state = do_jax_steps(start_jax_state)
    for _ in range(num_steps):
        orig_model._step_forward()
    assert orig_model.tc == final_jax_state.tc
    assert np.allclose(final_jax_state.state.qh, orig_model.qh, atol=0, rtol=1e-8)


@pytest.mark.parametrize("nz", [2, 3, 4])
@pytest.mark.parametrize("nx,ny", [(16, 32), (17, 15)])
def test_rectangular_stepping(nz, nx, ny):
    jax_model = pyqg_jax.steppers.SteppedModel(
        model=pyqg_jax.layered_model.LayeredModel(
            nx=nx, ny=ny, nz=nz, f=1e-4, precision=pyqg_jax.state.Precision.DOUBLE
        ),
        stepper=pyqg_jax.steppers.AB3Stepper(dt=DT),
    )
    init_state = jax_model.create_initial_state(jax.random.key(0))

    @jax.jit
    def do_jax_steps(init_state):
        final_state, _ = jax.lax.scan(
            lambda carry, _: (jax_model.step_model(carry), None),
            init_state,
            None,
            length=3,
        )
        return final_state.state.q

    final_state = do_jax_steps(init_state)
    assert np.all(np.isfinite(final_state))


def test_requires_at_least_two_layers():
    with pytest.raises(ValueError, match="nz >= 2"):
        pyqg_jax.layered_model.LayeredModel(nz=1, f=1e-4)


def test_rejects_wrong_array_size():
    with pytest.raises(ValueError, match="shape"):
        pyqg_jax.layered_model.LayeredModel(nz=3, f=1e-4, U=jnp.zeros(2))


@pytest.mark.parametrize("nz", [2, 3, 4])
def test_grad_through_rollout(nz):
    # gradients w.r.t. beta must be finite, including through the general
    # nz batched inverse
    def loss(beta):
        model = pyqg_jax.layered_model.LayeredModel(
            nx=24, nz=nz, f=1e-4, beta=beta, precision=pyqg_jax.state.Precision.DOUBLE
        )
        stepper = pyqg_jax.steppers.AB3Stepper(dt=DT)
        stepped = pyqg_jax.steppers.SteppedModel(model=model, stepper=stepper)
        state = stepped.create_initial_state(jax.random.key(0))
        state, _ = jax.lax.scan(
            lambda c, _: (stepped.step_model(c), None), state, None, length=10
        )
        return jnp.sum(jnp.abs(stepped.get_full_state(state).ph) ** 2).real

    g = jax.grad(loss)(1.5e-11)
    assert np.isfinite(float(g))


@pytest.mark.parametrize("nz", [2, 3])
def test_tree_flatten_roundtrip(nz):
    cfg = LAYERED2 if nz == 2 else LAYERED3
    model = _make_jax(cfg)
    leaves, treedef = jax.tree_util.tree_flatten(model)
    restored_model = jax.tree_util.tree_unflatten(treedef, leaves)
    same_shape = jax.tree_util.tree_map(
        lambda a, b: np.asarray(a).shape == np.asarray(b).shape,
        model,
        restored_model,
    )
    assert jax.tree_util.tree_all(same_shape)
