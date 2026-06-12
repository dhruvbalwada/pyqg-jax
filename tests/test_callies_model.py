# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import math
import warnings
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import pyqg_jax

CALLIES_PARAMS = {
    "nx": 64,
    "L": 5e5,
    "f": 1e-4,
    "Nm": 2e-3,
    "Nt": 8e-3,
    "Hm": 100.0,
    "Ht": 400.0,
    "Sm": 1e-4,
    "St": 1e-4,
    "hypodiff": 1e-16,
}
DT = 600.0


@pytest.mark.parametrize(
    "param",
    [
        # Model attributes
        "nz",
        "nx",
        "ny",
        "L",
        "W",
        "rek",
        "f",
        "g",
        "Nm",
        "Nt",
        "Hm",
        "Ht",
        "Sm",
        "St",
        "nu",
        "nun",
        "hypodiff",
        # Properties from kernel
        "nl",
        "nk",
        "kk",
        "ll",
        "Ubg",
        "filtr",
        # Properties from model
        "f2",
        "dk",
        "dl",
        "dx",
        "dy",
        "M",
        "wv2",
        "wv",
        "wv2i",
        # Properties from callies model
        "Hi",
        "H",
        "Qy",
    ],
)
def test_default_parameters_match(param):
    pyqg = pytest.importorskip("pyqg")
    if not hasattr(pyqg, "CalliesTwoEady"):
        pytest.skip("installed pyqg lacks CalliesTwoEady")
    jax_model = pyqg_jax.callies_model.CalliesTwoEady(
        dt=DT, precision=pyqg_jax.state.Precision.DOUBLE, **CALLIES_PARAMS
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        orig_model = pyqg.CalliesTwoEady(dt=DT, log_level=0, **CALLIES_PARAMS)
    jax_param = getattr(jax_model, param)
    orig_param = getattr(orig_model, param)
    if isinstance(orig_param, np.ndarray):
        np_jax_param = np.asarray(jax_param)
        assert jax_param.shape == orig_param.shape
        assert np.allclose(np_jax_param, orig_param, atol=0, rtol=1e-10)
    elif isinstance(orig_param, float):
        assert math.isclose(orig_param, float(jax_param), rel_tol=1e-12)
    elif isinstance(orig_param, int):
        assert jax_param == orig_param
    else:
        orig_param = np.asarray(orig_param)
        np_jax_param = np.asarray(jax_param)
        assert jax_param.shape == orig_param.shape
        assert np.allclose(np_jax_param, orig_param, atol=0, rtol=1e-10)


def test_match_final_step():
    pyqg = pytest.importorskip("pyqg")
    if not hasattr(pyqg, "CalliesTwoEady"):
        pytest.skip("installed pyqg lacks CalliesTwoEady")
    num_steps = 200
    jax_model = pyqg_jax.callies_model.CalliesTwoEady(
        dt=DT, precision=pyqg_jax.state.Precision.DOUBLE, **CALLIES_PARAMS
    )
    start_jax_state = jax_model.create_initial_state(jax.random.key(0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        orig_model = pyqg.CalliesTwoEady(
            log_level=0,
            dt=DT,
            tmax=DT * num_steps,
            twrite=num_steps + 10,
            **CALLIES_PARAMS,
        )
    orig_model.q = np.asarray(start_jax_state.q).copy().astype(np.float64)

    @jax.jit
    def do_jax_steps(init_state):
        stepper = pyqg_jax.steppers.AB3Stepper(dt=DT)
        stepped_model = pyqg_jax.steppers.SteppedModel(model=jax_model, stepper=stepper)
        final_state, _ = jax.lax.scan(
            lambda carry, _: (stepped_model.step_model(carry), None),
            stepped_model.initialize_stepper_state(init_state),
            None,
            length=num_steps + 1,
        )
        return final_state

    final_jax_state = do_jax_steps(start_jax_state)
    orig_model.run()
    assert orig_model.tc == final_jax_state.tc
    # AB3 + same filter application point -> roundoff-level agreement
    assert np.allclose(final_jax_state.state.q, orig_model.q, atol=0, rtol=1e-9)


@pytest.mark.parametrize("nx,ny", [(16, 32), (17, 15)])
def test_rectangular_stepping(nx, ny):
    jax_model = pyqg_jax.steppers.SteppedModel(
        model=pyqg_jax.callies_model.CalliesTwoEady(
            nx=nx, ny=ny, dt=DT, precision=pyqg_jax.state.Precision.DOUBLE
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


def test_initial_state_zero_mean():
    model = pyqg_jax.callies_model.CalliesTwoEady(
        nx=32, dt=DT, precision=pyqg_jax.state.Precision.DOUBLE
    )
    q = model.create_initial_state(jax.random.key(0)).q
    assert np.allclose(np.asarray(q).mean(axis=(-2, -1)), 0, atol=1e-15)


def test_grad_through_rollout():
    # gradients w.r.t. a physical parameter must be finite (overflow-safe
    # coth/csch in the inversion is what guarantees this)
    def loss(sm):
        model = pyqg_jax.callies_model.CalliesTwoEady(
            nx=32, dt=DT, Sm=sm, precision=pyqg_jax.state.Precision.DOUBLE
        )
        stepper = pyqg_jax.steppers.AB3Stepper(dt=DT)
        stepped = pyqg_jax.steppers.SteppedModel(model=model, stepper=stepper)
        state = stepped.create_initial_state(jax.random.key(0))
        state, _ = jax.lax.scan(
            lambda c, _: (stepped.step_model(c), None), state, None, length=20
        )
        return jnp.sum(jnp.abs(stepped.get_full_state(state).ph) ** 2).real

    g = jax.grad(loss)(1e-4)
    assert np.isfinite(float(g))


def test_tree_flatten_roundtrip():
    model = pyqg_jax.callies_model.CalliesTwoEady(dt=DT, **CALLIES_PARAMS)
    leaves, treedef = jax.tree_util.tree_flatten(model)
    restored_model = jax.tree_util.tree_unflatten(treedef, leaves)
    assert vars(restored_model) == vars(model)
