# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import numpy as np
import jax
import pyqg_jax


def _base():
    return pyqg_jax.qg_model.QGModel(nx=32, precision=pyqg_jax.state.Precision.DOUBLE)


def _ring(base, **kw):
    opts = {"k_in_forc": 2e-5, "k_out_forc": 6e-5, "mag_noise_forc": 1e-7}
    opts.update(kw)
    return pyqg_jax.parameterizations.ringforcing.apply_parameterization(base, **opts)


def _forcing_spectral(pmodel, base):
    state = pmodel.create_initial_state(jax.random.key(0))
    forced = pmodel.get_updates(state).model_state.qh
    unforced = base.get_updates(state.model_state).qh
    return np.asarray(forced - unforced), state


def test_forcing_supported_in_ring():
    base = _base()
    pmodel = _ring(base)
    dqh, _ = _forcing_spectral(pmodel, base)
    wvx = np.sqrt(np.asarray(base.k) ** 2 + np.asarray(base.l) ** 2)
    in_ring = (wvx > 2e-5) & (wvx <= 6e-5)
    outside = np.abs(dqh)[:, ~in_ring]
    assert outside.max() < 1e-20  # forcing vanishes outside the ring


def test_forcing_zero_mean():
    base = _base()
    dqh, _ = _forcing_spectral(_ring(base), base)
    # the (0, 0) spectral mode (domain mean) is zero
    assert np.abs(dqh[:, 0, 0]).max() < 1e-18


def test_zero_amplitude_no_forcing():
    base = _base()
    dqh, _ = _forcing_spectral(_ring(base, mag_noise_forc=0.0), base)
    assert np.abs(dqh).max() < 1e-20


def test_deterministic_given_state():
    base = _base()
    pmodel = _ring(base)
    a, _ = _forcing_spectral(pmodel, base)
    b, _ = _forcing_spectral(pmodel, base)
    assert np.array_equal(a, b)


def test_different_keys_differ():
    base = _base()
    d1, _ = _forcing_spectral(_ring(base, key=jax.random.key(1)), base)
    d2, _ = _forcing_spectral(_ring(base, key=jax.random.key(2)), base)
    assert not np.allclose(d1, d2)


def test_layers_surf_and_bottom():
    base = _base()
    for layers, forced_layer in [("surf", 0), ("bottom", -1)]:
        dqh, _ = _forcing_spectral(_ring(base, layers=layers), base)
        other = 1 if forced_layer == 0 else 0
        assert np.abs(dqh[other]).max() < 1e-20
        assert np.abs(dqh[forced_layer]).max() > 0


def test_steps_and_advances_prng():
    base = _base()
    pmodel = _ring(base)
    stepper = pyqg_jax.steppers.AB3Stepper(dt=3600.0)
    stepped = pyqg_jax.steppers.SteppedModel(model=pmodel, stepper=stepper)

    @jax.jit
    def run(init):
        return jax.lax.scan(
            lambda c, _: (stepped.step_model(c), None), init, None, length=10
        )[0]

    init = stepped.create_initial_state(jax.random.key(0))
    final = run(init)
    assert np.all(np.isfinite(np.asarray(final.state.model_state.q)))
    # the PRNG key in param_aux changed (forcing decorrelated in time)
    k0 = jax.random.key_data(init.state.param_aux.value)
    k1 = jax.random.key_data(final.state.param_aux.value)
    assert not np.array_equal(np.asarray(k0), np.asarray(k1))
