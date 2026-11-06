"""Delay-factor identities, deliberate model mismatch, and block counts."""

import numpy as np
from log_psplines.inference.model import prepare_channel_model, prepare_model
from numpyro.infer.util import log_density

from lisa_case_studies.notebook_tools import reference_config, reference_start
from lisa_case_studies.orbit_reference import reference_components
from lisa_case_studies.reference_matrix import coefficient_data
from lisa_case_studies.sine_reference import SineReference, loop_times, sine_gains
from lisa_case_studies.tutorial_data import MatrixSimulation


def test_sine_gain_matches_delay_operator_and_orbit_nulls():
    delays = np.array([8.29, 8.31, 8.35, 8.34, 8.30, 8.32])
    f = np.linspace(0.0004, 0.099, 107)
    expected = 1 - np.exp(-2j * np.pi * f[:, None] * loop_times(delays))
    np.testing.assert_allclose(sine_gains(f, delays), expected, rtol=1e-12, atol=1e-15)
    for channel, period in enumerate(loop_times(delays)):
        fnull = np.arange(1, 4) / period
        assert np.max(abs(sine_gains(fnull, delays)[:, channel])) < 1e-14


def test_frozen_sine_factorization_matches_full_response_but_evolving_envelope_is_approximate():
    anchor = np.array([8.29, 8.31, 8.35, 8.34, 8.30, 8.32])
    later = anchor * np.array([1.01, 0.99, 1.005, 0.995, 1.01, 0.99])
    f = np.array([0.001, 0.017, 0.0299, 0.041, 0.0601, 0.0899])
    ref = SineReference(64, 5, np.array([0, 315]), np.array([anchor, later]))
    actual = ref.spectrum(ref.epochs, f)
    full = MatrixSimulation(64, 5, ref.epochs, ref.light_times)
    truth = full.spectrum(ref.epochs, f, corrected=False)
    np.testing.assert_allclose(actual[0], truth[0], rtol=1e-10, atol=1e-55)
    envelope = sum(reference_components(f, anchor, second_generation=False))
    gain = sine_gains(f, later)
    expected_later = gain[..., :, None] * envelope * gain[..., None, :].conj()
    np.testing.assert_allclose(actual[1], expected_later, rtol=1e-10, atol=1e-55)
    # The fitted reference must not silently follow the generator's full
    # evolving envelope: only its three delay/sine factors evolve.
    scale = np.max(np.abs(truth[1]))
    assert np.max(np.abs(actual[1] - truth[1])) / scale > 1e-3


def test_replicate_statistics_keep_full_complex_covariance_and_eight_counts():
    rng = np.random.default_rng(19)
    coefficients = rng.normal(size=(11, 3, 8)) + 1j * rng.normal(size=(11, 3, 8))
    data = coefficient_data(coefficients, np.linspace(0.001, 0.099, 11))
    expected = sum(
        np.einsum("fi,fj->fij", coefficients[:, :, j], coefficients[:, :, j].conj())
        for j in range(8)
    )
    np.testing.assert_allclose(data.Y, expected, rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(data.counts, np.full(11, 8))
    assert data.Nb == 8


def test_reference_start_changes_only_initial_values_and_preserves_log_target():
    rng = np.random.default_rng(71)
    z = rng.normal(size=(49, 3)) + 1j * rng.normal(size=(49, 3))
    data = coefficient_data(z, np.linspace(0.001, 0.099, len(z)))
    config, _ = reference_config(data)
    kwargs, _ = prepare_model(data, config)
    baseline_model, baseline_kwargs, empirical = prepare_channel_model(kwargs, 2)
    with reference_start():
        model, channel_kwargs, initial = prepare_channel_model(kwargs, 2)
    assert not np.allclose(empirical["weights_delta_2_raw"], 0)
    np.testing.assert_array_equal(initial["weights_delta_2_raw"], 0)
    assert all(
        value == 0.3 for name, value in initial.items() if name.startswith("sigma")
    )
    before, _ = log_density(baseline_model, (), baseline_kwargs, initial)
    after, _ = log_density(model, (), channel_kwargs, initial)
    assert np.isfinite(before)
    np.testing.assert_allclose(before, after, rtol=1e-12)
    _, _, restored = prepare_channel_model(kwargs, 2)
    np.testing.assert_array_equal(
        restored["weights_delta_2_raw"], empirical["weights_delta_2_raw"]
    )
