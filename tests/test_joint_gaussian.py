"""Independent time-domain checks of the nonstationary, gapped likelihood."""

import numpy as np

from lisa_case_studies.joint_gaussian import (
    TVSimulator,
    covariance_bank,
    feature_vector,
    link_psds,
    source_bands,
    whiten_bank,
)
from lisa_case_studies.orbit_reference import link_transfers
from lisa_case_studies.spline_recovery import fit_splines


def small_simulator():
    n, dt = 32, 5.0
    centres, labels, basis = source_bands(n, dt, 8)
    epochs = np.array([0.0, 75.0, 155.0])
    delays = np.array(
        [
            [8.29, 8.31, 8.35, 8.34, 8.30, 8.32],
            [8.34, 8.28, 8.31, 8.29, 8.35, 8.30],
            [8.28, 8.36, 8.33, 8.31, 8.29, 8.34],
        ]
    )
    return TVSimulator(n, dt, epochs, delays, labels), centres, basis


def test_joint_bank_matches_direct_real_time_operator():
    sim, centres, _ = small_simulator()
    window = np.hanning(sim.n)
    window[9:14] = 0
    bins = np.array([2, 3, 7])
    bank = covariance_bank(sim, {"gap": window}, bins, len(centres))["gap"]
    expected = np.zeros_like(bank)
    # Independent calculation: apply each real latent Fourier coordinate
    # through the full time-domain simulator, then measure its features.
    for j in range(1, sim.n // 2 + 1):
        powers = link_psds(sim.frequency[[j]])[:, 0]
        for c in range(2):
            for source in range(6):
                for quadrature in range(1 if j == sim.n // 2 else 2):
                    series = np.zeros((sim.n, 3))
                    amplitude = np.sqrt(sim.n / (2 * sim.dt) * powers[c])
                    if j != sim.n // 2:
                        amplitude /= np.sqrt(2)
                    amplitude *= 1 if quadrature == 0 else 1j
                    for weight, delays in zip(sim.weights, sim.delays, strict=True):
                        h = link_transfers(sim.frequency[[j]], delays)[c][0, :, source]
                        if j == sim.n // 2:
                            h = h.real
                        coefficients = np.zeros((sim.n // 2 + 1, 3), complex)
                        coefficients[j] = h * amplitude
                        series += weight[:, None] * np.fft.irfft(
                            coefficients, n=sim.n, axis=0
                        )
                    feature = feature_vector(series, window, bins, sim.dt)
                    expected[c, sim.band_labels[j]] += np.outer(feature, feature)
    scale = np.max(np.abs(expected))
    np.testing.assert_allclose(bank / scale, expected / scale, rtol=1e-11, atol=1e-12)
    white, whitening, _ = whiten_bank(bank)
    np.testing.assert_allclose(
        white.sum(axis=(0, 1)), np.eye(len(whitening)), atol=2e-12
    )
    cov = bank.sum(axis=(0, 1))
    # Retain both gap-induced neighbouring-frequency and quadrature coupling.
    assert np.max(np.abs(cov[:3, 3:6])) > 0.01 * np.max(np.diag(cov))
    assert np.max(np.abs(cov[:9, 9:])) > 1e-5 * np.max(np.diag(cov))


def test_time_variation_and_shared_epoch_noise():
    sim, centres, _ = small_simulator()
    correction = np.zeros((2, len(centres)))
    changing = sim.simulate(correction, np.random.default_rng(4))
    fixed = TVSimulator(
        sim.n, sim.dt, sim.epochs, np.tile(sim.delays[0], (3, 1)), sim.band_labels
    )
    frozen = fixed.simulate(correction, np.random.default_rng(4))
    assert np.linalg.norm(changing - frozen) / np.linalg.norm(frozen) > 0.01
    # Identical filters at every epoch collapse to a single stationary filter.
    single = TVSimulator(sim.n, sim.dt, sim.epochs[:1], sim.delays[:1], sim.band_labels)
    np.testing.assert_allclose(
        frozen,
        single.simulate(correction, np.random.default_rng(4)),
        rtol=1e-12,
        atol=1e-32,
    )


def test_map_objective_and_stationarity_match_independent_dense_calculation():
    sim, centres, spline = small_simulator()
    window = np.hanning(sim.n)
    window[9:14] = 0
    bins = np.array([2, 3, 7])
    bank = covariance_bank(sim, {"gap": window}, bins, len(centres))["gap"]
    series = sim.simulate(np.zeros((2, len(centres))), np.random.default_rng(5))
    observed = feature_vector(series, window, bins, sim.dt)
    _, diagnostic = fit_splines(bank, observed, spline.basis, spline.penalty)
    white, whitening, _ = whiten_bank(bank)
    weights = np.asarray(diagnostic["weights"])
    y = whitening @ observed
    precision = 10 * spline.penalty + np.eye(weights.shape[1]) / 0.8**2

    def direct(flat):
        w = flat.reshape(weights.shape)
        covariance = np.einsum("ck,ckij->ij", np.exp(w @ spline.basis.T), white)
        return (
            np.linalg.slogdet(covariance)[1] + y @ np.linalg.solve(covariance, y)
        ) / 2 + np.einsum("ci,ij,cj->", w, precision, w) / 2

    np.testing.assert_allclose(
        diagnostic["objective"], direct(weights.ravel()), rtol=1e-11
    )
    h = 1e-5
    gradient = [
        (direct(weights.ravel() + h * d) - direct(weights.ravel() - h * d)) / (2 * h)
        for d in np.eye(weights.size)
    ]
    assert np.max(np.abs(gradient)) < 2e-4
