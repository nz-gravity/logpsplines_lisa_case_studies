"""Nonstationary XYZ simulation and exact Gaussian feature covariances.

All epochs act on the SAME stationary link-noise realization. Their smooth
time weights therefore preserve cross-epoch correlations. For computation,
smooth log-PSD corrections are constant within declared source-frequency
bands; the transfer functions retain the native Fourier resolution.
"""

from dataclasses import dataclass

import numpy as np
from log_psplines import SplineBasis
from scipy.linalg import solve_triangular

from .noise import C
from .orbit_reference import link_transfers


def link_psds(frequency):
    f = np.asarray(frequency)
    tm = (
        (3e-15) ** 2
        * (1 + (0.4e-3 / f) ** 2)
        * (1 + (f / 8e-3) ** 4)
        / (2 * np.pi * f * C) ** 2
    )
    oms = (15e-12) ** 2 * (1 + (2e-3 / f) ** 4) * (2 * np.pi * f / C) ** 2
    return np.stack([tm, oms])


def epoch_weights(time, epochs):
    """Piecewise linear partition of unity, constant beyond outer centres."""
    return np.stack([np.interp(time, epochs, row) for row in np.eye(len(epochs))])


def source_bands(n, dt, n_bands, n_interior=3):
    f = np.fft.rfftfreq(n, dt)
    edges = np.geomspace(f[1] * 0.999, f[-1] * 1.001, n_bands + 1)
    centres = np.sqrt(edges[:-1] * edges[1:])
    labels = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, n_bands - 1)
    spline = SplineBasis.from_grid(np.log(centres), n_interior)
    return centres, labels, spline


@dataclass
class TVSimulator:
    """Quasi-static, time-varying TDI-2 filters of shared link Fourier noise."""

    n: int
    dt: float
    epochs: np.ndarray
    delays: np.ndarray
    band_labels: np.ndarray

    @property
    def frequency(self):
        return np.fft.rfftfreq(self.n, self.dt)

    @property
    def weights(self):
        return epoch_weights(np.arange(self.n) * self.dt, self.epochs)

    def simulate(self, log_corrections, rng):
        """Return real (N,3) data; log_corrections has shape (2,n_bands)."""
        f = self.frequency
        power = link_psds(f[1:]) * np.exp(
            np.asarray(log_corrections)[:, self.band_labels[1:]]
        )
        noise = (
            rng.normal(size=(2, len(f) - 1, 6))
            + 1j * rng.normal(size=(2, len(f) - 1, 6))
        ) / np.sqrt(2)
        noise[:, -1] = rng.normal(size=(2, 6))
        noise *= np.sqrt(self.n / (2 * self.dt) * power[:, :, None])
        output = np.zeros((self.n, 3))
        for weight, delays in zip(self.weights, self.delays, strict=True):
            transfers = np.asarray(link_transfers(f[1:], delays))
            transfers[:, -1] = transfers[:, -1].real
            coefficients = np.zeros((len(f), 3), complex)
            coefficients[1:] = np.einsum("cfis,cfs->fi", transfers, noise)
            output += weight[:, None] * np.fft.irfft(coefficients, n=self.n, axis=0)
        return output

    def spectrum(self, time, frequency, log_corrections, band_labels):
        """Instantaneous quasi-static spectral matrix of the simulator."""
        weights = epoch_weights(np.asarray(time), self.epochs)
        accumulated = np.zeros((2, len(time), len(frequency), 3, 6), complex)
        for e, delays in enumerate(self.delays):
            h = np.asarray(link_transfers(frequency, delays))
            accumulated += h[:, None] * weights[e][None, :, None, None, None]
        power = link_psds(frequency) * np.exp(
            np.asarray(log_corrections)[:, band_labels]
        )
        return np.einsum("ctfis,ctfjs,cf->tfij", accumulated, accumulated.conj(), power)


def feature_vector(series, window, bins, dt):
    """Retain real AND imaginary components, frequency-major XYZ ordering."""
    transformed = np.fft.rfft(np.asarray(series) * window[:, None], axis=0)[bins]
    transformed *= np.sqrt(2 * dt / len(window))
    return np.r_[transformed.real.ravel(), transformed.imag.ravel()]


def covariance_bank(
    simulator, windows, bins, n_bands, *, chunk_size=1024, progress=None
):
    """Exact real covariance contributions per source band and component.

    Each window includes its actual gap gate. Correlations between epochs,
    channels, frequencies and quadratures are included. No circulant model
    for the OUTPUT data, proper-complex approximation or diagonal-bin
    likelihood is used. DC link noise is zero; real Nyquist noise is explicit.
    """
    n = simulator.n
    bins = np.asarray(bins, int)
    d = 6 * len(bins)
    result = {name: np.zeros((2, n_bands, d, d)) for name in windows}
    window_ffts = {
        name: np.fft.fft(simulator.weights * np.asarray(w)[None, :], axis=1) / n
        for name, w in windows.items()
    }
    f = simulator.frequency
    labels = simulator.band_labels
    for band in range(n_bands):
        indices = np.flatnonzero(
            (labels == band)
            & (np.arange(len(f)) > 0)
            & (np.arange(len(f)) < len(f) - 1)
        )
        for offset in range(0, len(indices), chunk_size):
            j = indices[offset : offset + chunk_size]
            transfer = np.asarray(
                [link_transfers(f[j], delays) for delays in simulator.delays]
            )
            powers = link_psds(f[j])
            for name, wf in window_ffts.items():
                wp = wf[:, (bins[:, None] - j[None, :]) % n]
                wm = wf[:, (bins[:, None] + j[None, :]) % n]
                for c in range(2):
                    plus = np.einsum(
                        "ekq,eqis->kiqs", wp, transfer[:, c], optimize=True
                    )
                    minus = np.einsum(
                        "ekq,eqis->kiqs", wm, transfer[:, c].conj(), optimize=True
                    )
                    a = (plus + minus).reshape(3 * len(bins), len(j) * 6)
                    b = (1j * (plus - minus)).reshape(a.shape)
                    amplitude = np.repeat(np.sqrt(powers[c] / 2), 6)
                    ga = np.concatenate([a.real, a.imag], axis=0) * amplitude
                    gb = np.concatenate([b.real, b.imag], axis=0) * amplitude
                    result[name][c, band] += ga @ ga.T + gb @ gb.T
        if progress is not None:
            progress(band + 1, n_bands)
    # Nyquist is one real source vector, not a proper-complex pair.
    j = n // 2
    transfers = np.asarray(
        [link_transfers(f[[j]], delays) for delays in simulator.delays]
    ).real[:, :, 0]
    for name, wf in window_ffts.items():
        window = wf[:, (bins - j) % n]
        for c in range(2):
            response = np.einsum("ek,eis->kis", window, transfers[:, c]).reshape(
                3 * len(bins), 6
            )
            real = np.concatenate([response.real, response.imag], axis=0)
            result[name][c, labels[j]] += link_psds(f[[j]])[c, 0] * (real @ real.T)
    return result


def whiten_bank(bank):
    """Fixed nominal-covariance whitening; no truth or fitted scales enter."""
    bank = np.asarray(bank)
    nominal = bank.sum(axis=(0, 1))
    diagonal = np.sqrt(np.diag(nominal))
    normalized = nominal / np.outer(diagonal, diagonal)
    root = np.linalg.cholesky(normalized)
    whitening = solve_triangular(root, np.diag(1 / diagonal), lower=True)
    whitened = np.einsum("ia,ckab,jb->ckij", whitening, bank, whitening, optimize=True)
    whitened = 0.5 * (whitened + whitened.swapaxes(-1, -2))
    return whitened, whitening, float(np.linalg.cond(normalized))
