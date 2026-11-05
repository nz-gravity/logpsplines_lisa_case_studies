"""Orbit-dependent TDI-2 reference and matrix-valued window convolution.

The mixing matrix and link covariance are Eqs. 2, 9, 28--29 of
https://arxiv.org/html/2610.28726v1. No fitted dip positions are introduced.
Epoch averaging is an explicitly local-stationary approximation.
"""

import numpy as np

from .noise import C

LINKS = np.array([12, 23, 31, 13, 32, 21])
DAY = 86400.0


def orbit_light_times(time, *, model="keplerian"):
    """Six directed photon travel times from a physical orbit model [s].

    Keplerian orbits are an offline approximation, not the paper's ESA orbit.
    The ESA option fetches lisaorbits' versioned trailing ephemerides.
    """
    from lisaorbits import KeplerianOrbits, OEMOrbits

    time = np.asarray(time, dtype=float)
    if model == "keplerian":
        orbit = KeplerianOrbits()
        origin = 90 * DAY
    elif model == "esa":
        orbit = OEMOrbits.from_included("esa-trailing", version="1.0.0")
        origin = 2073211130.8175
    else:
        raise ValueError("orbit model must be keplerian or esa")
    return np.asarray(orbit.compute_ltt(origin + time, link=LINKS)), origin


def mixing_matrix(frequency, light_times):
    """TDI-2 map from the six directed eta links to XYZ at one epoch."""
    f = np.asarray(frequency, dtype=float)
    delays = np.asarray(light_times, dtype=float)
    if f.ndim != 1 or np.any(f <= 0) or not np.isfinite(f).all():
        raise ValueError("frequencies must be finite and positive")
    if delays.shape != (6,) or not np.isfinite(delays).all() or np.any(delays <= 0):
        raise ValueError("six positive light times in LINK order are required")
    z = np.exp(-2j * np.pi * f[:, None] * delays)
    z12, z23, z31, z13, z32, z21 = z.T
    r1, r2, r3 = z12 * z21, z23 * z32, z31 * z13
    mixing = np.zeros((len(f), 3, 6), dtype=complex)
    mixing[:, 0, [0, 2, 3, 5]] = np.stack(
        [1 - r3, -z13 * (1 - r1), -(1 - r1), z12 * (1 - r3)], axis=-1
    )
    mixing[:, 1, [0, 1, 4, 5]] = np.stack(
        [-z21 * (1 - r2), 1 - r1, z23 * (1 - r1), -(1 - r2)], axis=-1
    )
    mixing[:, 2, [1, 2, 3, 4]] = np.stack(
        [-z32 * (1 - r3), 1 - r2, z31 * (1 - r2), -(1 - r3)], axis=-1
    )
    mixing *= np.stack([1 - r1 * r3, 1 - r1 * r2, 1 - r2 * r3], axis=-1)[..., None]
    return mixing


def link_transfers(frequency, light_times):
    """Maps from six independent TM and six independent OMS sources to XYZ."""
    mixing = mixing_matrix(frequency, light_times)
    z = np.exp(-2j * np.pi * np.asarray(frequency)[:, None] * np.asarray(light_times))
    tm = mixing.copy()
    for a, b in ((0, 5), (1, 4), (2, 3)):
        tm[:, :, a] += mixing[:, :, b] * z[:, b, None]
        tm[:, :, b] += mixing[:, :, a] * z[:, a, None]
    return tm, mixing


def tdi2_kernels(frequency, light_times):
    """Full Hermitian TM/OMS transfer matrices at one orbit epoch."""
    tm_transfer, mixing = link_transfers(frequency, light_times)
    oms = mixing @ np.swapaxes(mixing.conj(), -1, -2)
    tm = tm_transfer @ np.swapaxes(tm_transfer.conj(), -1, -2)
    return tm, oms


def reference_components(frequency, light_times):
    """Physical fractional-frequency PSD matrices, one epoch or epoch average."""
    f = np.asarray(frequency)
    delays = np.atleast_2d(light_times)
    kernels = [np.zeros((len(f), 3, 3), complex) for _ in range(2)]
    for epoch in delays:
        for total, value in zip(kernels, tdi2_kernels(f, epoch), strict=True):
            total += value / len(delays)
    acceleration = (
        (3e-15) ** 2
        * (1 + (0.4e-3 / f) ** 2)
        * (1 + (f / 8e-3) ** 4)
        / (2 * np.pi * f * C) ** 2
    )
    oms = (15e-12) ** 2 * (1 + (2e-3 / f) ** 4) * (2 * np.pi * f / C) ** 2
    return tuple(
        kernel * power[:, None, None]
        for kernel, power in zip(kernels, (acceleration, oms), strict=True)
    )


def two_sided_matrix(spectrum, n):
    """Extend one-sided real-series CSD with S(-f)=conjugate(S(f))."""
    values = np.asarray(spectrum, complex)
    if n % 2 or values.shape != (n // 2 + 1, 3, 3):
        raise ValueError("require an even time grid and (n/2+1,3,3) spectrum")
    values = values.copy()
    values[0] = values[0].real
    values[-1] = values[-1].real
    return np.concatenate([values, values[-2:0:-1].conj()], axis=0)


def windowed_matrix(spectrum, window):
    """Exact same-frequency 3x3 covariance for a stationary circulant surrogate.

    GapLike's scalar convolved diagonal is extended to complex cross spectra.
    This does NOT include the cross-frequency covariance induced by a gap.
    """
    w = np.asarray(window, float)
    full = two_sided_matrix(spectrum, len(w))
    kernel = np.abs(np.fft.fft(w)) ** 2 / len(w) ** 2
    convolved = np.fft.ifft(
        np.fft.fft(full, axis=0) * np.fft.fft(kernel)[:, None, None], axis=0
    )
    return convolved[: len(w) // 2 + 1]


def simulate_circulant_xyz(spectrum, dt, rng):
    """Real XYZ data with the stated stationary, orbit-AVERAGED covariance."""
    spectrum = np.asarray(spectrum, complex).copy()
    n = 2 * (len(spectrum) - 1)
    spectrum[0] = spectrum[0].real
    spectrum[-1] = spectrum[-1].real
    eigenvalues, rotation = np.linalg.eigh(spectrum)
    scale = np.max(np.abs(eigenvalues), axis=-1)
    if np.any(eigenvalues[:, 0] < -1e-9 * scale):
        raise ValueError("input spectrum is not positive semidefinite")
    root = rotation * np.sqrt(np.maximum(eigenvalues, 0))[:, None, :]
    white = (
        rng.normal(size=(len(spectrum), 3)) + 1j * rng.normal(size=(len(spectrum), 3))
    ) / np.sqrt(2)
    white[0] = rng.normal(size=3)
    white[-1] = rng.normal(size=3)
    coefficients = np.einsum("fij,fj->fi", root, white) * np.sqrt(n / (2 * dt))
    return np.fft.irfft(coefficients, n=n, axis=0)
