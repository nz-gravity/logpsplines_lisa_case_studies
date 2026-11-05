"""Equal-arm SciRDv1 instrumental spectra in fractional-frequency PSD (1/Hz).

Mathematical model: LDC AnalyticNoise.psd(equal_arms=True), with orthonormal
A/E/T. First-generation Michelson transfer functions are used. No foreground,
orbital response, unequal arms or link asymmetry is included.

Source: https://lisa.pages.in2p3.fr/LDC/_modules/ldc/lisa/noise/noise.html
"""

import numpy as np

C = 299792458.0
ARM_LENGTH = 2.5e9
XYZ_TO_AET = np.array(
    [
        [-1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
        [1 / np.sqrt(6), -2 / np.sqrt(6), 1 / np.sqrt(6)],
        [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
    ]
)


def aet_components(frequency):
    """Return (acceleration, OMS) one-sided diagonal A/E/T PSD templates."""
    f = np.asarray(frequency, dtype=float)
    if f.ndim != 1 or not len(f) or not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("Frequency must be a finite positive vector in Hz")
    x = 2 * np.pi * f * ARM_LENGTH / C
    sin2, cos = np.sin(x) ** 2, np.cos(x)
    # Physical displacement/acceleration allocations converted to fractional
    # frequency. The free inference parameters later multiply PSD, not ASD.
    acceleration = (3e-15) ** 2 * (1 + (0.4e-3 / f) ** 2) * (1 + (f / 8e-3) ** 4)
    pm = acceleration / (2 * np.pi * f * C) ** 2
    op = (15e-12) ** 2 * (1 + (2e-3 / f) ** 4) * (2 * np.pi * f / C) ** 2
    a_acc = 32 * sin2 * (1 + cos + cos**2) * pm
    a_oms = 8 * sin2 * (2 + cos) * op
    # Half-angle expression avoids cancellation in the near-null T channel.
    one_minus_cos = 2 * np.sin(x / 2) ** 2
    t_acc = 32 * sin2 * one_minus_cos**2 * pm
    t_oms = 16 * sin2 * one_minus_cos * op
    return np.stack([a_acc, a_acc, t_acc], axis=-1), np.stack(
        [a_oms, a_oms, t_oms], axis=-1
    )


def xyz_components(frequency):
    """Rotate each independent noise component into full XYZ covariance."""
    acc, oms = aet_components(frequency)
    return tuple(
        np.einsum("ai,fa,aj->fij", XYZ_TO_AET, value, XYZ_TO_AET)
        for value in (acc, oms)
    )


def time_scales(time):
    """Smooth illustrative PSD-scale drift, not an orbital noise prediction."""
    u = np.asarray(time)
    acc = np.exp(
        0.9 * np.sin(2 * np.pi * u) + 0.5 * np.exp(-(((u - 0.68) / 0.11) ** 2))
    )
    oms = np.exp(0.25 * np.cos(2 * np.pi * u + 0.4))
    return acc, oms


def sample_real_powers(spectrum, replicates, rng):
    """Independent local real Gaussian coefficients: P/S ~ chi-square(n)."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    return np.asarray(spectrum) * rng.chisquare(replicates, np.shape(spectrum))


def sample_complex(spectrum, replicates, rng):
    """Independent proper-complex vectors with E[z z^H] = S, not S/2."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    spectrum = np.asarray(spectrum)
    root = np.linalg.cholesky(spectrum)
    shape = (*spectrum.shape[:-1], replicates)
    white = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)
    return root @ white


def coherence(spectrum):
    """Magnitude-squared coherence of a spectral matrix."""
    values = np.asarray(spectrum)
    diagonal = np.diagonal(values, axis1=-2, axis2=-1).real
    return np.abs(values) ** 2 / (diagonal[..., :, None] * diagonal[..., None, :])
