"""Channel-level simulations shared by the four teaching notebooks.

The evolving example interpolates spectral-matrix roots at orbit epochs,
acting on shared latent Fourier noise. It is a Gaussian matrix demonstration,
not an instrument/link-noise simulation or a noncommuting TDI-delay solver.
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from .joint_gaussian import epoch_weights, feature_vector
from .orbit_reference import reference_components


def workspace():
    """Resolve a checkout from either its root or docs/notebooks directory."""
    for root in (Path.cwd(), *Path.cwd().parents):
        if (root / "src/lisa_case_studies").is_dir() and (
            root / "pyproject.toml"
        ).is_file():
            return root / "outputs" / "notebooks"
    raise FileNotFoundError("Open the notebook inside the case-study checkout.")


def load_data(name):
    path = workspace() / "data" / f"{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            "Run notebook 01_generate_data.ipynb first; it creates the shared datasets."
        )
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def matrix_root(matrix):
    """Hermitian PSD square root; clip only roundoff-negative eigenvalues."""
    values, vectors = np.linalg.eigh(matrix)
    scale = np.max(np.abs(values), axis=-1)
    if np.any(values[..., 0] < -1e-10 * scale):
        raise ValueError("The input spectral matrix is not positive semidefinite.")
    return (
        vectors * np.sqrt(np.maximum(values, 0))[..., None, :]
    ) @ vectors.conj().swapaxes(-1, -2)


def lisa_matrix(frequency, light_times):
    """Known full XYZ reference, in fractional-frequency squared / Hz."""
    return sum(reference_components(np.asarray(frequency), light_times))


def residual_matrix(frequency):
    """Prescribed smooth full-matrix departure; independent of fitted knots."""
    f = np.maximum(np.asarray(frequency), 1e-8)
    u = np.log(f / 0.004)
    h = np.stack(
        [
            0.35 * np.exp(-(((u + 0.5) / 1.1) ** 2)),
            -0.18 + 0.28 * np.exp(-(((u - 1.1) / 1.2) ** 2)),
            0.12 * np.tanh(u / 2),
        ],
        axis=-1,
    )
    theta = np.zeros((*f.shape, 3, 3), complex)
    theta[..., 1, 0] = 0.10 * np.exp(-(u**2) / 5) + 0.06j * np.tanh(u / 2)
    theta[..., 2, 0] = -0.07 * np.exp(-((u - 1) ** 2) / 4) + 0.04j
    theta[..., 2, 1] = 0.06 * np.tanh(u) - 0.03j * np.exp(-(u**2) / 6)
    lower = np.eye(3) + theta + theta @ theta
    return (lower * np.exp(h)[..., None, :]) @ lower.conj().swapaxes(-1, -2)


def amplitude(time, duration):
    """Slow channel-A amplitude envelope; PSD multiplier is its square."""
    u = np.asarray(time) / duration
    return np.exp(
        0.5 * (0.9 * np.sin(2 * np.pi * u) + 0.45 * np.exp(-(((u - 0.68) / 0.11) ** 2)))
    )


@dataclass
class MatrixSimulation:
    n: int
    dt: float
    epochs: np.ndarray
    light_times: np.ndarray

    @property
    def time(self):
        return np.arange(self.n) * self.dt

    @property
    def frequency(self):
        return np.fft.rfftfreq(self.n, self.dt)

    @cached_property
    def weights(self):
        return epoch_weights(self.time, self.epochs)

    def roots_at(self, frequency):
        roots = []
        for delays in self.light_times:
            spectrum = lisa_matrix(frequency, delays)
            values, vectors = np.linalg.eigh(spectrum)
            # Declared simulation floor for roundoff-level null modes, shared
            # by generation, the reference and the exact covariance controls.
            values = np.maximum(values, values[..., -1, None] * 1e-12)
            positive = (vectors * values[..., None, :]) @ vectors.conj().swapaxes(
                -1, -2
            )
            roots.append(np.linalg.cholesky(positive))
        return np.asarray(roots)

    @cached_property
    def roots(self):
        return self.roots_at(self.frequency[1:])

    def draw(self, seed, *, corrected=True):
        """Real XYZ; same latent white-noise realization at every orbit epoch."""
        rng = np.random.default_rng(seed)
        white = (
            rng.normal(size=(self.n // 2, 3)) + 1j * rng.normal(size=(self.n // 2, 3))
        ) / np.sqrt(2)
        white[-1] = rng.normal(size=3)
        white *= np.sqrt(self.n / (2 * self.dt))
        correction = (
            matrix_root(residual_matrix(self.frequency[1:]))
            if corrected
            else np.broadcast_to(np.eye(3), self.roots.shape[1:])
        )
        result = np.zeros((self.n, 3))
        for weight, root in zip(self.weights, self.roots, strict=True):
            transfer = root @ correction
            transfer[-1] = transfer[-1].real
            coefficients = np.zeros((self.n // 2 + 1, 3), complex)
            coefficients[1:] = np.einsum("fij,fj->fi", transfer, white)
            result += weight[:, None] * np.fft.irfft(coefficients, n=self.n, axis=0)
        return result

    def reference_factor(self, time, frequency):
        return np.einsum(
            "et,efij->tfij",
            epoch_weights(np.asarray(time), self.epochs),
            self.roots_at(frequency),
        )

    def spectrum(self, time, frequency, *, corrected=True):
        root = self.reference_factor(time, frequency)
        correction = (
            residual_matrix(frequency)
            if corrected
            else np.broadcast_to(np.eye(3), root.shape[1:])
        )
        return root @ correction @ root.conj().swapaxes(-1, -2)

    def window_references(self, windows):
        """Exact nominal same-frequency blocks; Whittle fit still approximate."""
        n = self.n
        wf = {
            name: np.fft.fft(self.weights * np.asarray(w)[None], axis=1) / n
            for name, w in windows.items()
        }
        root = self.roots.copy()
        root[:, -1] = root[:, -1].real
        result = {name: np.zeros((n, 3, 3), complex) for name in windows}
        for e in range(len(root)):
            for d in range(e + 1):
                positive = root[e] @ root[d].conj().swapaxes(-1, -2)
                full = np.concatenate(
                    [np.zeros((1, 3, 3)), positive, positive[-2::-1].conj()]
                )
                fft = np.fft.fft(full, axis=0)
                for name, window in wf.items():
                    value = np.fft.ifft(
                        fft * np.fft.fft(window[e] * window[d].conj())[:, None, None],
                        axis=0,
                    )
                    result[name] += (
                        value if e == d else value + value.conj().swapaxes(-1, -2)
                    )
        return {
            name: 0.5
            * (value[: n // 2 + 1] + value[: n // 2 + 1].conj().swapaxes(-1, -2))
            for name, value in result.items()
        }

    def feature_covariance(self, window, bins, *, correction=None, chunk=1024):
        """Exact joint real covariance, including cross-frequency/pseudo terms.

        correction is an optional (n/2,3,3) residual spectral matrix on the
        positive native grid. This evaluates a point model; it does not fit
        or marginalise theta under this different, joint likelihood.
        """
        bins = np.asarray(bins, int)
        wf = np.fft.fft(self.weights * np.asarray(window)[None], axis=1) / self.n
        correction_root = (
            np.broadcast_to(np.eye(3), self.roots.shape[1:])
            if correction is None
            else matrix_root(correction)
        )
        transfer = self.roots @ correction_root[None]
        transfer[:, -1] = transfer[:, -1].real
        result = np.zeros((6 * len(bins), 6 * len(bins)))
        for start in range(1, self.n // 2, chunk):
            j = np.arange(start, min(start + chunk, self.n // 2))
            wp = wf[:, (bins[:, None] - j[None]) % self.n]
            wm = wf[:, (bins[:, None] + j[None]) % self.n]
            plus = np.einsum("ekq,eqis->kiqs", wp, transfer[:, j - 1], optimize=True)
            minus = np.einsum(
                "ekq,eqis->kiqs", wm, transfer[:, j - 1].conj(), optimize=True
            )
            a = (plus + minus).reshape(3 * len(bins), -1) / np.sqrt(2)
            b = (1j * (plus - minus)).reshape(a.shape) / np.sqrt(2)
            ga, gb = np.r_[a.real, a.imag], np.r_[b.real, b.imag]
            result += ga @ ga.T + gb @ gb.T
        nyquist = np.einsum(
            "ek,eis->kis", wf[:, (bins - self.n // 2) % self.n], transfer[:, -1]
        ).reshape(3 * len(bins), 3)
        g = np.r_[nyquist.real, nyquist.imag]
        return result + g @ g.T

    def features(self, series, window, bins):
        return feature_vector(series, window, bins, self.dt)


def load_orbit_simulation():
    data = load_data("short_xyz")
    return data, MatrixSimulation(
        int(data["n"]), float(data["dt"]), data["epochs"], data["light_times"]
    )


def data_receipt(path, **settings):
    """Small source receipt alongside the notebook-generated data."""
    import hashlib
    import json
    from importlib.metadata import version

    receipt = dict(
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        library_version=version("LogPSplinePSD"),
        numpy_version=version("numpy"),
        **settings,
    )
    (path / "generation.json").write_text(json.dumps(receipt, indent=2))
