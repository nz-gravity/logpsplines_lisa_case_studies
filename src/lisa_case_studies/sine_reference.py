"""Orbit-derived sine dips around a frozen Michelson covariance envelope.

The data generator evolves the complete covariance. This cheaper reference
only evolves three loop factors, keeping the first-generation envelope at
the start of the week. It therefore deliberately approximates the orbit
response. No injected spectrum or fitted noise allocation is consulted.
"""

from dataclasses import dataclass

import numpy as np

from .orbit_reference import orbit_light_times, reference_components
from .tutorial_data import MatrixSimulation


def loop_times(light_times):
    """XYZ four-link loop times in seconds, in the declared LINKS order."""
    t12, t23, t31, t13, t32, t21 = np.asarray(light_times).T
    r1, r2, r3 = t12 + t21, t23 + t32, t31 + t13
    return np.stack([r1 + r3, r1 + r2, r2 + r3], axis=-1)


def sine_gains(frequency, light_times):
    """1-exp(-2 pi i f tau) = 2i exp(-pi i f tau) sin(pi f tau)."""
    phase = np.pi * np.asarray(frequency)[..., None] * loop_times(light_times)
    return 2j * np.exp(-1j * phase) * np.sin(phase)


@dataclass
class SineReference(MatrixSimulation):
    """Same observation operators as MatrixSimulation, approximate PSD roots.

    The frozen envelope is derived from geometry, never from the generated
    data or residual_matrix. Conditional theta integration remains wholly in
    LogPSplinePSD. The relative eigenvalue floor follows the generator.
    """

    envelope_light_times: np.ndarray | None = None

    def spectrum(self, time, frequency, *, corrected=False):
        """The nominal reference never supplies the injected residual."""
        if corrected:
            raise ValueError(
                "Pass an inferred correction explicitly during reconstruction"
            )
        return super().spectrum(time, frequency, corrected=False)

    def draw(self, seed, *, corrected=False):
        """Draw from the nominal reference, without an oracle correction."""
        if corrected:
            raise ValueError("SineReference does not use the injected residual")
        return super().draw(seed, corrected=False)

    def roots_at(self, frequency):
        anchor = self.envelope_light_times
        if anchor is None:
            anchor = self.light_times[0]
        envelope = sum(reference_components(frequency, anchor, second_generation=False))
        roots = []
        for delays in self.light_times:
            gain = sine_gains(frequency, delays)
            spectrum = gain[..., :, None] * envelope * gain[..., None, :].conj()
            values, vectors = np.linalg.eigh(spectrum)
            values = np.maximum(values, values[..., -1, None] * 1e-12)
            positive = (vectors * values[..., None, :]) @ vectors.conj().swapaxes(
                -1, -2
            )
            roots.append(np.linalg.cholesky(positive))
        return np.asarray(roots)


def sine_reference_for(simulation):
    """Use known loop times at every epoch, freezing the envelope on day 0."""
    anchor, _ = orbit_light_times(np.array([0.0]))
    return SineReference(
        simulation.n,
        simulation.dt,
        simulation.epochs,
        simulation.light_times,
        envelope_light_times=anchor[0],
    )
