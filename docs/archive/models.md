# Model conventions

For the first two cases, the instrument templates follow the equal-arm `AnalyticNoise` model in the
[LISA Data Challenge documentation](https://lisa.pages.in2p3.fr/LDC/_modules/ldc/lisa/noise/noise.html).
The mathematical noise allocations and transfer functions are implemented
directly; this repo does not depend on LDC or copy its implementation.

- Frequency is in Hz; plots use mHz.
- PSDs refer to dimensionless fractional frequency and have units 1/Hz.
- Arm length is 2.5 Gm; the baseline acceleration and displacement ASDs are
  3 fm s⁻²/√Hz and 15 pm/√Hz.
- `XYZ_TO_AET` is an **orthonormal** rotation. Normalizations from other
  conventions must be converted before comparing auto spectra.
- These are first-generation equal-arm Michelson templates. They are not the
  orbit-dependent second-generation TDI response used for the paper fits.
- All six links share the same component levels. In this approximation the
  A/E/T covariance is diagonal; XYZ auto spectra are equal and cross spectra
  are nonzero and real.

## Observation models

The scalar example uses `P = S × chi_square(n)` and `counts = n`. Its log
likelihood, omitting data-only constants, is

`−½ Σ [n log S + P/S]`.

The XYZ example samples proper complex vectors `z ~ CN(0, S)` with
`E[z zᴴ] = S`. It stores **sums** `Y = Σ z zᴴ`, not averages. With `n`
replicates its matrix log likelihood is

`−Σ_f [n log det S + trace(S⁻¹ Y)]`.

The covariance and pseudo-covariance normalizations are tested separately.
Fixed baseline whitening changes this log likelihood by a data-independent,
parameter-independent additive constant, so it preserves the posterior.

The scalar holdout is a fixed set of time rows. The XYZ example also saves
an independent replicate set for predictive checks. These examples do not
select spline complexity or noise-model variants on their evaluation data.

For a more realistic comparison, retain the same normalization checks while
adding a real time-domain generator and its actual transform, orbital
response, unequal arms or foregrounds. Do not relabel independent coefficients
as transformed WDM/Fourier data.

## Seven-day time-domain case

The third case uses unequal, evolving light travel times, the TDI-2 mixing
matrix and stationary TM/OMS link spectra from
[Noisy Pipelines I](https://arxiv.org/html/2610.28726v1). Its nominal noise
levels and fractional-frequency units agree with the first two cases;
its additional TDI-2 response factor and orbit geometry change the spectral
matrices. Do not compare the raw XYZ amplitudes across cases without their
transfer functions.

Orbit-epoch filters act on shared stationary Fourier sources and are
interpolated in time. A real gate/window and selected Fourier transform
produce the observations. The [exact benchmark](likelihood.md) propagates
those same operators, retaining all correlations for its compressed
features. The main [full matrix spline correction](reference_matrix.md)
uses a window-aware nominal reference, adaptive Whittle statistics and
analytically marginalised Cholesky theta; it is an independent-frequency
approximation. This is an explicit discretized, quasi-static simulation, without
the paper's full instrument and conditioning chain.
