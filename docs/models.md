# Model conventions

The instrument templates follow the equal-arm `AnalyticNoise` model in the
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
