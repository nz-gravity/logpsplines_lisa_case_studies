# Seven-day evolving XYZ with gaps

This case simulates **120,960 real XYZ samples over seven days** at a
five-second cadence. A parametric orbit response places the moving dips;
a full multivariate log-P-spline learns a smooth correction to that response.
Continuous and gapped observations use identical priors, knots, adaptive bins
and final evaluation support.

```sh
uv sync --locked --extra gapped
uv run --extra gapped lisa-demo-gaps --output outputs/seven_day
```

## Reference plus a full matrix correction

Write the known physical reference as $S_0(t,f)=L_0(t,f)L_0(t,f)^\dagger$.
The recovered spectrum is

$$S(t,f)=L_0(t,f)\,R(f)\,L_0(t,f)^\dagger.$$

The positive-definite matrix $R(f)$ uses **nine real spline fields**:
three log innovation variances and the real/imaginary parts of three
cross-channel Cholesky regression coefficients. This learns changes in
cross spectra and coherence as well as auto spectra. $R=I$ gives the
uncorrected parametric reference. Dip positions are not additional sine-fit
parameters.

The residual correction is frequency-dependent. Orbital evolution supplies
the time variation over this short segment; freely varying residual time
surfaces are not inferred. This follows the short-segment idea in
[Burke et al., *Noisy Pipelines I*](https://arxiv.org/html/2610.28726v1),
using offline Keplerian orbits instead of their ESA ephemerides.

The simulator applies eight orbit-epoch TDI-2 response filters to the
**same stationary link-noise realization**, interpolating their time-domain
outputs with a partition of unity. It is genuinely time-varying, but remains
a quasi-static approximation to noncommuting TDI delays. It omits the
paper's full instrument, filtering and decimation chain. Its declared
source-frequency bands discretize the injected secondary spectra; those
bands are distinct from adaptive likelihood bins.

## Adaptive bins and marginalised theta

The main fit uses all **59,634 native Fourier bins** from 0.4 to 99 mHz.
It whitens each coefficient by its own orbit-and-window reference before
pooling. The nominal reference includes all epoch cross terms and the actual
GapLike gap gate and taper.

The library's adaptive matrix pilot selects bin edges from **gapped training
data only**. The same partition is frozen for both observations before
inference. Full complex cross-products and their counts are summed from
unsmoothed native data. Prespecified 0.1 mHz neighbourhoods around the
30, 60 and 90 mHz nulls retain native bins. In the saved run, 59,634 native
frequencies become 599 aggregates, with widths from 1 to 256.

Both fits use twelve cubic basis functions for each of the nine fields,
on log-frequency coordinates. The pinned library analytically marginalises
the six Gaussian theta spline fields while fitting the three diagonal
fields and smoothing scales with VI. It restores theta from the exact
conditional Gaussian before reconstructing matrices. Three VI restarts use
6,000 updates per row; their selection uses training ELBO only.
See the [model, priors and inference details](reference_matrix.md).

## What is exact, and what is approximate?

The main matrix correction uses an **independent-frequency Whittle
approximation**. It models all XYZ cross spectra, but does not retain
cross-frequency covariance or real-data pseudo-covariance. The nominal
window reference is exact for the declared simulator's same-frequency
covariance; propagating an arbitrary residual correction through that
reference is still an approximation.

The retained **exact joint Gaussian benchmark** includes channel, frequency,
quadrature and epoch correlations for 48 selected Fourier bins / 288 real
features. It fits two smooth secondary-component corrections with MAP and
its own priors. Its smaller information budget and narrower model make it
a covariance control, not an equal-information contest with full matrix VI.
[Benchmark derivation and controls](likelihood.md).

**GapLike is useful for XYZ.** Its current independent-channel likelihood
is not a drop-in correlated XYZ likelihood. We use its observation tools
and explicitly extend the covariance to XYZ. The
[GapLike and XYZ explanation](gaplike_xyz.md) distinguishes these pieces.

## Saved standard result, seed 7

![Reference-corrected multivariate spline recovery](assets/seven_day_xyz.png)

```{include} assets/seven_day_errors.md
```

The full correction reduces auto-spectrum error relative to the uncorrected
reference in this realization. Continuous and gapped errors are close.
Coherence error is larger than for the tightly constrained two-component
benchmark: additional matrix flexibility does not improve every metric.
The benchmark uses fewer data and different priors, so its RMSE does not
establish which likelihood is superior.

Scores use the same fixed 17-epoch reconstruction grid and native bins
within 0.6 mHz of all three nulls. Detailed errors include signed residuals,
complex cross spectra, coherence and adjacent-frequency residual differences.
Known geometry strongly constrains the dip shapes; these plots do not show
unconstrained splines discovering response zeros.

All standard-run rows and restarts satisfy the stated ELBO plateau check.
The maximum restart disagreement in reconstructed auto spectra is about
0.0016 dex. [ELBO traces](assets/seven_day_elbo.png) and the
[detailed diagnostics](assets/seven_day_errors.json) accompany the result.
These are numerical checks of exploratory VI, not proof of posterior
accuracy or NUTS-equivalent uncertainty.

Four independent realizations, seeds 1007–1010, are generated only after
partition selection and fitting. Their mean marginal quadratic per dimension
is approximately 1.000 for both observations. The benchmark separately
scores its full joint Gaussian on the same independent seeds. No raw
ELBO/likelihood comparison is made across these different targets, and no
NUTS recommendation follows from this single simulation.

## Outputs

- `comparison.png`, `errors.md`, `errors.json`: main comparison and diagnostics.
- `matrix_recovery.npz`: physical spectral matrices, corrections, native
  frequencies, actual adaptive edges/counts and conditioned references.
- `continuous_correction.nc`, `gapped_correction.nc`: library `PSDResult`
  files with restored theta draws, in reference coordinates. These are
  HDF5-backed; use `PSDResult.from_netcdf`. Frequency coordinates in these
  files are log-frequency spline coordinates, not Hz.
- `*_basis.npz`, `*_elbo.npz`, `*_vi.json`, `elbo.png`: actual library bases,
  penalties, three-restart loss traces and stability checks.
- `benchmark_comparison.png`, `benchmark_errors.*`, `benchmark_run.json`,
  `recovery.npz`: the exact compressed control and its real simulated data.
- `run.json`: versions, source hashes, approximations and validation settings.

The [saved receipt](assets/seven_day_run.json) records the source version.
Use `--benchmark-only` to run the exact control alone. `--quick` keeps all
seven days and all native bins in the main fit, with shorter VI, fewer stored
draws, fewer source bands and a smaller exact benchmark.
