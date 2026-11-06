# GapLike and correlated XYZ

**GapLike can be part of an XYZ analysis.** The distinction is between its
gap tools and the covariance assumed by its current multichannel likelihood.

The [fourth notebook](notebooks/04_short_xyz_gaps.ipynb) uses GapLike's masks,
edge gates and segment windows on all three time streams. An approximate orbit-derived sine
reference supplies the channel and response-dip structure. Its exact point benchmark
also retains the frequency correlations created by the observation window.

## Why not stack XYZ into a scalar solver?

The pinned package's multichannel CG interface assumes independent channels
with identical covariance; see its
[documented assumption](https://github.com/gaplike/gaplike/blob/99c61931f3f47997d589f255191052ccdbb79f97/src/gaplike/cg.py#L164-L167).
XYZ have cross spectra, so that independence assumption generally fails.
Unequal, evolving arms also leave correlations after a simple AET rotation.

A correlated model requires channel blocks and the actual observation operator:

$$C_{\rm obs}=G\,C_{XYZ}\,G^\dagger.$$

A time gap couples Fourier frequencies. Frequency-dependent whitening and
a time-domain mask do not generally commute, so whitening spectra and then
calling an independent-channel exact solver does not automatically produce
the original XYZ likelihood.

## What the walkthrough does

| Calculation | Channel correlations | Frequency correlations | Purpose |
|---|---|---|---|
| Full matrix spline NUTS fit | Included in the residual spectral matrix | Whittle approximation, with a window-aware nominal reference | Recovery with adaptive bins and marginalised theta |
| Exact selected-feature covariance | Included | Included, with real/imaginary and epoch cross terms | Independent point-model benchmark |

The exact covariance calculation propagates the full observation operator
for each candidate reference-plus-correction model. The independent oracle
uses the original full-response generator; the learned model uses its
approximate sine reference. Both use the original observation windows. It does not fit noise-component
levels. The main fitter's analytic theta marginalisation applies to its
Whittle target; substituting the exact gapped target needs a new derivation.

A full-time-series XYZ solver could use a block covariance operator with
matrix-free CG and a matching determinant method. That general extension
is not implemented here. The current notebook provides an exact, small
covariance control and a clearly labelled fast approximation.
