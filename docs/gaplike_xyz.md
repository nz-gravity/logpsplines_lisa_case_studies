# Can GapLike fit XYZ?

**Yes, GapLike can be part of an XYZ analysis.** The distinction is between
using its gap tools and assuming its current multichannel likelihood already
models correlated XYZ.

## What we use

The pinned [GapLike](https://github.com/gaplike/gaplike) implementation provides
our gap masks, edge gates and segment windows. These operators apply to
all three time streams. Its exact stationary scalar likelihood also supplies
an independent test control.

We extend the covariance calculation to the full complex XYZ matrix and,
for the compressed benchmark, to the joint real/imaginary Fourier vector.
GapLike defines the observation; the case-study adapter supplies the extra
channel and orbit structure.

## What stacking three channels means

In the pinned package's multichannel CG interface, channels are independent
with identical covariance. See its
[documented assumption](https://github.com/gaplike/gaplike/blob/99c61931f3f47997d589f255191052ccdbb79f97/src/gaplike/cg.py#L165-L166).
Stacking XYZ into that interface would therefore omit their cross spectra.
XYZ are correlated even before gaps. Unequal, evolving arms generally also
leave residual correlations after a simple XYZ-to-AET rotation.

A correlated XYZ likelihood needs a block covariance such as

$$C_{\rm obs}=G\,C_{XYZ}\,G^\dagger,$$

where $G$ contains the actual observation masks/transforms and $C_{XYZ}$
contains all channel cross blocks. A time gap also mixes Fourier frequencies.
Frequency-dependent whitening and a time-domain mask do not generally commute,
so whitening three spectra and then invoking an independent-channel exact
solver does not automatically solve the original problem.

This is a missing model/operator extension, not a fundamental obstacle to
GapLike or XYZ. A block covariance operator could also use matrix-free CG
and a matching determinant method. That general full-time-series extension
is not implemented in this demonstration.

## The two fitting paths here

| Path | Channel dependence | Frequency dependence | Role |
|---|---|---|---|
| Reference + full matrix spline | Full 3×3 complex residual covariance | Independent-frequency Whittle approximation; nominal window reference included | Fast library demonstration with adaptive bins and marginalised theta |
| Exact compressed Gaussian | Full XYZ and real/imaginary dependence | Full coupling among retained features, including gaps and orbit epochs | Benchmark for the declared simulator and fixed feature set |

The first path analytically marginalises theta under its regression-based
Whittle model. Simply replacing its likelihood with the exact joint gapped
covariance does not preserve that Gaussian integration. The second path
fits a narrower two-component correction directly and supplies a control
for the correlations omitted by the fast path.

Both paths operate on the same seven-day simulation and actual GapLike masks.
Neither treats zero-filled gap samples as extra independent data. The exact
benchmark is exact for its retained Fourier features, not for the entire
three-channel time record.
