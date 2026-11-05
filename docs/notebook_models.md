# Simulation and likelihood notes

The four notebooks estimate channel PSDs and the full XYZ spectral matrix.
This page collects details that are useful after reading the walkthroughs.

## What is generated

`MatrixSimulation` uses a full LISA-shaped XYZ reference matrix constructed
from orbit-dependent TDI-2 covariance kernels. It forms lower-triangular
spectral factors at eight Keplerian orbit epochs. A relative eigenvalue
floor of $10^{-12}$ keeps numerical null modes positive; the same rule is
used in generation, reference calculations and exact covariance controls.

All epochs act on **shared** stationary Gaussian Fourier noise. The real
outputs are combined using piecewise-linear weights $a_e(t)$ that sum to one.
With a prescribed, smooth positive-definite residual matrix $R(f)$,

$$x(t)=\sum_e a_e(t)\operatorname{irfft}\{L_{0,e}(f)R(f)^{1/2}\xi(f)\}(t).$$

The reference factor and evolutionary spectral matrix are

$$L_0(t,f)=\sum_e a_e(t)L_{0,e}(f),\qquad
S(t,f)=L_0(t,f)R(f)L_0(t,f)^\dagger.$$

This is a controlled Gaussian **spectral-matrix simulator**, not a link-level
instrument simulation or an implementation of noncommuting TDI delay operators.
It deliberately separates known response structure from unknown matrix shape.
The response controls dip positions; the fitted residual need not find them
from scratch.

The stationary XYZ data freeze the geometry. Channel A is formed from a
frozen-geometry XYZ draw and multiplied by a slow amplitude envelope; its
local spectrum is the stationary channel spectrum times the envelope squared.
The seven-day XYZ data keep the residual stationary while the reference evolves.
The notebooks estimate neither individual noise allocations nor component levels.

## Units and real-series conventions

Time is in seconds in saved data; plots use elapsed days or hours. Frequencies
are in Hz, plotted in mHz. Channel data are fractional frequency, and physical
one-sided PSDs have units 1/Hz. The latent DC coefficient is zero. Nyquist uses
a real draw and the real response, as required by a real FFT.

The time-varying notebook calls the public `local_wishart_grid` function with
Hann windows. For its one-channel complex coefficients, `2*Y` is the power
sum and `2*counts` is the number of real degrees of freedom under the local
Whittle approximation. No extra unknown physical-unit conversion is fitted.
Numerical scaling uses training powers only and is explicitly undone.

Finite windows change leakage and dependence. A local evolutionary spectrum
need not equal an expected finite-window periodogram. Hann tapering reduces
leakage; it does not establish frequency independence or posterior calibration.

## Standard multivariate estimation

Notebook 3 follows the inverse-Cholesky construction in
[Vajpeyi et al., arXiv:2607.04833](https://arxiv.org/abs/2607.04833):

$$S^{-1}=T^\dagger D^{-1}T,\qquad
D=\operatorname{diag}(e^{h_1},e^{h_2},e^{h_3}),\qquad
T_{jl}=-\theta_{jl}\ (j>l),\quad T_{jj}=1.$$

Three log-diagonal fields and six real/imaginary regression fields describe
all auto and cross spectra. The pinned library integrates Gaussian theta
weights analytically and restores their joint conditional draws before
reconstruction. The remaining inference is variational in these walkthroughs.
Exact conditional integration does not make the overall VI approximation exact.

The reported multivariate point is a mean of whole reconstructed matrices,
which preserves positive definiteness. Elementwise matrix medians need not.
Coherence shown in a notebook is computed from its displayed point matrix.

The first XYZ fit uses the ordinary library interface on `TimeSeries`, with
Hann-tapered blocks and adaptive frequency aggregation. It starts below
25 mHz to introduce the vanilla estimator before the nulls are addressed
with an orbit reference in the fourth notebook.

## Short segments, reference correction and adaptive bins

The seven-day approach takes inspiration from
[Burke et al., Noisy Pipelines I](https://arxiv.org/html/2610.28726v1).
It is not a reproduction of their complete instrument/conditioning pipeline.

Notebook 4 fits $R(f)$ in $S(t,f)=L_0(t,f)R(f)L_0(t,f)^\dagger$.
The inferred correction is frequency-only; the reference supplies orbital
time variation. A changing residual would require additional time fields.

The nominal Fourier reference propagates all epoch cross terms and the
actual GapLike gate/taper. Its per-frequency complex matrices are checked
against the full joint covariance. Native coefficients are whitened before
pooling. The observation model still approximates the conditioned matrix as
$C_w(f)\simeq A_{w,0}(f)R(f)A_{w,0}(f)^\dagger$.
Conditioning and a frequency-dependent correction need not commute.

The adaptive pilot uses observed gapped training data only. Both fits share
the same partition, fixed before inference. Unsmoothed complex products and
actual counts are summed; native bins in prescribed null neighbourhoods
remain separate. “Adaptive” does not mean the likelihood changes during VI.
The residual is approximately constant within an aggregate; this is not a
posterior error guarantee.

The reference adapter uses the library's unchanged stationary prior,
max-normalized derivative penalties with a $10^{-6}$ ridge and
$\sigma\sim\mathrm{HalfNormal}(1.28)$. Its guide starts at the known
reference, avoiding zero empirical conditional variances at rank-one cells.
It restores endpoint multiplicity when evaluating the library's stationary
breakpoint basis on the native grid. Tests compare this reconstruction with
the library's own fitting-grid evaluation.

## Exact gapped point benchmark

For a fixed subset of Fourier bins, collect both real and imaginary XYZ
coefficients in a real vector $y$. Propagating the same epoch, matrix and
window operators yields its full covariance $C(R)$ and exact Gaussian target

$$\log L(R)=-\tfrac12\{\log|C(R)|+y^T C(R)^{-1}y\}+\mathrm{constant}.$$

This includes channel, frequency, quadrature and epoch cross terms under the
declared simulator. It is exact for the retained features, not the entire
three-channel record. An independent dense time-domain operator test checks
normalisation and all covariance blocks.

The fourth notebook evaluates the reference, a reconstructed matrix and an
injected-matrix oracle on independent realizations. The oracle is used only
for this final control. The benchmark does not optimize a component model,
fit a joint posterior, or analytically marginalise theta under this different
likelihood. Outside the fitted band, the recovered residual is extended
constantly at the nearest boundary; the score includes that assumption.

The fast fit remains an independent-frequency Whittle approximation. Its
window ENBW correction approximately tempers information; it does not
restore the omitted covariance. Three VI restarts and loss checks help
inspect numerical stability, without asserting calibrated uncertainty.
