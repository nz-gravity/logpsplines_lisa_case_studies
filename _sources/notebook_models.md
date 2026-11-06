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
With a prescribed, smooth positive-definite residual matrix $R_{\rm inj}(f)$,

$$x(t)=\sum_e a_e(t)\operatorname{irfft}\{F_e(f)R_{\rm inj}(f)^{1/2}\xi(f)\}(t).$$

The generator's factor and evolutionary spectral matrix are

$$F(t,f)=\sum_e a_e(t)F_e(f),\qquad
S_{\rm gen}(t,f)=F(t,f)R_{\rm inj}(f)F(t,f)^\dagger.$$

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

The XYZ notebooks use a deliberately approximate sine reference. At each
orbit epoch, the three known four-link loop times $\tau_j$ give

$$g_j(f)=2i e^{-i\pi f\tau_j}\sin(\pi f\tau_j)
       =1-e^{-2\pi i f\tau_j},$$

$$S_{0,e}(f)=\operatorname{diag}(g_e)B_\star(f)
             \operatorname{diag}(g_e)^\dagger.$$

$B_\star$ is a full first-generation Michelson covariance envelope frozen
at the initial Keplerian geometry. Only the loop factors follow the known
analysis geometry. The generator instead evolves the full response, so the
reference is not an injected truth spectrum. No dip parameters or component
noise levels are fitted. The sine factors place the dominant nulls, while
the frozen envelope retains additional known frequency structure.

Let $L_{0,e}=\operatorname{chol}(S_{0,e})$ and interpolate these factors with
the same epoch weights. The fitted model is
$S_{\rm fit}(t,f)=L_0(t,f)R_{\rm fit}(f)L_0(t,f)^\dagger$.
The ordinary multivariate library spline describes the full correction
$R_{\rm fit}$, not independent corrections for X, Y and Z.

Notebook 3 follows the inverse-Cholesky construction in
[Vajpeyi et al., arXiv:2607.04833](https://arxiv.org/abs/2607.04833):

$$R_{\rm fit}^{-1}=T^\dagger D^{-1}T,\qquad
D=\operatorname{diag}(e^{h_1},e^{h_2},e^{h_3}),\qquad
T_{jl}=-\theta_{jl}\ (j>l),\quad T_{jj}=1.$$

Three log-diagonal fields and six real/imaginary regression fields describe
all auto and cross spectra. The pinned library integrates Gaussian theta
weights analytically and restores their joint conditional draws before
reconstruction. Both XYZ notebooks use two NUTS chains for the remaining
log-diagonal weights and smoothing scales. Marginalisation is exact for the
chosen Whittle target. It does not change that observation approximation.

The reported multivariate point is a mean of whole reconstructed matrices,
which preserves positive definiteness. Elementwise matrix medians need not.
Coherence shown in a notebook is computed from its displayed point matrix.

Notebook 3 calls `local_wishart_grid(TimeSeries(...))` for eight Hann-tapered
blocks spanning 0.4–99 mHz. The sine reference is propagated through the
same nominal block window and the complex vectors are whitened before
pooling. `fit(WishartData, StationaryConfig(method="nuts"))` runs the ordinary
multivariate model without a custom likelihood. Counts retain all eight
blocks. The nominal circulant block covariance approximates the original
long-record finite-window covariance.

Both XYZ fits use the library's stationary prior on the **correction**.
This is a different prior on the physical spectrum from a raw-spectrum
spline fit. The reference transformation preserves fixed-matrix likelihood
differences before the declared window/coarse-graining approximations; it
does not make these two prior models equivalent.

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
remain separate. “Adaptive” does not mean the likelihood changes during NUTS.
The residual is approximately constant within an aggregate; this is not a
posterior error guarantee.

The reference adapter uses the library's unchanged stationary prior,
max-normalized derivative penalties with a $10^{-6}$ ridge and
$\sigma\sim\mathrm{HalfNormal}(1.28)$. The stationary eight-block fit uses
the library's observation-based initialisation. For the seven-day fit,
protected singleton matrices have zero empirical conditional variances and
give unusable starting values. A small process-local compatibility context
starts the log-diagonal correction at zero and smoothing scales at 0.3;
theta remains integrated, not fixed. The pinned `StationaryConfig` does not
expose an initializer argument, so this context replaces only
`empirical_init_values` during `fit`, then restores it. A numerical test
verifies the unchanged log target; no library files or environments are
modified. Both fits use the library's information-scaled coordinates.
The reconstruction adapter
restores endpoint multiplicity when evaluating the library's stationary
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

The fourth notebook evaluates the approximate sine reference, a reconstructed
matrix and an injected-matrix oracle on independent realizations. Each
candidate's covariance propagates its own spectral factors through the
complete observation operator; the oracle uses the original full-response
generator. The oracle is used only
for this final control. The benchmark does not optimize a component model,
fit a joint posterior, or analytically marginalise theta under this different
likelihood. Outside the fitted band, the recovered residual is extended
constantly at the nearest boundary; the score includes that assumption.

The NUTS fit remains an independent-frequency Whittle approximation. Its
window ENBW correction approximately tempers information; it does not
restore the omitted covariance. Each inference notebook displays chain traces,
divergences, rank-normalized split R-hat, bulk/tail ESS and tree-depth hits.
All retained chains enter reconstruction. The displayed posterior intervals
condition on the reference and Whittle model; they are not a demonstrated
coverage result for the full time-domain data-generating process.
