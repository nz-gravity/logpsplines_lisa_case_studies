# Parametric reference and multivariate correction

The physical model supplies the fast response structure. The full matrix
spline supplies smooth residual freedom. These are separate from the
observation operator and from the Gaussian integration of theta.

## Physical and observed references

At each epoch, the TDI-2 mixing matrix maps six independent TM and six
independent OMS sources into XYZ. Light travel times come from eight
Keplerian orbit epochs; all epoch filters act on shared link sources.
The known, unconditioned matrix is $S_0(t,f)=L_0L_0^\dagger$.

The physical reconstruction uses

$$S(t,f)=L_0(t,f)R(f)L_0(t,f)^\dagger.$$

For a window $w$, let $A_{w,0}(f)A_{w,0}(f)^\dagger=C_{w,0}(f)$ be the
nominal covariance of its normalized Fourier coefficients. We compute
$C_{w,0}$ with all epoch-pair cross terms and the actual gap/taper using
FFT convolutions. Its blocks agree with the independent joint Gaussian
benchmark. No fitted spectrum or injection truth determines this reference.

Before any pooling, the observed coefficients are whitened as
$z_w(f)=A_{w,0}(f)^{-1}\widetilde x_w(f)$. The fitted observation model is
$z_w(f)\sim\mathrm{CN}(0,R(f))$ independently across frequencies.
Equivalently,

$$C_w(f)\simeq A_{w,0}(f)R(f)A_{w,0}(f)^\dagger.$$

This is an approximation: conditioning and a frequency-dependent matrix
correction need not commute. The fit omits inter-frequency dependence and
pseudo-covariance even though the reference includes their effect on the
nominal same-frequency complex covariance. The library's window ENBW
correction approximately tempers information; it does not make the
likelihood exact or calibrate posterior widths.

The reference carries orbital time dependence; $R(f)$ is frequency-only for
this short-segment case. A time-dependent residual would require additional
time-spline fields and validation. Stationary secondary noise does not in
general imply a time-independent congruence residual; the shared $R(f)$
is an additional short-segment approximation.

## Full Cholesky spline and exact conditional integration

The residual covariance is

$$R=T^{-1}DT^{-\dagger},\qquad
D=\operatorname{diag}(e^{h_1},e^{h_2},e^{h_3}),\qquad
T_{jl}=-\theta_{jl}\ (j>l),\quad T_{jj}=1.$$

Three $h_j$ fields and six real/imaginary theta fields use the same twelve
cubic frequency basis functions. Knots are uniformly spaced in normalized
log frequency. Every reconstructed draw is positive definite; the reported
point is the **mean of reconstructed matrices**, which preserves positive
definiteness. It is not an elementwise matrix median.

The example uses the pinned library's stationary prior unchanged:
max-normalized integrated derivative penalties with a $10^{-6}$ ridge,
and each smoothing scale $\sigma\sim\mathrm{HalfNormal}(1.28)$.
The Gaussian coefficient precision is $P/\sigma^2$. Continuous and gapped
basis/penalty hashes must agree or execution fails.

Conditional on the diagonal fields and scales, the row likelihood is
quadratic in its Gaussian theta coefficients. The library analytically
integrates their joint Gaussian. VI approximates only the remaining
marginal posterior. Each retained guide draw receives a draw from the exact
conditional theta Gaussian for reconstruction. All six theta weight fields
remain in the saved posterior. Exact conditional integration does not make
VI exact, and it applies here to the **Whittle target**, not to the exact
gapped benchmark.

Three independently seeded full-Gaussian guides start at the known reference
$R=I$, with initial smoothing scales 0.3. This only changes starting values;
it leaves priors and likelihood unchanged. Empirical conditional variances
at rank-one singleton products are zero, so they are unsuitable initial
values. Native-grid reconstruction restores the endpoint multiplicities of
the library's stationary breakpoint basis; a separate test matches the
library's own reconstruction exactly on its fitting grid.

## Dynamic coarse-graining

The gapped training coefficients supply a running-mean **matrix** pilot.
The library accepts contiguous bins subject to a relative matrix-change
tolerance 0.35, maximum width 256 and pilot window 301. The pilot selects
edges only. Actual likelihood statistics are sums of unsmoothed complex
cross-products, including phase, with their true per-bin counts.

Native bins within 0.1 mHz of the nominal 30, 60 and 90 mHz nulls are
protected, and endpoint singleton bins preserve the reconstruction domain.
Both observations use the same frozen partition. No missing time samples,
truth, validation realizations or inferred PSDs select its edges. “Dynamic”
here means adaptive widths selected in preprocessing; the target does not
change during VI.

Whitening is performed on the native grid, so the parametric response is
not replaced by a single value across a wide bin. Pooling the residual
model is exact under independent coefficients only if $R$ is constant
inside an aggregate. The saved run's native and pooled negative log
likelihoods differ by about 4.9 (continuous) and 5.1 (gapped) at their reported
fitted curves, or less than $9\times10^{-5}$ per native bin. This is one
point check, not a posterior error bound or a fine-grid inference comparison.

## Fit selection and evaluation

Each observation uses three VI restarts and 6,000 updates per row. Select the
restart by minimum mean trailing negative ELBO, within the same training
target. A plateau flag compares adjacent trailing windows: the mean loss
change divided by represented native-bin count must be below $10^{-3}$.
Loss traces, row checks and restart reconstruction disagreement are saved.
This numerical criterion is not a posterior convergence diagnostic.

Only after all fits are frozen, generate four independent validation records.
Report their marginal complex quadratic scores separately from the exact
benchmark's joint real-Gaussian scores. Truth supplies final reconstruction
errors on a prespecified grid; it never chooses bins, priors, knots or restarts.

## Checks and environment

The isolated case-study environment pins LogPSplinePSD to commit
`132bc478aee7bb4de87259531e4c4a24b7f18d2f`, which includes adaptive counts
and marginalised theta. The published paper environment is unchanged.

Tests check reference-whitening likelihood differences, adaptive pooling of
complex products/counts, exact pooling for a piecewise-constant residual,
restored matrix reconstruction against the library, and a model trace with
theta scales but no sampled theta weights. Additional dense controls check
the complete nonstationary gapped covariance and its per-frequency reference.
