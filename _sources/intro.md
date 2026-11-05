# LISA case studies

Runnable LISA noise examples from the [LogPSplinePSD documentation](https://nz-gravity.github.io/LogPSplinePSD/intro.html).
Start with a simple spectrum, add correlated channels, then recover smooth
secondary noise through an evolving orbit response and real time-domain gaps.

| Case | Data and recovery | Inference |
|---|---|---|
| [Nonstationary A noise](time_varying.md) | Changing scalar noise, time-frequency spline versus stationary spline | Exploratory VI point reconstruction |
| [Correlated XYZ noise](xyz.md) | Equal-arm spectral matrices, TM and OMS noise levels | Two-parameter NUTS |
| [Seven-day evolving XYZ with gaps](gapped_xyz.md) | Real time series, Keplerian TDI-2 response, smooth link spectra, continuous versus gapped data | Reference + matrix spline VI; exact gap benchmark |

![Seven-day XYZ recovery](assets/seven_day_xyz.png)

The seven-day case follows the short-segment, orbit-response idea in
[Burke et al., *Noisy Pipelines I*](https://arxiv.org/html/2610.28726v1),
with a full multivariate spline correction, adaptive frequency aggregation
and marginalised Cholesky theta coefficients. A separate exact joint covariance
for retained Fourier features supplies a gap benchmark.
The response determines the moving dips; splines learn the smooth secondary
spectra. It is a compact teaching simulation, not a reproduction of the
paper's full instrument and conditioning pipeline.

[Install and run the examples](running.md), or
[inspect the code on GitHub](https://github.com/nz-gravity/logpsplines_lisa_case_studies).
The **Back to LogPSplinePSD** button remains in the header on every page.
