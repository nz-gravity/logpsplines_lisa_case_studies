# LISA PSD case studies

Four notebooks take you from simulated data to recovered spectra.
Start with data generation, then follow the estimation lesson you need.
Each notebook explains the next step, runs the code, and plots the result.

1. **[Generate the data](notebooks/01_generate_data.ipynb)** — inspect the
   full XYZ matrix, draw real time series, add time variation and gaps.
2. **[Recover a time-varying PSD](notebooks/02_time_varying_psd.ipynb)** —
   prepare local channel-A powers and sample a time-frequency spline with NUTS.
3. **[Estimate the XYZ spectral matrix](notebooks/03_xyz_psd.ipynb)** —
   use an approximate sine reference for the dips, run the library’s
   marginalised multivariate NUTS fitter, and inspect auto/cross spectra and coherence.
4. **[Short-duration XYZ with gaps](notebooks/04_short_xyz_gaps.ipynb)** —
   use a seven-day sine reference, marginalised NUTS and adaptive bins;
   compare continuous/gapped recovery and an exact covariance point benchmark.

The estimated quantities are **channel PSDs and spectral matrices**.
The notebooks do not fit instrument-noise component levels.

[Install and run](running.md) to execute the notebooks yourself.
Their saved outputs are already displayed here, so you can also read them
without starting a Python kernel.

The [LogPSplinePSD documentation](https://nz-gravity.github.io/LogPSplinePSD/intro.html)
explains the general API. Use the **Back to LogPSplinePSD** button on any page
to return there.
