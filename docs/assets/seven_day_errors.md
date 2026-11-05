| Fit | Auto log10 RMSE | Signed log10 residual | Complex cross RMSE* | Coherence RMSE | Validation q / dimension |
|---|---:|---:|---:|---:|---:|
| continuous | 0.0633 | +0.0141 | 0.0364 | 0.0002 | 0.951 |
| gapped | 0.0738 | +0.0157 | 0.0330 | 0.0001 | 1.015 |
| gapped independent bins | 0.0699 | +0.0109 | 0.0298 | 0.0001 | 1.020 |

*Complex cross errors normalized by the geometric mean of the corresponding true auto spectra. Four independent validation realizations; q/d should be near 1 in expectation, not forced by an in-sample amplitude fit.

All fits use the same 7 cubic basis functions per component and the same proper Gaussian prior; no knot tuning or posterior uncertainty claim.
