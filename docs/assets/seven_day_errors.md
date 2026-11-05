| Reconstruction | Fitting target | Auto log10 RMSE | Signed log10 residual | Complex cross RMSE* | Coherence RMSE |
|---|---|---:|---:|---:|---:|
| parametric reference | reference only | 0.0808 | -0.0411 | 0.0575 | 0.0004 |
| continuous | adaptive Whittle, collapsed VI | 0.0141 | -0.0035 | 0.0152 | 0.0033 |
| gapped | adaptive Whittle, collapsed VI | 0.0143 | -0.0037 | 0.0156 | 0.0045 |
| exact continuous benchmark | exact compressed Gaussian MAP | 0.0633 | +0.0141 | 0.0364 | 0.0002 |
| exact gapped benchmark | exact compressed Gaussian MAP | 0.0738 | +0.0157 | 0.0330 | 0.0001 |

*Complex cross errors are normalized by the geometric mean of the true auto spectra. Fixed 17-epoch evaluation grid with native null neighbourhoods. The library fits use all retained native frequencies, pooled into a shared adaptive partition; the exact benchmark uses a much smaller feature set and a two-component model. These rows do not isolate the effect of the likelihood or establish posterior calibration.
