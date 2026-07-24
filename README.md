## Methodology

This tool implements a double-bootstrap procedure:

1. **Detrending:** The series is decomposed into a systematic component
   (linear trend or mean) and residuals.
2. **Block bootstrap:** Residuals are resampled in contiguous blocks to
   preserve autocorrelation.
3. **Reconstruction:** For each bootstrap replicate, the trend is
   re-estimated and extrapolated forward.
4. **Prediction intervals:** Percentile intervals are computed for
   each forecast horizon.

For the complete methodology, see the accompanying paper:
[A Bootstrap-Based Forecasting Framework for Small Business Samples](paper_link)
