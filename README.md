# svi-ssvi-volatility-surface

Pipeline for constructing equity index volatility surfaces from market option prices.

Fetches live SPY options data from public Yahoo Finance API, extracts forward prices via put-call parity, computes implied volatilities using Black's model, and calibrates the volatility surface. The implementation supports two parameterizations:

- SVI (per-expiry): 5 parameters per slice, near-perfect fits, no cross-expiry consistency
- SSVI (joint): 3 global parameters, arbitrage-free by construction, smooth interpolation across tenors

The whole framework -- from option payoffs through Black-Scholes to SVI/SSVI calibration -- are in the accompanying note..

![SSVI Fitted Surface](output/plot_3d_smooth_20251230_224114.png)

## Files

- `data_pipeline.py`: Fetches option chains, extracts forwards via put-call parity, computes IVs
- `svi_calibration.py`: Per-expiry SVI fits with butterfly arbitrage check
- `ssvi_calibration.py`: Joint surface fit with power-law φ(θ)
- `visualization.py`: Generates diagnostic plots from calibration outputs

- `volatility_surface_note.pdf`: Mathematical background and explanation of the implementation

## Key Reference

- Gatheral & Jacquier (2014). *Arbitrage-free SVI volatility surfaces*. Quantitative Finance.
