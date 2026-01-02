# Volatility Surface Construction: SVI & SSVI Calibration

A pipeline for constructing and parameterizing equity index volatility surfaces from market option prices. Fetches live SPY options data, extracts implied volatilities, and calibrates both per-expiry SVI and joint SSVI parameterizations.

![SSVI Fitted Surface](outputs/plot_ssvi_surface.png)

## Quick Start

```bash
pip install numpy pandas scipy matplotlib yfinance

python data_pipeline.py      # Fetch options, compute IVs
python svi_calibration.py    # Fit per-expiry SVI
python ssvi_calibration.py   # Fit joint SSVI
python visualization.py      # Generate plots
```

## Documentation

See [volatility_surface_note.pdf](volatility_surface_note.pdf) for the full treatment -- from option pricing fundamentals through SVI/SSVI calibration.

## Key Reference

- Gatheral & Jacquier (2014). *Arbitrage-free SVI volatility surfaces*. Quantitative Finance.
