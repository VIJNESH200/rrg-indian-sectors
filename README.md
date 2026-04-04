# Relative Rotation Graph (RRG) - Indian Sectors

An interactive Python tool for visualizing the rotational momentum of Indian sector indices against the Nifty 50 benchmark. 

This tool recreates the JdK Relative Rotation Graph logic using weekly data to smooth out market noise, highlighting medium-term trends in equity markets. 

## Features
* **Weekly Data Integration:** Automatically downloads and aggregates weekly interval data via `yfinance` to filter out daily market volatility.
* **JdK RS-Ratio & RS-Momentum:** Calculates normalized rolling z-scores (baseline 100) to plot indices across Leading, Weakening, Lagging, and Improving quadrants.
* **Interactive UI:** Built with `matplotlib` widgets, featuring interactive checkboxes to isolate specific sectors and a dynamic slider to control the historical lookback tail.
* **Visual Clarity:** Implements alpha-fading tails to intuitively display momentum directionality and reduce data clustering.

## Key Features Engineered
* **Robust Data Pipeline:** Custom 1-hour CSV caching mechanism to prevent Yahoo Finance rate-limiting during active market analysis.
* **State-Managed UI:** Independent Python dictionary tracking to bypass Matplotlib garbage-collection bugs during rapid slider redrawing, keeping sector checkboxes perfectly synchronized.
* **Interactive Diagnostics:** Precision hover-tooltips displaying localized dates, RS-Ratio, and RS-Momentum for specific sector events.
* **Streamlined Control Panel:** Custom "Select All" and "Deselect All" Matplotlib button widgets for rapid sector isolation.

## Dependencies
* Python 3.8+
* `yfinance`
* `pandas`
* `numpy`
* `matplotlib`

## Usage
Run the script directly from your terminal:
```bash
python rrg_india.py

### Disclaimer
*Relative Rotation Graphs® and RRG® are registered trademarks of RRG Research. This repository is an unofficial, open-source, and strictly educational implementation of the mathematical concepts behind sector rotation. It is not affiliated with, endorsed by, or sponsored by RRG Research or Optuma. This tool is for informational purposes only and does not constitute financial advice.*
