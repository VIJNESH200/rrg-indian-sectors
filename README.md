# Relative Rotation Graph (RRG) - Indian Sectors

An interactive Python tool for visualizing the rotational momentum of Indian sector indices against the Nifty 50 benchmark. 

This tool recreates the JdK Relative Rotation Graph logic using weekly data to smooth out market noise, highlighting medium-term trends in equity markets. 

## Features
* **Weekly Data Integration:** Automatically downloads and aggregates weekly interval data via `yfinance` to filter out daily market volatility.
* **JdK RS-Ratio & RS-Momentum:** Calculates normalized rolling z-scores (baseline 100) to plot indices across Leading, Weakening, Lagging, and Improving quadrants.
* **Interactive UI:** Built with `matplotlib` widgets, featuring interactive checkboxes to isolate specific sectors and a dynamic slider to control the historical lookback tail.
* **Visual Clarity:** Implements alpha-fading tails to intuitively display momentum directionality and reduce data clustering.

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
