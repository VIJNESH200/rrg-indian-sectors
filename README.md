# 📈 Relative Rotation Graph (RRG) - Indian Sectors

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/VIJNESH200/rrg-indian-sectors/actions/workflows/ci.yml/badge.svg)](https://github.com/VIJNESH200/rrg-indian-sectors/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![yfinance](https://img.shields.io/badge/data-yfinance-green.svg)](https://pypi.org/project/yfinance/)
[![Matplotlib](https://img.shields.io/badge/GUI-Matplotlib-orange.svg)](https://matplotlib.org/)

An interactive, high-performance Python application for visualizing and analyzing the relative strength and momentum trajectories of major **Indian Sectoral Indices** against the **Nifty 50 Benchmark (`^NSEI`)** using **Relative Rotation Graphs (RRG®)**.

This tool implements the mathematical foundation of Julius de Kempenaer's (JdK) Relative Rotation Graph logic using weekly data intervals to filter out daily market noise, allowing traders, quantitative analysts, and investors to identify medium-term sector rotation and momentum shifts in the National Stock Exchange (NSE) equity market.

<img width="1919" height="1020" alt="image" src="https://github.com/user-attachments/assets/23c61c73-4e27-400f-a71a-4427c5f413c2" />


---

## 🌟 Key Features

- 📊 **Weekly Data Pipeline**: Automatically fetches adjusted closing prices for Nifty 50 and sector indices using `yfinance`, aggregated weekly to reduce intra-week volatility.
- 🧮 **Mathematical Precision**: Computes rolling normalized Z-Scores (baseline 100) for **JdK RS-Ratio** (Relative Strength) and **JdK RS-Momentum** (Rate of Change of Relative Strength).
- 🔄 **4-Quadrant Rotational Dynamics**: Plots sector trajectories across **Leading** (Green), **Weakening** (Yellow), **Lagging** (Red), and **Improving** (Blue) quadrants centered at $(100, 100)$.
- 🎨 **Spline Interpolation & Alpha Fading**: Utilizes quadratic B-splines (`scipy.interpolate.make_interp_spline`) with progressive alpha fading for tail curves, reducing data clutter while showing movement direction.
- 🎛️ **Interactive Controls**:
  - **Dynamic Lookback Slider**: Adjust historical tail length (1 to 24 weeks) on the fly.
  - **Sector Checkboxes**: Toggle individual sector visibility.
  - **Master Controls**: Quick **"Select All"** and **"Deselect All"** buttons.
  - **Hover Tooltips**: Real-time inspection of exact date, sector name, RS-Ratio, and RS-Momentum upon hovering over data points.
- ⚡ **Local CSV Caching**: 1-hour automatic local caching mechanism to prevent Yahoo Finance API rate limits during repeated analyses.
- 🛡️ **State-Managed UI**: Bypasses Matplotlib garbage collection issues via independent state dictionaries to maintain UI synchronization.

---

## 🧭 Quadrant Overview

The graph is divided into four distinct quadrants centered around the baseline benchmark point $(100, 100)$:

| Quadrant | Color | RS-Ratio | RS-Momentum | Market Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Leading** | 🟩 Green | $> 100$ | $> 100$ | Sector is outperforming the benchmark with strong positive momentum. |
| **Weakening** | 🟨 Yellow | $> 100$ | $< 100$ | Sector is still outperforming, but momentum is fading (potential exit warning). |
| **Lagging** | 🟥 Red | $< 100$ | $< 100$ | Sector is underperforming the benchmark with negative momentum. |
| **Improving** | 🟦 Blue | $< 100$ | $> 100$ | Sector is underperforming, but positive momentum is building (potential turnaround candidate). |

---

## 📐 Mathematical Formulation

1. **Relative Strength (RS)**:
   $$RS_{\text{sector}, t} = \frac{\text{Price}_{\text{sector}, t}}{\text{Price}_{\text{benchmark}, t}}$$

2. **JdK RS-Ratio**:
   Normalized rolling Z-Score centered around baseline 100:
   $$\text{RS-Ratio}_t = 100 + \frac{RS_t - \mu_{\text{window}}(RS)}{\sigma_{\text{window}}(RS)}$$

3. **JdK RS-Momentum**:
   Rate of Change (ROC) of RS-Ratio, normalized via rolling Z-Score around baseline 100:
   $$\text{ROC}_t = \frac{\text{RS-Ratio}_t - \text{RS-Ratio}_{t-1}}{\text{RS-Ratio}_{t-1}}$$
   $$\text{RS-Momentum}_t = 100 + \frac{\text{ROC}_t - \mu_{\text{window}}(\text{ROC})}{\sigma_{\text{window}}(\text{ROC})}$$

---

## 🏢 Tracked Indian Sector Indices

By default, the script tracks the main NSE sector indices against **Nifty 50 (`^NSEI`)**:

- **Nifty Bank** (`^NSEBANK`)
- **Nifty IT** (`^CNXIT`)
- **Nifty Auto** (`^CNXAUTO`)
- **Nifty FMCG** (`^CNXFMCG`)
- **Nifty Pharma** (`^CNXPHARMA`)
- **Nifty Metal** (`^CNXMETAL`)

*(Note: Custom tickers and additional sectors can easily be configured in `rrg_india.py`)*

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher installed on your system.

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/VIJNESH200/rrg-indian-sectors.git
   cd rrg-indian-sectors
   ```

2. **(Optional) Create a Virtual Environment**:
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

Run the interactive dashboard using:

```bash
python rrg_india.py
```

### Dashboard Interactivity:
- **Lookback Tail Slider**: Drag the slider at the bottom to adjust the number of historical weeks displayed (from 1 to 24 points).
- **Sector Checkboxes**: Click on sector names on the right panel to show/hide individual sector paths.
- **Select / Deselect All**: Use the top-right control buttons to toggle all sector lines instantly.
- **Data Inspection (Hover)**: Move your mouse cursor over any point along a sector's tail to view exact metric details (Date, RS-Ratio, RS-Momentum).

---

## ⚙️ Configuration

You can easily customize parameters by editing the `Config` dataclass inside `rrg_india.py`:

```python
@dataclass(frozen=True)
class Config:
    benchmark: str = "^NSEI"           # Benchmark index (e.g. Nifty 50)
    sectors: tuple[str, ...] = (      # Sector index tickers
        "^NSEBANK",
        "^CNXIT",
        "^CNXAUTO",
        "^CNXFMCG",
        "^CNXPHARMA",
        "^CNXMETAL",
    )
    lookback_period: str = "3y"       # Yahoo Finance fetch period
    interval: str = "1wk"              # Data frequency ("1wk", "1d")
    rolling_window: int = 14          # Rolling window size for Z-score normalization
    tail_periods: int = 12            # Default tail length (weeks)
```

---

## 📁 Project Structure

```text
rrg-indian-sectors/
├── rrg_india.py          # Main application script & Matplotlib dashboard
├── requirements.txt      # Python package dependencies
├── README.md             # Project documentation
├── LICENSE               # MIT License
└── rrg_cache_*.csv       # (Auto-generated) Local 1-hour price cache
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to check the [issues page](https://github.com/VIJNESH200/rrg-indian-sectors/issues) if you want to submit a pull request or suggest new features (e.g., adding Midcap/Smallcap indices, web dashboard integration, exports).

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## ⚠️ Disclaimer

*Relative Rotation Graphs® and RRG® are registered trademarks of RRG Research. This repository is an independent, open-source, and strictly educational implementation of the mathematical concepts behind relative strength sector rotation. It is not affiliated with, endorsed by, or sponsored by RRG Research or Optuma. This tool is intended for analytical and educational purposes only and does not constitute financial advice.*
