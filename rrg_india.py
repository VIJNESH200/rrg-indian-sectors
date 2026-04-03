#!/usr/bin/env python3
"""
Relative Rotation Graph (RRG) for Indian sector indices vs Nifty 50.

Dependencies (free/open-source):
    pip install yfinance pandas numpy matplotlib

Usage:
    python rrg_india.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class Config:
    benchmark: str = "^NSEI"
    sectors: tuple[str, ...] = (
        "^NSEBANK",
        "^CNXIT",
        "^CNXAUTO",
        "^CNXFMCG",
        "^CNXPHARMA",
        "^CNXMETAL",
    )
    lookback_period: str = "3y"
    interval: str = "1wk"
    rolling_window: int = 14
    tail_periods: int = 12


def download_prices(cfg: Config) -> pd.DataFrame:
    """Download adjusted close prices for benchmark + sector indices."""
    tickers = [cfg.benchmark, *cfg.sectors]
    data = yf.download(
        tickers=tickers,
        period=cfg.lookback_period,
        interval=cfg.interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if data.empty:
        raise RuntimeError("No data returned from Yahoo Finance.")

    if "Close" in data.columns:
        prices = data["Close"].copy()
    else:
        # yfinance returns a regular DataFrame for single ticker,
        # but we request many tickers; this is just a defensive fallback.
        prices = data.copy()

    prices = prices.dropna(how="all").ffill().dropna(how="any")

    missing = set(tickers) - set(prices.columns)
    if missing:
        raise RuntimeError(f"Missing ticker columns in downloaded data: {sorted(missing)}")

    return prices[tickers]


def zscore_to_100(series: pd.Series, window: int) -> pd.Series:
    """Normalize a series to baseline 100 using rolling z-score."""
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    std = std.replace(0, np.nan)
    z = (series - mean) / std
    return 100 + z


def compute_rrg_metrics(prices: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute JdK-style RS-Ratio (x) and RS-Momentum (y).

    RS is computed as Sector / Benchmark.
    RS-Ratio and RS-Momentum are normalized around 100 via rolling z-score.
    """
    benchmark_series = prices[cfg.benchmark]

    rs_ratio_df: dict[str, pd.Series] = {}
    rs_mom_df: dict[str, pd.Series] = {}

    for sector in cfg.sectors:
        rs = prices[sector] / benchmark_series
        rs_ratio = zscore_to_100(rs, cfg.rolling_window)

        # Momentum of RS-Ratio via 1-period ROC, then normalize to 100 baseline.
        rs_ratio_roc = rs_ratio.pct_change()
        rs_momentum = zscore_to_100(rs_ratio_roc, cfg.rolling_window)

        rs_ratio_df[sector] = rs_ratio
        rs_mom_df[sector] = rs_momentum

    rs_ratio_all = pd.DataFrame(rs_ratio_df)
    rs_mom_all = pd.DataFrame(rs_mom_df)

    return rs_ratio_all, rs_mom_all


def plot_rrg(rs_ratio: pd.DataFrame, rs_mom: pd.DataFrame, cfg: Config) -> None:
    """Plot Relative Rotation Graph with quadrants and rotational tails."""
    import matplotlib.patches as mpatches
    from matplotlib.widgets import CheckButtons, Slider

    fig, ax = plt.subplots(figsize=(12, 9))

    # Determine dynamic ranges with margin.
    combined_x = rs_ratio.stack().dropna()
    combined_y = rs_mom.stack().dropna()
    if combined_x.empty or combined_y.empty:
        raise RuntimeError("Not enough data to plot RRG.")

    x_min, x_max = combined_x.min(), combined_x.max()
    y_min, y_max = combined_y.min(), combined_y.max()

    pad_x = max(1.0, (x_max - x_min) * 0.15)
    pad_y = max(1.0, (y_max - y_min) * 0.15)

    x0, x1 = min(x_min - pad_x, 95), max(x_max + pad_x, 105)
    y0, y1 = min(y_min - pad_y, 95), max(y_max + pad_y, 105)

    # Quadrant shading centered at (100,100)
    ax.fill_between([100, x1], 100, y1, color="green", alpha=0.12, zorder=0)   # Leading
    ax.fill_between([100, x1], y0, 100, color="yellow", alpha=0.15, zorder=0)  # Weakening
    ax.fill_between([x0, 100], y0, 100, color="red", alpha=0.12, zorder=0)     # Lagging
    ax.fill_between([x0, 100], 100, y1, color="blue", alpha=0.10, zorder=0)    # Improving

    # Mid lines.
    ax.axvline(100, color="gray", linewidth=1.2, linestyle="--")
    ax.axhline(100, color="gray", linewidth=1.2, linestyle="--")

    cmap = plt.get_cmap("tab10")

    labels_list = [sector.replace("^", "") for sector in cfg.sectors]
    color_by_label = {lbl: cmap(i % 10) for i, lbl in enumerate(labels_list)}
    lines_by_label = {lbl: [] for lbl in labels_list}

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("JdK RS-Ratio (normalized, baseline 100)")
    ax.set_ylabel("JdK RS-Momentum (normalized, baseline 100)")
    ax.set_title("Relative Rotation Graph (RRG): Indian Sectors vs Nifty 50")

    # Quadrant Labels Legend instead of overlapping text
    quadrant_patches = [
        mpatches.Patch(color="green", alpha=0.3, label="Leading"),
        mpatches.Patch(color="yellow", alpha=0.3, label="Weakening"),
        mpatches.Patch(color="red", alpha=0.3, label="Lagging"),
        mpatches.Patch(color="blue", alpha=0.3, label="Improving"),
    ]
    ax.legend(handles=quadrant_patches, title="Quadrants", loc="upper left", frameon=True)

    ax.grid(alpha=0.25)
    
    # Adjust layout to make room for checkboxes and slider
    plt.subplots_adjust(left=0.08, right=0.82, top=0.92, bottom=0.15)
    
    # Add CheckButtons
    rax = fig.add_axes([0.84, 0.4, 0.14, 0.4]) # x, y, width, height
    rax.set_title("Sectors")
    visibility = [True] * len(labels_list)
    check = CheckButtons(rax, labels_list, visibility)
    ax._check_buttons = check  # Retain reference
    
    # Match colors with indices
    for i, lbl in enumerate(labels_list):
        check.labels[i].set_color(color_by_label[lbl])
        check.labels[i].set_fontweight("bold")

    def toggle_lines(label):
        # We need the index to determine the visibility status explicitly
        idx = labels_list.index(label)
        is_visible = check.get_status()[idx]
        for artist in lines_by_label[label]:
            artist.set_visible(is_visible)
        fig.canvas.draw_idle()

    check.on_clicked(toggle_lines)

    def draw_sectors(tail_periods_val):
        # Clear old lines
        for label, artist_list in lines_by_label.items():
            for artist in artist_list:
                try:
                    artist.remove()
                except ValueError:
                    pass
            lines_by_label[label] = []

        for i, sector in enumerate(cfg.sectors):
            df = pd.DataFrame({"x": rs_ratio[sector], "y": rs_mom[sector]}).dropna()
            tail = df.tail(tail_periods_val)
            if tail.empty:
                continue

            label = sector.replace("^", "")
            color = color_by_label[label]
            idx = labels_list.index(label)
            is_vis = check.get_status()[idx]

            artists = []
            points = tail[["x", "y"]].to_numpy()
            n_points = len(points)
            
            if n_points == 0:
                continue

            # Plot segments and scatter points with fading alphas
            for j in range(n_points):
                alpha = 0.1 + 0.9 * (j / max(1, n_points - 1)) if n_points > 1 else 1.0
                
                # Scatter points
                if j == n_points - 1:
                    sc = ax.scatter(points[j, 0], points[j, 1], color=color, s=95, edgecolor="black", linewidth=0.6, alpha=1.0, zorder=5)
                    txt = ax.text(points[j, 0] + 0.15, points[j, 1] + 0.15, label, fontsize=9, color=color, weight="bold")
                    artists.extend([sc, txt])
                else:
                    sc = ax.scatter(points[j, 0], points[j, 1], color=color, s=28, alpha=alpha, zorder=3)
                    artists.append(sc)
                
                # Connecting lines fading
                if j < n_points - 1:
                    line_alpha = 0.1 + 0.9 * ((j + 0.5) / max(1, n_points - 1))
                    line, = ax.plot(
                        [points[j, 0], points[j+1, 0]], 
                        [points[j, 1], points[j+1, 1]], 
                        color=color, linewidth=2, alpha=line_alpha, solid_capstyle='round'
                    )
                    artists.append(line)

            for art in artists:
                art.set_visible(is_vis)

            lines_by_label[label] = artists

        fig.canvas.draw_idle()

    # Draw initially
    draw_sectors(cfg.tail_periods)

    # Add Slider axis at the bottom
    slider_ax = fig.add_axes([0.25, 0.05, 0.5, 0.03])
    max_history = len(rs_ratio.dropna())
    slider_valmax = min(90, max_history) if max_history > 0 else 90
    slider = Slider(
        ax=slider_ax, 
        label='Data Points', 
        valmin=1, 
        valmax=slider_valmax, 
        valinit=cfg.tail_periods, 
        valstep=1
    )
    ax._slider = slider

    def update_slider(val):
        draw_sectors(int(val))
        
    slider.on_changed(update_slider)

    plt.show()


def main() -> int:
    cfg = Config()
    try:
        prices = download_prices(cfg)
        rs_ratio, rs_mom = compute_rrg_metrics(prices, cfg)
        plot_rrg(rs_ratio, rs_mom, cfg)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
