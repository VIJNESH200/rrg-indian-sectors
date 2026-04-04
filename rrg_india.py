#!/usr/bin/env python3
"""
Relative Rotation Graph (RRG) for Indian sector indices vs Nifty 50.

Dependencies (free/open-source):
    pip install yfinance pandas numpy matplotlib

Usage:
    python rrg_india.py
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.interpolate import make_interp_spline


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


def _cache_key(cfg: Config) -> str:
    """Generate a unique hash for the current configuration."""
    key = f"{cfg.interval}_{cfg.lookback_period}_{cfg.benchmark}_{''.join(cfg.sectors)}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def download_prices(cfg: Config) -> pd.DataFrame:
    """Download adjusted close prices, using a robust local CSV cache."""
    cache_file = Path(f"rrg_cache_{_cache_key(cfg)}.csv")
    tickers = [cfg.benchmark, *cfg.sectors]

    if cache_file.exists():
        file_age = time.time() - cache_file.stat().st_mtime
        if file_age < 3600:
            try:
                cached_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                cached_data.index = pd.to_datetime(cached_data.index)
                
                if all(ticker in cached_data.columns for ticker in tickers):
                    print(f"Cache hit! File is {file_age/60:.1f} minutes old.")
                    return cached_data[tickers]
            except Exception as e:
                print(f"Cache corrupted ({e}). Re-downloading...")
                cache_file.unlink(missing_ok=True)

    print("Downloading fresh data from Yahoo Finance...")
    data = yf.download(
        tickers=tickers, period=cfg.lookback_period, interval=cfg.interval,
        auto_adjust=True, progress=False, threads=True
    )

    if data.empty:
        raise RuntimeError("No data returned from Yahoo Finance.")

    # Bulletproof yfinance column parsing
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    elif "Close" in data.columns:
        prices = data["Close"].copy()
    else:
        prices = data.copy()

    prices = prices.dropna(how="all").ffill().dropna(how="any")
    prices.index = pd.to_datetime(prices.index)

    missing = set(tickers) - set(prices.columns)
    if missing:
        raise RuntimeError(f"Missing ticker columns: {sorted(missing)}")

    prices[tickers].to_csv(cache_file)
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
    from matplotlib.widgets import CheckButtons, Slider, Button

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
    
    # --- Tracking State & Tooltip Setup ---
    visibility_state = {lbl: True for lbl in labels_list}
    hover_data = [] # Stores (x, y, date_str, label, color)

    annot = ax.annotate(
        "", xy=(0,0), xytext=(12, 12), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
        fontsize=9, zorder=10
    )
    annot.set_visible(False)

    def draw_sectors(tail_periods_val):
        hover_data.clear() # Reset tooltip data on redraw
        
        # Safely remove old lines
        for label, artist_list in lines_by_label.items():
            for artist in artist_list:
                try: artist.remove()
                except ValueError: pass
            lines_by_label[label] = []

        for i, sector in enumerate(cfg.sectors):
            df = pd.DataFrame({"x": rs_ratio[sector], "y": rs_mom[sector]}).dropna()
            tail = df.tail(tail_periods_val)
            if tail.empty: continue

            label = labels_list[i]
            color = color_by_label[label]
            is_vis = visibility_state[label] 
            artists = []

            points = tail[["x", "y"]].to_numpy()
            n_points = len(points)

            # 1. Plot the Dots and Setup Hover Data
            for j in range(n_points):
                alpha = 0.1 + 0.9 * (j / max(1, n_points - 1)) if n_points > 1 else 1.0
                
                date_str = tail.index[j].strftime('%b %d, %Y')
                hover_data.append((points[j, 0], points[j, 1], date_str, label, color))

                if j == n_points - 1:
                    # Current week (Large dot with text)
                    sc = ax.scatter(points[j, 0], points[j, 1], color=color, s=95, edgecolor="black", zorder=5)
                    txt = ax.text(points[j, 0] + 0.15, points[j, 1] + 0.15, label, fontsize=9, color=color, weight="bold")
                    artists.extend([sc, txt])
                else:
                    # Historical weeks (Small fading dots)
                    sc = ax.scatter(points[j, 0], points[j, 1], color=color, s=28, alpha=alpha, zorder=3)
                    artists.append(sc)

            # 2. Plot the Smooth Fading Spline Curves
            if n_points >= 3:
                # Setup the parametric variables
                t = np.arange(n_points)
                t_smooth = np.linspace(0, n_points - 1, n_points * 10) # 10x resolution
                
                # Calculate the tight quadratic curve (k=2)
                spl_x = make_interp_spline(t, points[:, 0], k=2) 
                spl_y = make_interp_spline(t, points[:, 1], k=2)
                x_smooth, y_smooth = spl_x(t_smooth), spl_y(t_smooth)

                # Draw the smooth curve in micro-segments to keep the fading effect
                for seg in range(len(t_smooth) - 1):
                    progress = t_smooth[seg] / max(1, n_points - 1)
                    line_alpha = 0.1 + 0.9 * progress
                    line, = ax.plot([x_smooth[seg], x_smooth[seg+1]], [y_smooth[seg], y_smooth[seg+1]], 
                                    color=color, linewidth=2, alpha=line_alpha, solid_capstyle='round', zorder=2)
                    artists.append(line)
                    
            elif n_points == 2:
                # Fallback to a straight line if exactly 2 points exist
                line, = ax.plot([points[0, 0], points[1, 0]], [points[0, 1], points[1, 1]], 
                                color=color, linewidth=2, alpha=0.55, solid_capstyle='round', zorder=2)
                artists.append(line)

            for art in artists:
                art.set_visible(is_vis)

            lines_by_label[label] = artists
            
        fig.canvas.draw_idle()

    # --- UI Checkboxes ---
    rax = fig.add_axes((0.84, 0.35, 0.14, 0.4))
    rax.set_title("Sectors")
    check = CheckButtons(rax, labels_list, [True]*len(labels_list))
    ax._check_buttons = check  # Retain reference
    
    for i, lbl in enumerate(labels_list):
        check.labels[i].set_color(color_by_label[lbl])
        check.labels[i].set_fontweight("bold")

    def toggle_lines(label):
        visibility_state[label] = not visibility_state[label]
        is_vis = visibility_state[label]
        for artist in lines_by_label[label]:
            artist.set_visible(is_vis)
        
        # Hide tooltip immediately if sector is toggled off while hovering
        if not is_vis and annot.get_visible() and label in annot.get_text():
            annot.set_visible(False)
            
        fig.canvas.draw_idle()

    check.on_clicked(toggle_lines)

    # --- NEW: UI Select/Deselect All Buttons ---
    ax_select = fig.add_axes((0.84, 0.86, 0.14, 0.04))
    ax_deselect = fig.add_axes((0.84, 0.81, 0.14, 0.04))
    
    btn_select = Button(ax_select, 'Select All', hovercolor='0.9')
    btn_deselect = Button(ax_deselect, 'Deselect All', hovercolor='0.9')
    
    # Keep references to prevent matplotlib from garbage-collecting the buttons
    ax._btn_select = btn_select
    ax._btn_deselect = btn_deselect

    def select_all(event):
        for i, lbl in enumerate(labels_list):
            if not visibility_state[lbl]:
                check.set_active(i) # Automatically triggers toggle_lines
                
    def deselect_all(event):
        for i, lbl in enumerate(labels_list):
            if visibility_state[lbl]:
                check.set_active(i) # Automatically triggers toggle_lines

    btn_select.on_clicked(select_all)
    btn_deselect.on_clicked(deselect_all)

    # --- UI Hover Logic ---
    def on_hover(event):
        if event.inaxes != ax:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return

        best_dist = float('inf')
        best_pt = None

        # Find closest point, ignoring hidden sectors
        for hx, hy, date_str, lbl, color in hover_data:
            if not visibility_state[lbl]: continue 

            dist = (hx - event.xdata)**2 + (hy - event.ydata)**2
            if dist < best_dist:
                best_dist = dist
                best_pt = (hx, hy, date_str, lbl, color)

        if best_dist < 0.2: # Hit threshold
            hx, hy, date_str, lbl, color = best_pt
            annot.xy = (hx, hy)
            annot.set_text(f"{lbl}\nDate: {date_str}\nRatio: {hx:.2f}\nMom: {hy:.2f}")
            annot.get_bbox_patch().set_edgecolor(color)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    # --- UI Slider (Capped at 24) ---
    slider_ax = fig.add_axes((0.25, 0.05, 0.5, 0.03))
    max_history = len(rs_ratio.dropna(how="all"))
    slider_valmax = min(24, max_history) if max_history > 0 else 24
    
    slider = Slider(ax=slider_ax, label='Data Points', valmin=1, valmax=slider_valmax, valinit=min(cfg.tail_periods, slider_valmax), valstep=1)
    ax._slider = slider

    def update_slider(val):
        draw_sectors(int(val))
        
    slider.on_changed(update_slider)

    draw_sectors(cfg.tail_periods)
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
