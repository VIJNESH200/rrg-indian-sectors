#!/usr/bin/env python3
"""
Relative Rotation Graph (RRG) for Indian sector indices vs Nifty 50.

Dependencies:
    pip install yfinance>=0.2.40 pandas>=2.0.0 numpy>=1.24.0 matplotlib>=3.6.0 scipy>=1.10.0

Usage:
    python rrg_india.py
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.widgets import Button, CheckButtons, Slider
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
        tickers=tickers,
        period=cfg.lookback_period,
        interval=cfg.interval,
        auto_adjust=True,
        progress=False,
        threads=True,
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


def compute_rrg_metrics(
    prices: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
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


class RRGPlotter:
    """Interactive Matplotlib visualizer for Relative Rotation Graphs."""

    def __init__(
        self, rs_ratio: pd.DataFrame, rs_mom: pd.DataFrame, cfg: Config
    ) -> None:
        self.rs_ratio = rs_ratio
        self.rs_mom = rs_mom
        self.cfg = cfg

        combined_x = self.rs_ratio.stack().dropna()
        combined_y = self.rs_mom.stack().dropna()
        if combined_x.empty or combined_y.empty:
            raise RuntimeError("Not enough data to plot RRG.")

        self.labels_list = [sector.replace("^", "") for sector in cfg.sectors]
        self.cmap = plt.get_cmap("tab10")
        self.color_by_label = {
            lbl: self.cmap(i % 10) for i, lbl in enumerate(self.labels_list)
        }
        self.lines_by_label: dict[str, list[Any]] = {
            lbl: [] for lbl in self.labels_list
        }
        self.visibility_state: dict[str, bool] = {
            lbl: True for lbl in self.labels_list
        }
        self.hover_data: list[tuple[float, float, str, str, Any]] = []

        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self._setup_axes(combined_x, combined_y)
        self._setup_annotation()
        self._setup_widgets()
        self._wire_events()

    def _setup_axes(self, combined_x: pd.Series, combined_y: pd.Series) -> None:
        x_min, x_max = combined_x.min(), combined_x.max()
        y_min, y_max = combined_y.min(), combined_y.max()

        pad_x = max(1.0, (x_max - x_min) * 0.15)
        pad_y = max(1.0, (y_max - y_min) * 0.15)

        x0, x1 = min(x_min - pad_x, 95), max(x_max + pad_x, 105)
        y0, y1 = min(y_min - pad_y, 95), max(y_max + pad_y, 105)

        # Quadrant shading centered at (100,100)
        self.ax.fill_between(
            [100, x1], 100, y1, color="green", alpha=0.12, zorder=0
        )  # Leading
        self.ax.fill_between(
            [100, x1], y0, 100, color="yellow", alpha=0.15, zorder=0
        )  # Weakening
        self.ax.fill_between(
            [x0, 100], y0, 100, color="red", alpha=0.12, zorder=0
        )  # Lagging
        self.ax.fill_between(
            [x0, 100], 100, y1, color="blue", alpha=0.10, zorder=0
        )  # Improving

        # Mid lines.
        self.ax.axvline(100, color="gray", linewidth=1.2, linestyle="--")
        self.ax.axhline(100, color="gray", linewidth=1.2, linestyle="--")

        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self.ax.set_xlabel("JdK RS-Ratio (normalized, baseline 100)")
        self.ax.set_ylabel("JdK RS-Momentum (normalized, baseline 100)")
        self.ax.set_title(
            "Relative Rotation Graph (RRG): Indian Sectors vs Nifty 50"
        )

        quadrant_patches = [
            mpatches.Patch(color="green", alpha=0.3, label="Leading"),
            mpatches.Patch(color="yellow", alpha=0.3, label="Weakening"),
            mpatches.Patch(color="red", alpha=0.3, label="Lagging"),
            mpatches.Patch(color="blue", alpha=0.3, label="Improving"),
        ]
        self.ax.legend(
            handles=quadrant_patches,
            title="Quadrants",
            loc="upper left",
            frameon=True,
        )
        self.ax.grid(alpha=0.25)
        plt.subplots_adjust(left=0.08, right=0.82, top=0.92, bottom=0.15)

    def _setup_annotation(self) -> None:
        self.annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9
            ),
            fontsize=9,
            zorder=10,
        )
        self.annot.set_visible(False)

    def _setup_widgets(self) -> None:
        # Sector checkboxes
        rax = self.fig.add_axes((0.84, 0.35, 0.14, 0.4))
        rax.set_title("Sectors")
        self.check = CheckButtons(
            rax, self.labels_list, [True] * len(self.labels_list)
        )
        self.ax._check_buttons = self.check  # Prevent GC

        for i, lbl in enumerate(self.labels_list):
            self.check.labels[i].set_color(self.color_by_label[lbl])
            self.check.labels[i].set_fontweight("bold")

        # Select / Deselect All Buttons
        ax_select = self.fig.add_axes((0.84, 0.86, 0.14, 0.04))
        ax_deselect = self.fig.add_axes((0.84, 0.81, 0.14, 0.04))

        self.btn_select = Button(ax_select, "Select All", hovercolor="0.9")
        self.btn_deselect = Button(ax_deselect, "Deselect All", hovercolor="0.9")
        self.ax._btn_select = self.btn_select
        self.ax._btn_deselect = self.btn_deselect

        # Lookback slider
        slider_ax = self.fig.add_axes((0.25, 0.05, 0.5, 0.03))
        max_history = len(self.rs_ratio.dropna(how="all"))
        slider_valmax = min(24, max_history) if max_history > 0 else 24
        self.slider = Slider(
            ax=slider_ax,
            label="Data Points",
            valmin=1,
            valmax=slider_valmax,
            valinit=min(self.cfg.tail_periods, slider_valmax),
            valstep=1,
        )
        self.ax._slider = self.slider

    def _wire_events(self) -> None:
        self.check.on_clicked(self.toggle_lines)
        self.btn_select.on_clicked(self.select_all)
        self.btn_deselect.on_clicked(self.deselect_all)
        self.slider.on_changed(self.update_slider)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)

    def draw_sectors(self, tail_periods_val: int) -> None:
        self.hover_data.clear()

        # Safely remove old lines
        for label, artist_list in self.lines_by_label.items():
            for artist in artist_list:
                try:
                    artist.remove()
                except ValueError:
                    pass
            self.lines_by_label[label] = []

        for i, sector in enumerate(self.cfg.sectors):
            df = pd.DataFrame(
                {"x": self.rs_ratio[sector], "y": self.rs_mom[sector]}
            ).dropna()
            tail = df.tail(tail_periods_val)
            if tail.empty:
                continue

            label = self.labels_list[i]
            color = self.color_by_label[label]
            is_vis = self.visibility_state[label]
            artists: list[Any] = []

            points = tail[["x", "y"]].to_numpy()
            n_points = len(points)

            for j in range(n_points):
                alpha = (
                    0.1 + 0.9 * (j / max(1, n_points - 1))
                    if n_points > 1
                    else 1.0
                )
                date_str = tail.index[j].strftime("%b %d, %Y")
                self.hover_data.append(
                    (points[j, 0], points[j, 1], date_str, label, color)
                )

                if j == n_points - 1:
                    sc = self.ax.scatter(
                        points[j, 0],
                        points[j, 1],
                        color=color,
                        s=95,
                        edgecolor="black",
                        zorder=5,
                    )
                    txt = self.ax.text(
                        points[j, 0] + 0.15,
                        points[j, 1] + 0.15,
                        label,
                        fontsize=9,
                        color=color,
                        weight="bold",
                    )
                    artists.extend([sc, txt])
                else:
                    sc = self.ax.scatter(
                        points[j, 0],
                        points[j, 1],
                        color=color,
                        s=28,
                        alpha=alpha,
                        zorder=3,
                    )
                    artists.append(sc)

            if n_points >= 3:
                t = np.arange(n_points)
                t_smooth = np.linspace(0, n_points - 1, n_points * 10)
                spl_x = make_interp_spline(t, points[:, 0], k=2)
                spl_y = make_interp_spline(t, points[:, 1], k=2)
                x_smooth, y_smooth = spl_x(t_smooth), spl_y(t_smooth)

                for seg in range(len(t_smooth) - 1):
                    progress = t_smooth[seg] / max(1, n_points - 1)
                    line_alpha = 0.1 + 0.9 * progress
                    (line,) = self.ax.plot(
                        [x_smooth[seg], x_smooth[seg + 1]],
                        [y_smooth[seg], y_smooth[seg + 1]],
                        color=color,
                        linewidth=2,
                        alpha=line_alpha,
                        solid_capstyle="round",
                        zorder=2,
                    )
                    artists.append(line)
            elif n_points == 2:
                (line,) = self.ax.plot(
                    [points[0, 0], points[1, 0]],
                    [points[0, 1], points[1, 1]],
                    color=color,
                    linewidth=2,
                    alpha=0.55,
                    solid_capstyle="round",
                    zorder=2,
                )
                artists.append(line)

            for art in artists:
                art.set_visible(is_vis)

            self.lines_by_label[label] = artists

        self.fig.canvas.draw_idle()

    def toggle_lines(self, label: str) -> None:
        self.visibility_state[label] = not self.visibility_state[label]
        is_vis = self.visibility_state[label]
        for artist in self.lines_by_label[label]:
            artist.set_visible(is_vis)

        if not is_vis and self.annot.get_visible() and label in self.annot.get_text():
            self.annot.set_visible(False)

        self.fig.canvas.draw_idle()

    def select_all(self, event: Any = None) -> None:
        for i, lbl in enumerate(self.labels_list):
            if not self.visibility_state[lbl]:
                self.check.set_active(i)

    def deselect_all(self, event: Any = None) -> None:
        for i, lbl in enumerate(self.labels_list):
            if self.visibility_state[lbl]:
                self.check.set_active(i)

    def update_slider(self, val: float) -> None:
        self.draw_sectors(int(val))

    def on_hover(self, event: Any) -> None:
        if event.inaxes != self.ax:
            if self.annot.get_visible():
                self.annot.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        best_dist = float("inf")
        best_pt = None

        for hx, hy, date_str, lbl, color in self.hover_data:
            if not self.visibility_state[lbl]:
                continue
            dist = (hx - event.xdata) ** 2 + (hy - event.ydata) ** 2
            if dist < best_dist:
                best_dist = dist
                best_pt = (hx, hy, date_str, lbl, color)

        if best_dist < 0.2 and best_pt is not None:
            hx, hy, date_str, lbl, color = best_pt
            self.annot.xy = (hx, hy)
            self.annot.set_text(
                f"{lbl}\nDate: {date_str}\nRatio: {hx:.2f}\nMom: {hy:.2f}"
            )
            self.annot.get_bbox_patch().set_edgecolor(color)
            self.annot.set_visible(True)
            self.fig.canvas.draw_idle()
        else:
            if self.annot.get_visible():
                self.annot.set_visible(False)
                self.fig.canvas.draw_idle()

    def show(self) -> None:
        self.draw_sectors(self.cfg.tail_periods)
        plt.show()


def plot_rrg(rs_ratio: pd.DataFrame, rs_mom: pd.DataFrame, cfg: Config) -> None:
    """Plot Relative Rotation Graph with quadrants and rotational tails."""
    plotter = RRGPlotter(rs_ratio, rs_mom, cfg)
    plotter.show()


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
