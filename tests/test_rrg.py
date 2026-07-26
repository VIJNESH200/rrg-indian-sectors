import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless unit tests

import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from rrg_india import (
    Config,
    RRGPlotter,
    _cache_key,
    compute_rrg_metrics,
    download_prices,
    zscore_to_100,
)


def test_config_defaults():
    cfg = Config()
    assert cfg.benchmark == "^NSEI"
    assert "^NSEBANK" in cfg.sectors
    assert cfg.lookback_period == "3y"
    assert cfg.interval == "1wk"
    assert cfg.rolling_window == 14
    assert cfg.tail_periods == 12


def test_cache_key_reproducibility():
    cfg1 = Config()
    cfg2 = Config()
    assert _cache_key(cfg1) == _cache_key(cfg2)
    assert len(_cache_key(cfg1)) == 8

    cfg_modified = Config(lookback_period="1y")
    assert _cache_key(cfg1) != _cache_key(cfg_modified)


def test_zscore_to_100_basic():
    # Linear increasing series: z-score should normalize around baseline 100
    series = pd.Series(np.linspace(10, 50, 30))
    z100 = zscore_to_100(series, window=10)

    # First 9 values should be NaN due to rolling window min_periods
    assert z100.iloc[:9].isna().all()
    # Values after window should be centered around 100
    assert not z100.iloc[9:].isna().any()
    assert (z100.iloc[9:] > 95).all() and (z100.iloc[9:] < 105).all()


def test_zscore_to_100_zero_std_handling():
    # Constant series has zero variance/std -> should return NaN instead of ZeroDivisionError
    series = pd.Series([50.0] * 20)
    z100 = zscore_to_100(series, window=5)
    assert z100.isna().all()


def test_compute_rrg_metrics_math():
    dates = pd.date_range("2024-01-01", periods=30, freq="W")
    cfg = Config(rolling_window=10)

    # Synthetic prices: Benchmark flat at 100, Sector BANK steadily outperforming (100 -> 200)
    benchmark_prices = np.full(30, 100.0)
    bank_prices = np.linspace(100.0, 200.0, 30)
    it_prices = np.full(30, 100.0)

    df = pd.DataFrame(
        {
            "^NSEI": benchmark_prices,
            "^NSEBANK": bank_prices,
            "^CNXIT": it_prices,
            "^CNXAUTO": it_prices,
            "^CNXFMCG": it_prices,
            "^CNXPHARMA": it_prices,
            "^CNXMETAL": it_prices,
        },
        index=dates,
    )

    rs_ratio, rs_mom = compute_rrg_metrics(df, cfg)

    assert rs_ratio.shape == (30, len(cfg.sectors))
    assert rs_mom.shape == (30, len(cfg.sectors))

    # Outperforming sector should have RS-Ratio > 100 in later periods
    assert rs_ratio["^NSEBANK"].dropna().iloc[-1] > 100.0


@patch("yfinance.download")
def test_download_prices_multiindex(mock_download, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = Config()

    dates = pd.date_range("2024-01-01", periods=10, freq="W")
    tickers = [cfg.benchmark, *cfg.sectors]

    # Create MultiIndex yfinance response structure
    tuples = [("Close", t) for t in tickers]
    cols = pd.MultiIndex.from_tuples(tuples, names=["Price", "Ticker"])
    mock_data = pd.DataFrame(np.random.rand(10, len(tickers)), index=dates, columns=cols)
    mock_download.return_value = mock_data

    prices = download_prices(cfg)
    assert prices.shape == (10, len(tickers))
    assert list(prices.columns) == tickers

    # Verify cache file creation
    cache_file = Path(f"rrg_cache_{_cache_key(cfg)}.csv")
    assert cache_file.exists()


@patch("yfinance.download")
def test_download_prices_cache_hit(mock_download, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = Config()
    tickers = [cfg.benchmark, *cfg.sectors]

    dates = pd.date_range("2024-01-01", periods=10, freq="W")
    fake_prices = pd.DataFrame(np.random.rand(10, len(tickers)), index=dates, columns=tickers)

    cache_file = Path(f"rrg_cache_{_cache_key(cfg)}.csv")
    fake_prices.to_csv(cache_file)

    # Should hit cache and NOT call yfinance.download
    prices = download_prices(cfg)
    mock_download.assert_not_called()
    assert prices.shape == (10, len(tickers))


def test_rrg_plotter_interactive_state():
    dates = pd.date_range("2024-01-01", periods=30, freq="W")
    cfg = Config(rolling_window=10, tail_periods=12)

    df_prices = pd.DataFrame(
        {
            "^NSEI": np.random.rand(30) + 100,
            "^NSEBANK": np.random.rand(30) + 100,
            "^CNXIT": np.random.rand(30) + 100,
            "^CNXAUTO": np.random.rand(30) + 100,
            "^CNXFMCG": np.random.rand(30) + 100,
            "^CNXPHARMA": np.random.rand(30) + 100,
            "^CNXMETAL": np.random.rand(30) + 100,
        },
        index=dates,
    )

    rs_ratio, rs_mom = compute_rrg_metrics(df_prices, cfg)

    plotter = RRGPlotter(rs_ratio, rs_mom, cfg)
    plotter.draw_sectors(12)

    # Initial state: all sectors visible
    assert all(plotter.visibility_state.values())

    # Deselect all
    plotter.deselect_all()
    assert not any(plotter.visibility_state.values())

    # Select all
    plotter.select_all()
    assert all(plotter.visibility_state.values())
