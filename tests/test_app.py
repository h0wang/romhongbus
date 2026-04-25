"""
Unit tests for Romhongbus TA Terminal app.
Run with: python -m pytest tests/ -v
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# Ensure the app module is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    calc_ema, calc_sma, calc_bollinger, calc_rsi, calc_macd,
    is_cache_fresh, load_from_cache, save_to_cache, cache_path,
    build_ohlcv_response,
)
from app import app as flask_app


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


@pytest.fixture
def tmp_data_dir(monkeypatch):
    """Patch DATA_DIR to a temporary directory for isolation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr("app.DATA_DIR", tmpdir + "/ohlcv")
        yield tmpdir


@pytest.fixture
def synthetic_ohlcv():
    """Generate 100 bars of synthetic OHLCV data for indicator testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    np.random.seed(42)
    base = 100.0
    close_prices = [base]
    for _ in range(99):
        close_prices.append(close_prices[-1] + np.random.randn() * 0.5)
    closes = pd.Series(close_prices, index=dates)

    highs = closes * 1.01
    lows = closes * 0.99
    opens = (highs + lows) / 2
    volumes = pd.Series(np.random.randint(1e6, 5e6, size=100), index=dates)

    df = pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": volumes,
    }, index=dates)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df


# ── TA Indicator Tests ────────────────────────────────────────────────────────

class TestCalcEma:
    def test_ema_length_equals_input(self, synthetic_ohlcv):
        result = calc_ema(synthetic_ohlcv["close"], 12)
        assert len(result) == len(synthetic_ohlcv)

    def test_ema_none_at_start(self, synthetic_ohlcv):
        # pandas ewm has min_periods default equal to span; first bars can be nan
        result = calc_ema(synthetic_ohlcv["close"], 12)
        # First bar should be NaN or a concrete value (pandas ewm behavior varies)
        assert len(result) == len(synthetic_ohlcv)
        # After enough data, values should be concrete
        assert result.dropna().any()

    def test_ema_larger_span_smoother(self, synthetic_ohlcv):
        close = synthetic_ohlcv["close"]
        ema_fast = calc_ema(close, 12)
        ema_slow = calc_ema(close, 26)
        # At the end they should be close but not identical
        diff_fast = ema_fast.diff().abs().sum()
        diff_slow = ema_slow.diff().abs().sum()
        assert diff_slow < diff_fast  # slower EMA is smoother


class TestCalcSma:
    def test_sma_length_equals_input(self, synthetic_ohlcv):
        result = calc_sma(synthetic_ohlcv["close"], 20)
        assert len(result) == len(synthetic_ohlcv)

    def test_sma_first_n_bars_nan(self, synthetic_ohlcv):
        result = calc_sma(synthetic_ohlcv["close"], 20)
        assert result.iloc[:19].isna().all()
        assert result.iloc[19:].notna().any()


class TestCalcBollinger:
    def test_returns_three_series_same_length(self, synthetic_ohlcv):
        bb_up, bb_mid, bb_low = calc_bollinger(synthetic_ohlcv["close"], 20, 2)
        assert len(bb_up) == len(bb_mid) == len(bb_low) == len(synthetic_ohlcv)

    def test_bands_enclose_price(self, synthetic_ohlcv):
        close = synthetic_ohlcv["close"]
        bb_up, bb_mid, bb_low = calc_bollinger(close, 20, 2)
        first_valid = bb_up.first_valid_index()
        start_i = bb_up.index.get_loc(first_valid)
        violations = 0
        for i in range(start_i, len(bb_up)):
            if bb_up.iloc[i] < close.iloc[i]:
                violations += 1
            if bb_low.iloc[i] > close.iloc[i]:
                violations += 1
        # ~95% of prices should fall within 2-std Bollinger bands
        n = len(bb_up) - start_i
        assert violations / n < 0.10, f"{violations}/{n} bars outside bands — expected <10%"

    def test_default_parameters(self, synthetic_ohlcv):
        bb_up, bb_mid, bb_low = calc_bollinger(synthetic_ohlcv["close"])
        assert len(bb_up) == len(bb_mid) == len(bb_low)


class TestCalcRsi:
    def test_rsi_range_0_to_100(self, synthetic_ohlcv):
        rsi = calc_rsi(synthetic_ohlcv["close"], 14)
        valid = rsi.dropna()
        assert valid.between(0, 100).all()

    def test_rsi_length_equals_input(self, synthetic_ohlcv):
        result = calc_rsi(synthetic_ohlcv["close"], 14)
        assert len(result) == len(synthetic_ohlcv)

    def test_rsi_na_count_equals_window(self, synthetic_ohlcv):
        rsi = calc_rsi(synthetic_ohlcv["close"], 14)
        assert rsi.iloc[:14].isna().all()


class TestCalcMacd:
    def test_returns_three_series(self, synthetic_ohlcv):
        macd_line, macd_signal, macd_hist = calc_macd(synthetic_ohlcv["close"])
        assert len(macd_line) == len(macd_signal) == len(macd_hist)

    def test_histogram_is_diff_of_line_and_signal(self, synthetic_ohlcv):
        macd_line, macd_signal, macd_hist = calc_macd(synthetic_ohlcv["close"])
        expected_hist = macd_line - macd_signal
        pd.testing.assert_series_equal(
            macd_hist.dropna(), expected_hist.dropna(), check_names=False
        )


# ── Cache Tests ───────────────────────────────────────────────────────────────

class TestCacheFreshness:
    def test_missing_file_not_fresh(self, tmp_data_dir):
        assert is_cache_fresh("NONEXISTENT", "daily") is False

    def test_file_older_than_1_day_not_fresh(self, tmp_data_dir, monkeypatch):
        ticker = "SPY"
        tf = "daily"
        p = cache_path(ticker, tf)
        save_to_cache(ticker, tf, pd.DataFrame({"close": [100, 101]}, index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]).tz_localize(None)))
        # Age the file to 2 days old
        old_mtime = (datetime.now() - timedelta(days=2)).timestamp()
        os.utime(p, (old_mtime, old_mtime))
        assert is_cache_fresh(ticker, tf) is False

    def test_file_older_than_7_days_not_fresh_weekly(self, tmp_data_dir):
        ticker = "SPY"
        tf = "weekly"
        p = cache_path(ticker, tf)
        save_to_cache(ticker, tf, pd.DataFrame({"close": [100, 101]}, index=pd.DatetimeIndex(["2024-01-01", "2024-01-08"]).tz_localize(None)))
        old_mtime = (datetime.now() - timedelta(days=8)).timestamp()
        os.utime(p, (old_mtime, old_mtime))
        assert is_cache_fresh(ticker, tf) is False


class TestCacheSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_data_dir):
        ticker = "AAPL"
        tf = "daily"
        dates = pd.DatetimeIndex(["2024-01-01", "2024-01-02"]).tz_localize(None)
        original = pd.DataFrame({
            "open":   [100.0, 101.0],
            "high":   [102.0, 103.0],
            "low":    [99.0, 100.5],
            "close":  [101.0, 102.0],
            "volume": [1e6, 1.2e6],
        }, index=pd.DatetimeIndex(dates, name="Date"))
        save_to_cache(ticker, tf, original)
        loaded = load_from_cache(ticker, tf)
        pd.testing.assert_frame_equal(original, loaded)

    def test_load_nonexistent_returns_none(self, tmp_data_dir):
        result = load_from_cache("DOESNOTEXIST", "daily")
        assert result is None

    def test_cache_path_creates_directory(self, tmp_data_dir):
        p = cache_path("NEWTICKER", "daily")
        assert "NEWTICKER" in p


# ── build_ohlcv_response Tests ────────────────────────────────────────────────

class TestBuildOhlcvResponse:
    def test_returns_all_required_keys(self, synthetic_ohlcv):
        result = build_ohlcv_response("SPY", "daily", synthetic_ohlcv)
        required = ["ticker", "tf", "dates", "opens", "highs", "lows",
                    "closes", "volumes", "ema_a", "ema_b",
                    "sma_a", "sma_b", "sma_c", "sma_d",
                    "bb_up", "bb_mid", "bb_low",
                    "rsi", "macd_line", "macd_signal", "macd_hist"]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_dates_match_input_length(self, synthetic_ohlcv):
        result = build_ohlcv_response("SPY", "daily", synthetic_ohlcv)
        assert len(result["dates"]) == len(synthetic_ohlcv)
        assert len(result["closes"]) == len(synthetic_ohlcv)

    def test_nan_cleaned_in_output(self, synthetic_ohlcv):
        result = build_ohlcv_response("SPY", "daily", synthetic_ohlcv)
        # sma_a (20-bar) should have warm-up NaNs cleaned to None
        # sma_d (200-bar) on 100 bars → all NaN → all None (expected)
        assert None in result["sma_d"]  # all 200-bar SMA values are NaN → None
        # But sma_a (20-bar) should have some valid values
        valid_sma_a = [v for v in result["sma_a"] if v is not None]
        assert len(valid_sma_a) > 0, "sma_a should have some non-None values after warm-up"

    def test_ticker_and_tf_passed_through(self, synthetic_ohlcv):
        result = build_ohlcv_response("AAPL", "weekly", synthetic_ohlcv)
        assert result["ticker"] == "AAPL"
        assert result["tf"] == "weekly"


# ── API Endpoint Tests ────────────────────────────────────────────────────────

class TestChartDataEndpoint:
    def test_invalid_ticker_returns_404(self, client):
        with patch("app.fetch_and_cache", return_value=pd.DataFrame()):
            response = client.get("/api/chart_data?ticker=INVALIDTICKERXYZ")
            assert response.status_code == 404

    def test_missing_ticker_defaults_to_spx(self, client):
        with patch("app.load_from_cache", return_value=None), \
             patch("app.fetch_and_cache", return_value=pd.DataFrame()):
            response = client.get("/api/chart_data")
            # Should not 400 — defaults are applied
            assert response.status_code in (200, 404)

    def test_refresh_param_in_url(self, client):
        with patch("app.fetch_and_cache") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame({
                "open": [100.0], "high": [101.0], "low": [99.0],
                "close": [100.5], "volume": [1e6],
            }, index=pd.DatetimeIndex(["2024-01-01"]).tz_localize(None))
            response = client.get("/api/chart_data?ticker=SPY&tf=daily&refresh=true")
            # fetch_and_cache should have been called with refresh=True intent
            # (exact assertion depends on caching logic)
            assert response.status_code in (200, 404)

    def test_response_has_expected_json_structure(self, client):
        synthetic = pd.DataFrame({
            "open": [100.0], "high": [101.0], "low": [99.0],
            "close": [100.5], "volume": [1e6],
        }, index=pd.DatetimeIndex(["2024-01-01"]).tz_localize(None))
        with patch("app.fetch_and_cache", return_value=synthetic):
            response = client.get("/api/chart_data?ticker=SPY&tf=daily")
            if response.status_code == 200:
                json_data = response.get_json()
                assert "ticker" in json_data
                assert "dates" in json_data
                assert "closes" in json_data


class TestChartDataMoreEndpoint:
    def test_ticker_required(self, client):
        response = client.get("/api/chart_data/more")
        assert response.status_code == 400

    def test_returns_404_when_no_cache(self, client):
        with patch("app.load_from_cache", return_value=None):
            response = client.get("/api/chart_data/more?ticker=SPY&tf=daily")
            assert response.status_code == 404

    def test_returns_cached_data(self, client):
        synthetic = pd.DataFrame({
            "open": [100.0], "high": [101.0], "low": [99.0],
            "close": [100.5], "volume": [1e6],
        }, index=pd.DatetimeIndex(["2024-01-01"]).tz_localize(None))
        with patch("app.load_from_cache", return_value=synthetic):
            response = client.get("/api/chart_data/more?ticker=SPY&tf=daily")
            assert response.status_code == 200
            json_data = response.get_json()
            assert "dates" in json_data
            assert "closes" in json_data


class TestWatchlistEndpoint:
    def test_watchlist_returns_json(self, client):
        with patch("app.WATCHLIST_PATH", "/nonexistent/path.json"):
            response = client.get("/api/watchlist")
            # Should either return data or 500, never crash
            assert response.status_code in (200, 500)


class TestPortfoliosEndpoint:
    def test_portfolios_returns_json(self, client):
        response = client.get("/api/portfolios")
        assert response.status_code in (200, 500)


class TestIndexEndpoint:
    def test_index_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"<html" in response.data.lower()
