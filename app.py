#!/usr/bin/env /home/clawpi/.openclaw/venv/bin/python3
"""
TA Terminal — Bloomberg-style charting app
Flask + Plotly + yfinance
Access: http://clawpi.local:5000
"""

import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import yfinance as yf
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

app = Flask(__name__)
app.template_folder = "templates"

# ── Data Cache ────────────────────────────────────────────────────────────────
DATA_DIR = "/home/clawpi/.openclaw/data/ohlcv"
CACHE_FRESHNESS = {  # max age before re-fetch (days)
    "daily":   1,
    "weekly":  7,
    "monthly": 30,
}
MAX_LOOKBACK = {  # longest history to ever fetch (years)
    "daily":   15,
    "weekly":  20,
    "monthly": 30,
}
PERIOD_MAP = {  # yfinance period for full fetch
    "daily":   "15y",
    "weekly":  "20y",
    "monthly": "30y",
}

def ticker_data_dir(ticker):
    d = os.path.join(DATA_DIR, ticker.upper())
    os.makedirs(d, exist_ok=True)
    return d

def cache_path(ticker, tf):
    return os.path.join(ticker_data_dir(ticker), f"{tf}.csv")

def is_cache_fresh(ticker, tf):
    p = cache_path(ticker, tf)
    if not os.path.exists(p):
        return False
    age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(p))).days
    return age_days < CACHE_FRESHNESS.get(tf, 1)

def load_from_cache(ticker, tf):
    p = cache_path(ticker, tf)
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_csv(p, parse_dates=["Date"], index_col="Date")
        df.index = pd.DatetimeIndex(df.index)
        return df.sort_index()
    except Exception:
        return None

def save_to_cache(ticker, tf, df):
    p = cache_path(ticker, tf)
    df.sort_index().to_csv(p)

def fetch_and_cache(ticker, tf, period=None, start=None, end=None):
    """Fetch from yfinance and save to cache. Appends to existing cache if any."""
    interval_map = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}
    interval = interval_map.get(tf, "1d")

    # Determine period / start
    if period is None:
        max_years = MAX_LOOKBACK.get(tf, 15)
        period = f"{max_years}y"

    t = yf.Ticker(ticker)
    if start:
        df = t.history(start=start, end=end, interval=interval, auto_adjust=False)
    else:
        df = t.history(period=period, interval=interval, auto_adjust=False)

    if df.empty:
        return df

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = [c.lower() for c in df.columns]
    # Normalise to UTC-naive DatetimeIndex before caching
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)

    # Append or replace in cache
    existing = load_from_cache(ticker, tf)
    if existing is not None and start:
        # Merge: keep existing, fill gap with new
        combined = pd.concat([existing, df]).sort_index()
        # Remove duplicates (keep newer)
        combined = combined[~combined.index.duplicated(keep="last")]
    elif existing is not None:
        combined = df  # full refresh
    else:
        combined = df

    save_to_cache(ticker, tf, combined)
    return combined

# ── TA Indicators ────────────────────────────────────────────────────────────

def calc_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calc_sma(series, span):
    return series.rolling(span).mean()

def calc_bollinger(series, window=20, num_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return sma + num_std * std, sma, sma - num_std * std

def calc_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

# ── OHLCV Response Builder ───────────────────────────────────────────────────

def build_ohlcv_response(ticker, tf, df):
    """Compute indicators and return JSON-serialisable dict for a df."""
    df = df.copy()
    close = df["close"]

    ema_a  = calc_ema(close, 12)
    ema_b  = calc_ema(close, 26)
    sma_a  = calc_sma(close, 20)
    sma_b  = calc_sma(close, 60)
    sma_c  = calc_sma(close, 120)
    sma_d  = calc_sma(close, 200)
    bb_up, bb_mid, bb_low = calc_bollinger(close, 20, 2)
    rsi    = calc_rsi(close, 14)
    macd_line, macd_signal, macd_hist = calc_macd(close, 12, 26, 9)

    def clean_nan(obj):
        if isinstance(obj, (float, np.floating)):
            return None if np.isnan(obj) else float(obj)
        return obj

    return {
        "ticker":      ticker,
        "tf":          tf,
        "dates":       df.index.strftime("%Y-%m-%d").tolist(),
        "opens":       [clean_nan(v) for v in df["open"].tolist()],
        "highs":       [clean_nan(v) for v in df["high"].tolist()],
        "lows":        [clean_nan(v) for v in df["low"].tolist()],
        "closes":      [clean_nan(v) for v in df["close"].tolist()],
        "volumes":     [clean_nan(v) for v in (df["volume"] / 1e6).tolist()],
        "ema_a":       [clean_nan(v) for v in ema_a.tolist()],
        "ema_b":       [clean_nan(v) for v in ema_b.tolist()],
        "sma_a":       [clean_nan(v) for v in sma_a.tolist()],
        "sma_b":       [clean_nan(v) for v in sma_b.tolist()],
        "sma_c":       [clean_nan(v) for v in sma_c.tolist()],
        "sma_d":       [clean_nan(v) for v in sma_d.tolist()],
        "bb_up":       [clean_nan(v) for v in bb_up.tolist()],
        "bb_mid":      [clean_nan(v) for v in bb_mid.tolist()],
        "bb_low":      [clean_nan(v) for v in bb_low.tolist()],
        "rsi":         [clean_nan(v) for v in rsi.tolist()],
        "macd_line":   [clean_nan(v) for v in macd_line.tolist()],
        "macd_signal": [clean_nan(v) for v in macd_signal.tolist()],
        "macd_hist":   [clean_nan(v) for v in macd_hist.tolist()],
    }


# ── Chart Data API ───────────────────────────────────────────────────────────

@app.route("/api/chart_data")
def chart_data():
    ticker = request.args.get("ticker", "^SPX").strip().upper()
    tf     = request.args.get("tf", "daily")
    refresh = request.args.get("refresh", "false").lower() == "true"

    # Indicator settings
    ema1    = int(request.args.get("ema1", 12))
    ema2    = int(request.args.get("ema2", 26))
    sma1    = int(request.args.get("sma1", 20))
    sma2    = int(request.args.get("sma2", 60))
    sma3    = int(request.args.get("sma3", 120))
    sma4    = int(request.args.get("sma4", 200))
    bb_w    = int(request.args.get("bb_window", 20))
    bb_s    = float(request.args.get("bb_std", 2.0))
    rsi_w   = int(request.args.get("rsi_window", 14))
    macd_f  = int(request.args.get("macd_fast", 12))
    macd_s  = int(request.args.get("macd_slow", 26))
    macd_si = int(request.args.get("macd_signal", 9))

    # Load from cache or fetch
    fresh = is_cache_fresh(ticker, tf)
    if refresh or not fresh:
        df = fetch_and_cache(ticker, tf)
    else:
        df = load_from_cache(ticker, tf)
        if df is None:
            df = fetch_and_cache(ticker, tf)

    if df is None or df.empty:
        return jsonify({"error": f"No data for {ticker}"}), 404

    result = build_ohlcv_response(ticker, tf, df)
    result["settings"] = {
        "ema1": ema1, "ema2": ema2,
        "sma1": sma1, "sma2": sma2, "sma3": sma3, "sma4": sma4,
        "bb_window": bb_w, "bb_std": bb_s,
        "rsi_window": rsi_w,
        "macd_fast": macd_f, "macd_slow": macd_s, "macd_signal": macd_si,
    }
    result["loaded_from"] = result["dates"][0]
    result["loaded_to"]   = result["dates"][-1]
    return jsonify(result)


@app.route("/api/chart_data/more")
def chart_data_more():
    """Fetch older data for a ticker/timeframe and return as JSON (for prepend)."""
    ticker = request.args.get("ticker", "").strip().upper()
    tf     = request.args.get("tf", "daily")

    if not ticker:
        return jsonify({"error": "ticker required"}), 400

    df = load_from_cache(ticker, tf)
    if df is None or df.empty:
        return jsonify({"error": f"No cached data for {ticker}"}), 404

    result = build_ohlcv_response(ticker, tf, df)
    result["loaded_from"] = result["dates"][0]
    result["loaded_to"]   = result["dates"][-1]
    return jsonify(result)


# ── Watchlist API ────────────────────────────────────────────────────────────
WATCHLIST_PATH = "/home/clawpi/.openclaw/data/watchlist/watchlist.json"
PORTFOLIO_DIR  = "/home/clawpi/.openclaw/data/portfolios"

@app.route("/api/watchlist")
def api_watchlist():
    try:
        with open(WATCHLIST_PATH) as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Portfolios API ────────────────────────────────────────────────────────────
@app.route("/api/portfolios")
def api_portfolios():
    try:
        files = {"irish": "portfolio_irish_latest.json",
                 "ignite": "portfolio_ignite_latest.json"}
        result = {}
        for key, fname in files.items():
            path = os.path.join(PORTFOLIO_DIR, fname)
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                    result[key] = {
                        "name":     data.get("account", {}).get("platform", key.title()),
                        "as_of":    data.get("date", ""),
                        "holdings": data.get("holdings", []),
                    }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Main Page ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    default_ticker = request.args.get("ticker", "SPY")
    return render_template("index.html", default_ticker=default_ticker)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
