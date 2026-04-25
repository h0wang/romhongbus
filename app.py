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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

app = Flask(__name__)
app.template_folder = "templates"

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

# ── Chart Data API ───────────────────────────────────────────────────────────

@app.route("/api/chart_data")
def chart_data():
    ticker = request.args.get("ticker", "^SPX").strip().upper()
    tf = request.args.get("tf", "daily")  # daily | weekly | monthly

    # Indicator settings from query params
    ema1  = int(request.args.get("ema1", 12))
    ema2  = int(request.args.get("ema2", 26))
    sma1  = int(request.args.get("sma1", 20))
    sma2  = int(request.args.get("sma2", 60))
    sma3  = int(request.args.get("sma3", 120))
    sma4  = int(request.args.get("sma4", 200))
    bb_w  = int(request.args.get("bb_window", 20))
    bb_s  = float(request.args.get("bb_std", 2.0))
    rsi_w = int(request.args.get("rsi_window", 14))
    macd_f = int(request.args.get("macd_fast", 12))
    macd_s = int(request.args.get("macd_slow", 26))
    macd_sig = int(request.args.get("macd_signal", 9))

    interval_map = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}
    period_map = {"daily": "4y", "weekly": "10y", "monthly": "30y"}
    period = period_map.get(tf, "2y")
    interval = interval_map.get(tf, "1d")

    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval)
        if df.empty:
            return jsonify({"error": f"No data for {ticker}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    df = df.copy()
    close = df["Close"]

    # ── Indicators ──
    ema_a  = calc_ema(close, ema1)
    ema_b  = calc_ema(close, ema2)
    sma_a  = calc_sma(close, sma1)
    sma_b  = calc_sma(close, sma2)
    sma_c  = calc_sma(close, sma3)
    sma_d  = calc_sma(close, sma4)
    bb_up, bb_mid, bb_low = calc_bollinger(close, bb_w, bb_s)
    rsi    = calc_rsi(close, rsi_w)
    macd_line, macd_signal, macd_hist = calc_macd(close, macd_f, macd_s, macd_sig)

    # ── Build OHLC for candlesticks ──
    dates     = df.index.strftime("%Y-%m-%d").tolist()
    opens     = df["Open"].tolist()
    highs     = df["High"].tolist()
    lows      = df["Low"].tolist()
    closes    = close.tolist()
    volumes   = (df["Volume"] / 1e6).tolist()  # in millions

    def clean_nan(obj):
        if isinstance(obj, float):
            return None if np.isnan(obj) else obj
        return obj

    return jsonify({
        "ticker":    ticker,
        "tf":        tf,
        "dates":     dates,
        "opens":     [clean_nan(v) for v in opens],
        "highs":     [clean_nan(v) for v in highs],
        "lows":      [clean_nan(v) for v in lows],
        "closes":    [clean_nan(v) for v in closes],
        "volumes":   [clean_nan(v) for v in volumes],
        "ema_a":     [clean_nan(v) for v in ema_a.tolist()],
        "ema_b":     [clean_nan(v) for v in ema_b.tolist()],
        "sma_a":     [clean_nan(v) for v in sma_a.tolist()],
        "sma_b":     [clean_nan(v) for v in sma_b.tolist()],
        "sma_c":     [clean_nan(v) for v in sma_c.tolist()],
        "sma_d":     [clean_nan(v) for v in sma_d.tolist()],
        "bb_up":     [clean_nan(v) for v in bb_up.tolist()],
        "bb_mid":    [clean_nan(v) for v in bb_mid.tolist()],
        "bb_low":    [clean_nan(v) for v in bb_low.tolist()],
        "rsi":       [clean_nan(v) for v in rsi.tolist()],
        "macd_line":   [clean_nan(v) for v in macd_line.tolist()],
        "macd_signal": [clean_nan(v) for v in macd_signal.tolist()],
        "macd_hist":   [clean_nan(v) for v in macd_hist.tolist()],
        "settings": {
            "ema1": ema1, "ema2": ema2,
            "sma1": sma1, "sma2": sma2, "sma3": sma3, "sma4": sma4,
            "bb_window": bb_w, "bb_std": bb_s,
            "rsi_window": rsi_w,
            "macd_fast": macd_f, "macd_slow": macd_s, "macd_signal": macd_sig,
        }
    })


# ── Watchlist API ──────────────────────────────────────────────────────────
WATCHLIST_PATH = "/home/clawpi/.openclaw/workspace/watchlist.json"
PORTFOLIO_DIR  = "/home/clawpi/.openclaw/workspace"

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
    default_ticker = request.args.get("ticker", "DGE.L")
    return render_template("index.html", default_ticker=default_ticker)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
