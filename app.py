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
    limit  = request.args.get("limit", type=int)   # if set, return last N candles only

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

    # Apply limit to df before building response (last N candles)
    if limit and limit > 0:
        df = df.tail(limit)

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

# ── Portfolio Performance API ────────────────────────────────────────────────
def _irish_performance():
    """Build equity curve for the Irish portfolio from its transaction CSV."""
    import csv
    from collections import defaultdict
    from datetime import datetime, timedelta
    import yfinance as yf
    import json as _json

    CSV_PATH = "/home/clawpi/.openclaw/data/portfolios/TransactionHistory-Irish.csv"
    PORT_PATH = "/home/clawpi/.openclaw/data/portfolios/portfolio_irish_latest.json"

    # ── 1. Parse CSV ────────────────────────────────────────────────────────
    trades = []    # [{date, ticker, action, qty, gbp}]
    cash_flows = []  # [{date, gbp}]
    tickers = set()
    stored_prices = {}   # ticker -> GBP per-share cost (for GBX normalisation)

    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            date   = row.get('Date', '').strip()
            action = row.get('Action', '').strip().upper()
            ticker = row.get('Ticker', '').strip()
            qty    = float(row.get('Quantity', '0') or 0)
            gbp    = float(row.get('GBP', '0') or 0)

            if action == 'DEPOSIT':
                cash_flows.append({'date': date, 'gbp': gbp})
            elif action == 'BUY' and ticker:
                tickers.add(ticker)
                trades.append({'date': date, 'ticker': ticker, 'action': 'BUY', 'qty': qty, 'gbp': abs(gbp)})
                # Store per-share price in GBP for GBX normalisation
                price_str = row.get('Price', '').strip()
                if price_str and price_str != '0':
                    stored_prices[ticker] = float(price_str)
            elif action == 'SELL' and ticker:
                tickers.add(ticker)
                trades.append({'date': date, 'ticker': ticker, 'action': 'SELL', 'qty': qty, 'gbp': abs(gbp)})

    if not trades:
        return jsonify({"error": "No trades found for Irish"}), 404

    trades.sort(key=lambda x: x['date'])
    cash_flows.sort(key=lambda x: x['date'])
    tickers = sorted(tickers)
    start_date = trades[0]['date']
    end_date   = datetime.now().strftime('%Y-%m-%d')

    # ── 2. Fetch historical prices (weekly) ─────────────────────────────────
    price_data = {}
    for tk in tickers:
        try:
            adj = yf.Ticker(tk).history(start=start_date, end=end_date, interval="1wk", auto_adjust=True)
            for ts, row in adj.iterrows():
                ds = str(ts.date())
                price_data.setdefault(tk, {})[ds] = float(row['Close'])
        except Exception:
            pass

    # ── 3. Build weekly curve ───────────────────────────────────────────────
    positions  = defaultdict(lambda: {'qty': 0, 'cost_gbp': 0.0})
    divs_received = 0.0
    net_invested  = 0.0
    account_cash  = 0.0
    curve = []
    trade_idx  = 0
    cash_idx   = 0

    cursor = datetime.strptime(min(trades[0]['date'] for t in trades), '%Y-%m-%d')
    end    = datetime.strptime(end_date, '%Y-%m-%d')
    while cursor.weekday() != 4:
        cursor += timedelta(days=1)

    while cursor <= end:
        week_end = cursor.strftime('%Y-%m-%d')

        while trade_idx < len(trades) and trades[trade_idx]['date'] <= week_end:
            tr = trades[trade_idx]
            trade_idx += 1
            if tr['action'] == 'BUY':
                positions[tr['ticker']]['qty']      += tr['qty']
                positions[tr['ticker']]['cost_gbp'] += tr['gbp']
                account_cash -= tr['gbp']
            elif tr['action'] == 'SELL':
                pos = positions[tr['ticker']]
                if pos['qty'] > 0:
                    avg = pos['cost_gbp'] / pos['qty']
                    sell_qty = min(tr['qty'], pos['qty'])
                    account_cash += sell_qty * avg
                    pos['qty']      -= sell_qty
                    pos['cost_gbp'] -= sell_qty * avg
            elif tr['action'] == 'DIV':
                divs_received += tr['gbp']
                account_cash  += tr['gbp']

        while cash_idx < len(cash_flows) and cash_flows[cash_idx]['date'] <= week_end:
            net_invested += cash_flows[cash_idx]['gbp']
            account_cash  += cash_flows[cash_idx]['gbp']
            cash_idx += 1

        # Portfolio value at week-end
        total_value = 0.0
        for tk, pos in positions.items():
            if pos['qty'] > 0:
                prices = price_data.get(tk, {})
                if prices:
                    past = sorted((d for d in prices if d <= week_end), reverse=True)
                    if past:
                        cur_price = prices[past[0]]
                        # GBX stocks (price in pence) need /100 to convert to GBP
                        stored_price = stored_prices.get(tk, cur_price)
                        if stored_price > 0 and cur_price > stored_price * 10:
                            cur_price = cur_price / 100
                        total_value += pos['qty'] * cur_price

        ret = (total_value + account_cash) / net_invested if net_invested > 0 else 0
        curve.append({
            'date':   week_end,
            'return': round(ret, 4),
            'value':  round(total_value + account_cash, 2),
        })

        cursor += timedelta(days=7)

    # ── 4. Override last point with portfolio JSON ───────────────────────────
    try:
        with open(PORT_PATH) as f:
            port = _json.load(f)
        holdings_now = {h['ticker']: h for h in port.get('holdings', [])}
        cur_equity = sum(h.get('value_gbp', 0) for h in holdings_now.values())
        cur_cash   = port['account'].get('cash_gbp', 0) or 0
        last_date  = curve[-1]['date'] if curve else end_date
        cur_net_inv = sum(cf['gbp'] for cf in cash_flows if cf['date'] <= last_date)
        if curve:
            curve[-1]['return'] = round((cur_equity + cur_cash) / cur_net_inv, 4) if cur_net_inv > 0 else 0
            curve[-1]['value']   = round(cur_equity + cur_cash, 2)
    except Exception:
        pass

    return jsonify({
        'tickers':      tickers,
        'start':        start_date,
        'net_invested': round(net_invested, 2),
        'curve':        curve,
    })


@app.route("/api/portfolio/performance")
def api_portfolio_performance():
    """
    Returns equity curve for a portfolio as JSON.
    Ignite: reconstructed from CSV transaction history + historical yfinance prices.
    Irish: not supported (no transaction history available).
    """
    import csv, re
    from collections import defaultdict
    from datetime import datetime, timedelta

    key = request.args.get("key", "").strip()

    if key == "irish":
        return _irish_performance()

    if key != "ignite":
        return jsonify({"error": "Unknown portfolio: " + key}), 400

    CSV_PATH = "/home/clawpi/.openclaw/media/inbound/TransactionHistory-WX3OA-_01-04-2012_-_24-04-2026---7e27c37b-f1b1-43c5-9133-21140ab11330.csv"
    RATE = 1.3502  # USD/GBP

    ALIASES = {
        'BERKSHIRE HATHAWAY': 'BRK-B',
        'MARKEL CORP':        'MKL',
        'WALT DISNEY':        'DIS',
        'ALIBABA GROUP':      'BABA',
        'ALPHABET INC':       'GOOGL',
        'ALPHABET':           'GOOGL',
        'PDD HOLDINGS':       'PDD',
        'PAYPAL':             'PYPL',
        'SMITH & WESSON':     'SWBI',
        'DIDI GLOBAL':        'DIDIY',   # switched to DIDIY after delisting
        'ISHARES USD TREASURY':'IB01.L',
    }

    def ticker_of(mkt):
        m = mkt.upper()
        return next((t for name, t in ALIASES.items() if name in m), None)

    # ── 1. Parse CSV trades ─────────────────────────────────────────────────
    trades = []   # [{date, ticker, action, qty, price_usd, gbp}]
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            date   = row.get('DateUtc', '')[:10]
            mkt    = row.get('MarketName', '')
            summ   = row.get('Summary', '')
            ctype  = row.get('Transaction type', '')
            gbp    = float(row.get('PL Amount', '0').replace(',', '') or 0)

            t = ticker_of(mkt)
            if not t:
                continue

            m2 = re.search(r'CONS\s*(\d+)\s*@\s*([\d.]+)', mkt)
            if m2:
                qty   = int(m2.group(1))
                price = float(m2.group(2))
                if ctype == 'WITH':   # BUY
                    trades.append({'date': date, 'ticker': t, 'action': 'BUY',  'qty': qty, 'price': price, 'gbp': abs(gbp)})
                elif ctype == 'DEPO' and gbp < 0:  # SELL
                    trades.append({'date': date, 'ticker': t, 'action': 'SELL', 'qty': qty, 'price': price, 'gbp': abs(gbp)})
            elif 'DIVIDEND' in summ and ctype == 'DEPO' and gbp > 0:
                trades.append({'date': date, 'ticker': t, 'action': 'DIV', 'qty': 0, 'price': 0, 'gbp': gbp})

    trades.sort(key=lambda x: x['date'])

    # ── 1b. Parse cash flows (card payments + FX transfers — for net invested) ──
    cash_flows = []   # [{date, gbp}] — positive = cash in, negative = cash out
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            mkt  = row.get('MarketName', '')
            gbp  = float(row.get('PL Amount', '0').replace(',', '') or 0)
            date = row.get('DateUtc', '')[:10]
            if 'Card payment' in mkt:
                cash_flows.append({'date': date, 'gbp': gbp})   # positive = deposit
            elif 'Transfer from GBP' in mkt:
                cash_flows.append({'date': date, 'gbp': gbp})   # net FX (can be + or -)
    cash_flows.sort(key=lambda x: x['date'])

    # ── 2. Determine date range ───────────────────────────────────────────────
    if not trades:
        return jsonify({"error": "No trades found"}), 404
    start_date = trades[0]['date']
    end_date   = datetime.now().strftime('%Y-%m-%d')

    # ── 3. Fetch historical prices for all tickers ────────────────────────────
    tickers = sorted(set(t['ticker'] for t in trades))
    price_data = {}   # ticker -> {date_str -> usd_close}
    for tk in tickers:
        try:
            # Use daily data for accurate weekly curve sampling
            adj = yf.Ticker(tk).history(start=start_date, end=end_date, interval="1wk", auto_adjust=True)
            for ts, row in adj.iterrows():
                ds = str(ts.date())
                price_data.setdefault(tk, {})[ds] = float(row['Close'])
        except Exception:
            pass

    # ── 4. Build weekly equity curve ────────────────────────────────────────
    # Use week-ending dates (Fridays) as the sampling points
    from datetime import datetime
    dates = sorted(set(t['date'] for t in trades))

    # Walk through chronologically, track positions, sample portfolio value weekly
    positions  = defaultdict(lambda: {'qty': 0, 'cost_gbp': 0.0})
    divs_received = 0.0
    net_invested  = 0.0   # cumulative cash in via card payments + FX
    account_cash  = 0.0   # actual cash in the account at this point
    curve = []   # [{date, return, value}]
    trade_idx  = 0
    cash_idx   = 0

    # Sample every Friday from start to end
    cursor = datetime.strptime(min(dates), '%Y-%m-%d')
    end    = datetime.strptime(end_date, '%Y-%m-%d')
    # Find next Friday
    while cursor.weekday() != 4:
        cursor += timedelta(days=1)

    while cursor <= end:
        week_end = cursor.strftime('%Y-%m-%d')

        # Apply all trades up to and including this week
        while trade_idx < len(trades) and trades[trade_idx]['date'] <= week_end:
            tr = trades[trade_idx]
            trade_idx += 1
            if tr['action'] == 'BUY':
                positions[tr['ticker']]['qty']      += tr['qty']
                positions[tr['ticker']]['cost_gbp'] += tr['gbp']
                account_cash -= tr['gbp']   # cash pays for the buy
            elif tr['action'] == 'SELL':
                pos = positions[tr['ticker']]
                if pos['qty'] > 0:
                    avg = pos['cost_gbp'] / pos['qty']
                    sell_qty = min(tr['qty'], pos['qty'])
                    sell_proceeds = sell_qty * avg   # GBP received from sale
                    account_cash += sell_proceeds
                    pos['qty']      -= sell_qty
                    pos['cost_gbp'] -= sell_qty * avg
            elif tr['action'] == 'DIV':
                divs_received += tr['gbp']
                account_cash += tr['gbp']   # dividend adds to cash

        # Apply cash flows (card payments + FX) up to this week
        while cash_idx < len(cash_flows) and cash_flows[cash_idx]['date'] <= week_end:
            net_invested += cash_flows[cash_idx]['gbp']
            account_cash  += cash_flows[cash_idx]['gbp']
            cash_idx += 1

        # Portfolio value at this week-end using historical prices
        total_value = 0.0
        for tk, pos in positions.items():
            if pos['qty'] > 0:
                prices = price_data.get(tk, {})
                if prices:
                    past = sorted((d for d in prices if d <= week_end), reverse=True)
                    if past:
                        total_value += positions[tk]['qty'] * prices[past[0]] / RATE

        # Cash in the account at this week-end (card deposits accumulated, net of trades)
        # cash = net_invested - sum of all BUY costs (since card deposits go to cash, trades deduct from it)
        account_cash = net_invested - sum(p['cost_gbp'] for p in positions.values())
        # Return = (holdings value + cash + divs) / net invested
        ret = (total_value + account_cash + divs_received) / net_invested if net_invested > 0 else 0
        curve.append({
            'date':   week_end,
            'return': round(ret, 4),
            'value':  round(total_value + account_cash + divs_received, 2),
        })

        cursor += timedelta(days=7)

    # ── 5. Override last point with accurate IG portfolio JSON values ─
    try:
        with open("/home/clawpi/.openclaw/data/portfolios/portfolio_ignite_latest.json") as f:
            port = json.load(f)
        holdings_now = {h["ticker"]: h for h in port.get("holdings", [])}
        cur_equity   = sum(h.get("cur_val_gbp", 0) for h in holdings_now.values())
        cur_cash     = port.get("cash_gbp", 0) or 0
        cur_divs     = port.get("total_divs_gbp", 0) or 0
        last_date = curve[-1]["date"] if curve else end_date
        cur_net_inv = sum(cf["gbp"] for cf in cash_flows if cf["date"] <= last_date)
        if curve:
            curve[-1]["return"] = round((cur_equity + cur_cash + cur_divs) / cur_net_inv, 4) if cur_net_inv > 0 else 0
            curve[-1]["value"]   = round(cur_equity + cur_cash + cur_divs, 2)
    except Exception:
        pass

    return jsonify({
        "name":    "Ignite",
        "start":   start_date,
        "curve":   curve,
        "tickers": tickers,
    })



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
                    # cash_gbp lives at top-level for Ignite, inside account for Irish
                    cash_gbp = data.get("cash_gbp") or data.get("account", {}).get("cash_gbp") or 0
                    result[key] = {
                        "name":        data.get("account", {}).get("platform", key.title()),
                        "as_of":       data.get("date", ""),
                        "holdings":    data.get("holdings", []),
                        "all_tickers": data.get("all_tickers", [x["ticker"] for x in data.get("holdings", [])]),
                        "cash_gbp":    cash_gbp,
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
