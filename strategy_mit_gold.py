#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FULL STRATEGY SCRIPT (READABLE + COMMENTED VERSION)
--------------------------------------------------
Features:
- GOLD Vault (principal-only, LIFO-based withdrawal with date validation)
- B1.2 Ladder Buy Engine (15% / 20% / 25% from ATH)
- ATR + Volume + RSI Low-Marker Heavy-Buy Engine
- ATH_EVENT system (only 1 printed per month)
- Full Take-Profit State Machine:
      New ATH → TP1 → TP2 → TP3 → lock → only reset on next ATH
- Per-ticker detailed tabular report
- Combined multi-ticker portfolio report
- Gold vault report with remaining principal and profit
- Clean printing with ANSI colors for readability
"""

import os
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime


# ---------------------------------------------------------------------------
# ANSI COLORS FOR PRETTY TERMINAL OUTPUT
# ---------------------------------------------------------------------------

COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_CYAN = "\033[36m"
COLOR_MAGENTA = "\033[35m"
COLOR_WHITE = "\033[97m"


# ---------------------------------------------------------------------------
# GLOBAL CONFIGURATION CONSTANTS
# ---------------------------------------------------------------------------

# Gold ticker used for the vault deposits (GLD or IAU are common)
GOLD_TICKER = "GLD"

# Monthly gold deposit amount
MONTHLY_GOLD_DEPOSIT = 1000.0

# Strategy start date
STRATEGY_START_DATE = "2021-12-01"

# Data loading start date (historical data)
HISTORICAL_START_DATE = "2019-01-01"

# List of tickers to process
TICKERS = [
    "NVDA", "MSFT", "PLTR", "TSLA", "AMZN",
    "GOOG", "META", "AAPL"
]

#TICKERS = [
#    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
#    "ADANIGREEN.NS", "TATAPOWER.NS", "POLYCAB.NS", "DMART.NS", "TATAELXSI.NS"
#]

# Take-profit fraction per phase
TP_FRAC = 0.15

# Technical indicator windows
RSI_PERIOD = 14
VOL_PERIOD = 60
ATR_PERIOD = 60

# Checkpoint settings
CHECKPOINT_FILE = "strategy_checkpoint.pkl"
ENABLE_CHECKPOINTING = False
RESUME_FROM_CHECKPOINT = False
LOG_DIR = "logs"

# Parameter sets
DEFAULT_PARAMETERS = {
    "name": "US",
    "B1_THRESHOLDS": [0.15, 0.20, 0.25],
    "B1_BASE_AMOUNTS": {0.15: 500, 0.20: 750, 0.25: 1000},
    "MAX_NORM_CAP": 2250.0,
    "HEAVY_BASE": 2000.0,
    "MAX_HEAVY_PER_CYCLE": 3,
    "LOW_MARKER_DRAWDOWN_THRESH": 0.30,
    "LOW_MARKER_EMA_MULT": 0.85,
    "LOW_MARKER_VOL_MULT": 1.30,
    "LOW_MARKER_ATR_MULT": 1.30,
    "LOW_MARKER_CLV_THRESH": 0.40,
    "LOW_MARKER_RSI_THRESH": 40,
    "TREND_FILTER_DEEP_CRASH": 0.40,
    "TREND_FILTER_DIP_25": 0.25,
    "TREND_FILTER_DIP_15_TO_20": 0.15,
    "TREND_FILTER_EMA_EXTENDED": 1.30,
    "TP1_MULT": 1.20,
    "TP2_MULT": 1.40,
    "TP3_MULT": 1.60,
    "EMA_PERIOD": 200
}

INDIA_PARAMETERS = {
    "name": "INDIA",
    "B1_THRESHOLDS": [0.05, 0.075, 0.10],
    "B1_BASE_AMOUNTS": {0.05: 25000, 0.075: 50000, 0.10: 75000},
    "MAX_NORM_CAP": 100000.0,
    "HEAVY_BASE": 150000.0,
    "MAX_HEAVY_PER_CYCLE": 3,
    "LOW_MARKER_DRAWDOWN_THRESH": 0.15,
    "LOW_MARKER_EMA_MULT": 0.93,
    "LOW_MARKER_VOL_MULT": 1.15,
    "LOW_MARKER_ATR_MULT": 1.15,
    "LOW_MARKER_CLV_THRESH": 0.35,
    "LOW_MARKER_RSI_THRESH": 45,
    "TREND_FILTER_DEEP_CRASH": 0.25,
    "TREND_FILTER_DIP_25": 0.20,
    "TREND_FILTER_DIP_15_TO_20": 0.07,
    "TREND_FILTER_EMA_EXTENDED": 1.10,
    "TP1_MULT": 1.12,
    "TP2_MULT": 1.20,
    "TP3_MULT": 1.30,
    "EMA_PERIOD": 150
}

def resolve_parameters(ticker: str):
    return DEFAULT_PARAMETERS


def select_b1_base_amount(drawdown, thresholds, base_amounts):
    eligible = [thr for thr in thresholds if drawdown >= thr - 1e-9]
    if not eligible:
        return 0.0
    chosen = max(eligible)
    if chosen in base_amounts:
        return base_amounts[chosen]
    fallback_key = min(base_amounts.keys())
    return base_amounts[fallback_key]


# ---------------------------------------------------------------------------
# CHECKPOINT HELPERS
# ---------------------------------------------------------------------------

def load_checkpoint():
    if not (ENABLE_CHECKPOINTING and RESUME_FROM_CHECKPOINT):
        return None
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        print(f"{COLOR_RED}[CHECKPOINT]{COLOR_RESET} Failed to load checkpoint: {exc}")
        return None


def save_checkpoint(state):
    if not ENABLE_CHECKPOINTING:
        return
    try:
        with open(CHECKPOINT_FILE, "wb") as f:
            pickle.dump(state, f)
        print(f"{COLOR_GREEN}[CHECKPOINT]{COLOR_RESET} State saved to {CHECKPOINT_FILE}")
    except Exception as exc:
        print(f"{COLOR_RED}[CHECKPOINT]{COLOR_RESET} Failed to save checkpoint: {exc}")


# Numerical tolerances
EPSILON_SHARES = 1e-12
EPSILON_PRINCIPAL = 1e-9
EPSILON_XIRR = 1e-12



# ---------------------------------------------------------------------------
# 1. DATA LOADING HELPERS
# ---------------------------------------------------------------------------

def load_close(ticker: str, start="2015-01-01") -> pd.Series:
    """
    Load split-adjusted Close prices.
    Tries yf.download() first; if multi-index or empty, falls back to ticker.history().
    Ensures timezone-agnostic datetime index.
    """
    df = yf.download(
        ticker,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="column"
    )

    # Fix multi-index columns (yfinance sometimes returns OHLC as multiindex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # Fallback if Close missing
    if "Close" not in df.columns or df.empty:
        hist = yf.Ticker(ticker).history(start=start, auto_adjust=True)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(-1)
        close = hist["Close"]
    else:
        close = df["Close"]

    close = close.astype(float).dropna()

    # Remove timezone (ensure naive timestamps)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)

    return close


def load_ohlcv(ticker: str, start="2015-01-01") -> pd.DataFrame:
    """
    Load OHLCV (Open, High, Low, Close, Volume) with auto-adjust.
    Used for technical indicators.
    """
    needed = ["Open", "High", "Low", "Close", "Volume"]

    df = yf.download(
        ticker,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="column"
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # If incomplete, fallback to ticker.history()
    if df.empty or not set(needed).issubset(df.columns):
        df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

    df = df[needed].astype(float)

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    return df


# ---------------------------------------------------------------------------
# 2. RSI CALCULATION
# ---------------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    Standard RSI using Wilder's smoothing.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ---------------------------------------------------------------------------
# 3. B1 SIGNAL DETECTION (15%, 20%, 25% from ATH)
# ---------------------------------------------------------------------------

def detect_bottoms_b1(close: pd.Series, thresholds) -> list:
    """
    Detect ladder-buy events.
    thresholds = e.g. [0.15, 0.20, 0.25]
    Returns list of tuples: (date, price, ath_before_correction)
    """
    ath = -np.inf
    hit = {}
    signals = []

    for date, price in close.items():
        price = float(price)

        # Update ATH
        if price > ath:
            ath = price
            hit = {thr: False for thr in thresholds}
            continue

        if ath <= 0:
            continue

        # Drawdown from ATH
        dd = (ath - price) / ath

        # Fire signals in sequence
        for thr in thresholds:
            if not hit[thr] and dd >= thr:
                signals.append((date, price, ath))
                hit[thr] = True

    return signals


# ---------------------------------------------------------------------------
# 4. GOLD VAULT (PRINCIPAL-ONLY LIFO WITH DATE VALIDATION)
# ---------------------------------------------------------------------------

class GoldLot:
    """
    Represents a single gold deposit lot.
    - date: deposit date
    - sh: shares held
    - pr: principal remaining (cannot use profit)
    """
    def __init__(self, date, shares, principal):
        self.date = pd.Timestamp(date).to_pydatetime().replace(tzinfo=None)
        self.shares = float(shares)
        self.principal_remaining = float(principal)


class GoldVault:
    """
    LIFO principal-only withdrawal.
    - Withdraw only up to principal remaining.
    - Cannot use profit.
    - Must not withdraw from future-dated lots.
    """
    def __init__(self, gold_prices: pd.Series):
        if getattr(gold_prices.index, "tz", None) is not None:
            gold_prices = gold_prices.tz_localize(None)
        self.prices = gold_prices.dropna()
        self.lots = []
        self.total_principal = 0.0

    # --- Helper: Price as of specific date -------------------------------
    def price_asof(self, date):
        date = pd.Timestamp(date)
        s = self.prices[self.prices.index <= date]
        if s.empty:
            return float(self.prices.iloc[0])
        return float(s.iloc[-1])

    # --- Deposit ----------------------------------------------------------
    def deposit(self, date, amount):
        price = self.price_asof(date)
        shares = amount / price
        lot = GoldLot(date, shares, amount)
        self.lots.append(lot)
        self.total_principal += amount

        print(f"{COLOR_YELLOW}[GOLD_DEPOSIT]{COLOR_RESET} {date} +${amount:.2f} @ {price:.2f}")

    # --- LIFO withdraw (principal only) ----------------------------------
    def withdraw_principal(self, date, amount_needed):
        if amount_needed <= 0:
            return 0.0

        price = self.price_asof(date)
        remaining = amount_needed
        withdrawn = 0.0

        # LIFO: newest lots first
        for lot in sorted(self.lots, key=lambda x: x.date, reverse=True):
            if lot.date > pd.Timestamp(date):
                continue

            if remaining <= 0:
                break
            if lot.shares <= 0 or lot.principal_remaining <= 0:
                continue

            lot_value = lot.shares * price
            max_cash_from_lot = min(lot_value, lot.principal_remaining)

            if max_cash_from_lot <= 0:
                continue

            take = min(remaining, max_cash_from_lot)
            shares_to_sell = take / price
            shares_to_sell = min(shares_to_sell, lot.shares)
            take = shares_to_sell * price

            lot.shares -= shares_to_sell
            lot.principal_remaining -= take
            remaining -= take
            withdrawn += take

        # Clean empty lots
        self.lots = [
            x for x in self.lots
            if x.shares > EPSILON_PRINCIPAL or x.principal_remaining > EPSILON_PRINCIPAL
        ]

        if withdrawn > 0:
            print(f"{COLOR_YELLOW}[GOLD_WITHDRAW]{COLOR_RESET} {date} -${withdrawn:.2f} @ {price:.2f}")

        return withdrawn

    # --- Current gold vault market value ---------------------------------
    def current_value(self):
        if self.prices.empty:
            return 0
        last = float(self.prices.iloc[-1])
        total_shares = sum(l.shares for l in self.lots)
        return total_shares * last

    # --- Principal remaining ---------------------------------------------
    def principal_left(self):
        return sum(l.principal_remaining for l in self.lots)


# ---------------------------------------------------------------------------
# 5. PORTFOLIO ENGINE (NORMAL BUYS + HEAVY BUYS + STATE-MACHINE TPs)
# ---------------------------------------------------------------------------

class Cycle:
    """
    Represents a buy cycle.
    - buy_price: price paid
    - shares: number of shares currently held
    - heavy: True for heavy low-marker buys (never sold)
    """
    def __init__(self, buy_price, shares, heavy=False):
        self.buy_price = float(buy_price)
        self.shares = float(shares)
        self.heavy = heavy


class TPStateMachine:
    """
    State machine:
        WAIT_FOR_ATH → TP1 → TP2 → TP3 → LOCKED (no more sells)
    Reset only when NEW ATH occurs.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = "WAIT_FOR_ATH"  # waiting for a new ATH
        self.ath_level = None  # float ATH value
        # reset TP stage flags
        if hasattr(self, 'parent'):
            self.parent.tp1_done = False
            self.parent.tp2_done = False
            self.parent.tp3_done = False

    def set_ath(self, ath):
        self.ath_level = float(ath)
        self.phase = "TP1"  # enable TP1

    def next_phase(self):
        if self.phase == "TP1":
            self.phase = "TP2"
        elif self.phase == "TP2":
            self.phase = "TP3"
        elif self.phase == "TP3":
            self.phase = "LOCKED"

    def can_tp(self):
        return self.phase in ("TP1", "TP2", "TP3")

    def required_multiplier(self):
        params = getattr(getattr(self, "parent", None), "params", {})
        if self.phase == "TP1":
            return params.get("TP1_MULT")
        if self.phase == "TP2":
            return params.get("TP2_MULT")
        if self.phase == "TP3":
            return params.get("TP3_MULT")
        return None


class Portfolio:
    """
    - Holds cycles (normal + heavy)
    - Executes buys, sells
    - Tracks low-marker pool
    - Tracks full trade log
    - Uses TP state machine per ticker
    """
    def __init__(self, ticker, params):
        self.cycles = []
        self.trade_log = []
        self.ticker = ticker
        self.params = params
        self.heavy_counts = {}

        self.invested = 0.0
        self.profit_booked = 0.0
        self.low_marker_pool = 0.0
        self.gold_used = 0.0
        self.tp_used = 0.0
        self.new_money_used = 0.0

        # TP flags
        self.tp1_done = False
        self.tp2_done = False
        self.tp3_done = False

        # state machine
        self.tp = TPStateMachine()
        self.tp.parent = self

    # ---------------------------------------------
    # Normal buy (never uses low-marker pool)
    # ---------------------------------------------
    def buy(self, date, price, amount, max_cap):
        if self.invested + amount > max_cap:
            amount = max(0, max_cap - self.invested)

        if amount <= 0:
            return 0.0

        shares = amount / price
        self.cycles.append(Cycle(price, shares, heavy=False))
        self.invested += amount

        self.trade_log.append({
            "type": "BUY",
            "date": date,
            "price": float(price),
            "amount": float(amount),
            "shares": float(shares),
            "shares_sold": 0.0,
            "realized": 0.0,
            "profit": 0.0,
            "reason": "NORMAL_BUY"
        })

        self.trade_log[-1]["from_gold"] = 0.0
        self.trade_log[-1]["from_tp_pool"] = 0.0
        self.trade_log[-1]["from_new_money"] = float(amount)

        return amount

    # ---------------------------------------------
    # Heavy buy (low-marker heavy buy)
    # ---------------------------------------------
    def heavy_buy(self, date, price, base_amount):
        # Fixed order size: do NOT add low_marker_pool here
        amount = base_amount

        if amount <= 0:
            return 0.0

        shares = amount / price
        self.cycles.append(Cycle(price, shares, heavy=True))
        self.invested += amount

        self.trade_log.append({
            "type": "HEAVY_BUY",
            "date": date,
            "price": float(price),
            "amount": float(amount),
            "shares": float(shares),
            "shares_sold": 0.0,
            "realized": 0.0,
            "profit": 0.0,
            "reason": "HEAVY_LOW_MARKER"
        })

        # funding split will be set after we know gold_used / tp_used / new_money
        self.trade_log[-1]["from_gold"] = 0.0
        self.trade_log[-1]["from_tp_pool"] = 0.0
        self.trade_log[-1]["from_new_money"] = float(amount)

        return amount

    # ---------------------------------------------
    # Sell fraction (only from normal cycles)
    # ---------------------------------------------
    def sell_fraction(self, date, price, fraction, reason):
        normal_shares = sum(c.shares for c in self.cycles if not c.heavy)
        if normal_shares <= 0:
            return 0.0

        target = normal_shares * fraction
        remaining = target
        realized_total = 0.0
        profit_total = 0.0

        for c in self.cycles:
            if c.heavy or remaining <= 0:
                continue

            sell_here = min(c.shares, remaining)
            realized = sell_here * price
            cost = sell_here * c.buy_price
            profit = realized - cost

            c.shares -= sell_here
            remaining -= sell_here

            realized_total += realized
            profit_total += profit

        # update pools
        self.low_marker_pool += profit_total
        self.profit_booked += profit_total

        self.trade_log.append({
            "type": "SELL",
            "date": date,
            "price": float(price),
            "amount": 0.0,
            "shares": 0.0,
            "shares_sold": float(target - remaining),
            "realized": float(realized_total),
            "profit": float(profit_total),
            "reason": reason
        })

        # remove empty cycles
        self.cycles = [c for c in self.cycles if c.shares > EPSILON_SHARES]

        return realized_total

    # ---------------------------------------------
    # Current portfolio value
    # ---------------------------------------------
    def current_value(self, price):
        return sum(c.shares * price for c in self.cycles)


# ---------------------------------------------------------------------------
# XIRR CALCULATION
# ---------------------------------------------------------------------------

def compute_xirr(cashflows, guess=0.10):
    """
    cashflows = list of tuples [(date, amount), ...]
        amount < 0 → investment
        amount > 0 → withdrawal / sell
    guess = initial rate guess

    Returns a decimal XIRR value (e.g. 0.15 = 15%).
    """

    # Convert to numpy-friendly arrays
    dates = np.array([pd.Timestamp(d) for d, _ in cashflows])
    amounts = np.array([a for _, a in cashflows], dtype=float)

    # Convert dates to year fractions
    t0 = dates[0]
    years = np.array([(d - t0).days / 365.0 for d in dates], dtype=float)

    # Define NPV function for rate r
    def npv(r):
        return np.sum(amounts / (1 + r)**years)

    # Try Newton’s method
    r = guess
    for _ in range(200):
        # Derivative of NPV wrt r
        d_npv = np.sum(-years * amounts / (1 + r)**(years + 1))

        if abs(d_npv) < EPSILON_XIRR:  # prevent divide-by-zero
            break

        new_r = r - npv(r) / d_npv

        if abs(new_r - r) < EPSILON_XIRR:
            return new_r

        r = new_r

    # Fallback: try simple bisection between -99% to +200%
    lo, hi = -0.99, 2.00
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(mid) > 0:
            lo = mid
        else:
            hi = mid
    return mid


# ---------------------------------------------------------------------------
# 6. ATH EVENT FILTERING AND STATE MACHINE RESET
# ---------------------------------------------------------------------------

def detect_ath_events(close: pd.Series, signals):
    """
    Correct ATH detection:
    - extract ath_pre from each B1 signal
    - dedupe by month
    """

    # Extract ATH values and their correct dates
    ath_list = []
    for (dt, price, ath_pre) in signals:
        # find the actual ATH date for ath_pre (peak before dip)
        ath_date = close[close == ath_pre].index.min()
        if pd.notna(ath_date):
            ath_list.append((ath_date, ath_pre))

    # De-duplicate by month
    monthly = {}
    for d, px in ath_list:
        ym = (d.year, d.month)
        if ym not in monthly:
            monthly[ym] = (d, px)

    # Build event table
    result = []
    for ym, (d, px) in monthly.items():
        result.append({
            "type": "ATH_EVENT",
            "date": d,
            "price": float(px),
            "amount": 0.0,
            "shares": 0.0,
            "shares_sold": 0.0,
            "realized": 0.0,
            "profit": 0.0,
            "reason": "NEW_ATH"
        })

    return result


# ---------------------------------------------------------------------------
# 7. RUN STRATEGY FOR ONE TICKER
# ---------------------------------------------------------------------------

def run_ticker_strategy(ticker, gold_vault, params, start_date=STRATEGY_START_DATE, existing_state=None):
    """
    Runs full strategy for a single ticker.
    Includes:
    - B1 ladder buys
    - ATR heavy buys
    - ATH event detection
    - TP state machine
    - trade log assembly
    """

    close = load_close(ticker, HISTORICAL_START_DATE)
    if close.empty:
        print(f"{COLOR_RED}[SKIP]{COLOR_RESET} {ticker}: no price data available.")
        return None

    ohlcv = load_ohlcv(ticker, HISTORICAL_START_DATE).reindex(close.index)
    if ohlcv.empty:
        print(f"{COLOR_RED}[SKIP]{COLOR_RESET} {ticker}: no OHLCV data available.")
        return None

    start_ts = pd.Timestamp(start_date)
    b1_thresholds = params["B1_THRESHOLDS"]
    b1_base_amounts = params["B1_BASE_AMOUNTS"]
    max_norm_cap = params["MAX_NORM_CAP"]
    heavy_base = params["HEAVY_BASE"]
    max_heavy_per_cycle = params["MAX_HEAVY_PER_CYCLE"]
    trend_deep_crash = params["TREND_FILTER_DEEP_CRASH"]
    trend_dip_25 = params["TREND_FILTER_DIP_25"]
    trend_dip_mid = params["TREND_FILTER_DIP_15_TO_20"]
    trend_ema_ext = params["TREND_FILTER_EMA_EXTENDED"]
    tp1_mult = params["TP1_MULT"]
    tp2_mult = params["TP2_MULT"]
    tp3_mult = params["TP3_MULT"]
    ema_period = params["EMA_PERIOD"]
    low_marker_drawdown = params["LOW_MARKER_DRAWDOWN_THRESH"]
    low_marker_ema = params["LOW_MARKER_EMA_MULT"]
    low_marker_vol = params["LOW_MARKER_VOL_MULT"]
    low_marker_atr = params["LOW_MARKER_ATR_MULT"]
    low_marker_clv = params["LOW_MARKER_CLV_THRESH"]
    low_marker_rsi = params["LOW_MARKER_RSI_THRESH"]
    resume_from = start_ts - pd.Timedelta(days=1)
    if existing_state and existing_state.get("last_processed"):
        resume_from = pd.Timestamp(existing_state["last_processed"])
    if existing_state and existing_state.get("portfolio"):
        p = existing_state["portfolio"]
        p.params = params
        if not hasattr(p, "heavy_counts"):
            p.heavy_counts = {}
        fresh_portfolio = False
    else:
        p = Portfolio(ticker, params)
        fresh_portfolio = True

    high = ohlcv["High"]
    low = ohlcv["Low"]
    volume = ohlcv["Volume"]

    # Build ATH-cycle IDs (increment whenever a new all-time high is set)
    cycle_by_date = {}
    cycle_id = 0
    ath_running = -np.inf
    for dt, px in close.items():
        if px > ath_running:
            ath_running = px
            cycle_id += 1
        cycle_by_date[dt] = cycle_id

    # EMA (SMA used here exactly like original)
    ema200 = close.rolling(ema_period).mean()
    rsi14 = compute_rsi(close)

    # LOG ATR ---------------------------------------------
    log_close = np.log(close)
    log_prev = log_close.shift(1)
    log_high = np.log(high)
    log_low = np.log(low)

    lr1 = (log_high - log_low).abs()
    lr2 = (log_high - log_prev).abs()
    lr3 = (log_low - log_prev).abs()
    log_tr = pd.concat([lr1, lr2, lr3], axis=1).max(axis=1)

    vol60 = volume.rolling(VOL_PERIOD).mean()
    atr60 = log_tr.rolling(ATR_PERIOD).mean()

    # DRAWDOWN & CLV ----------------------------------------
    ath_series = close.cummax()
    drawdown = (ath_series - close) / ath_series
    day_range = (high - low).replace(0, np.nan)
    clv = (close - low) / day_range

    # LOW-MARKER CONDITION ----------------------------------
    condA = drawdown >= low_marker_drawdown
    condB = close <= ema200 * low_marker_ema
    condC = volume >= vol60 * low_marker_vol
    condD = log_tr >= atr60 * low_marker_atr
    condE = clv >= low_marker_clv
    condF = rsi14 <= low_marker_rsi

    low_marker = condA & condB & condC & condD & condE & condF
    marker_dates = [
        d for d in low_marker[low_marker].index
        if pd.Timestamp(d) > resume_from
    ]

    # B1 SIGNALS --------------------------------------------
    signals = [
        (d, price, ath_pre)
        for (d, price, ath_pre) in detect_bottoms_b1(close, b1_thresholds)
        if pd.Timestamp(d) > resume_from
    ]

    prior_ath_events = list(existing_state.get("ath_events", [])) if existing_state else []
    ath_events = prior_ath_events + detect_ath_events(close, signals)
    tp_cycles = []
    last_tp_cycle_ath = -np.inf

    # -------------------------
    # MAIN STRATEGY LOOP - PROCESS BUYS THEN TPS
    # -------------------------
    
    for (date, price, ath_pre) in signals:
        date = pd.Timestamp(date)

        # Trend filter
        dd = (ath_pre - price) / ath_pre

        if dd >= trend_deep_crash:
            trend_ok = True
        elif dd >= trend_dip_25:
            trend_ok = True
        elif dd >= trend_dip_mid:
            trend_ok = price <= ema200.loc[date] * trend_ema_ext
        else:
            trend_ok = False

        if not trend_ok:
            continue

        # Determine base amount from DD
        base_amt = select_b1_base_amount(dd, b1_thresholds, b1_base_amounts)

        executed = p.buy(date, price, base_amt, max_norm_cap)
        if executed > 0:
            if ath_pre > last_tp_cycle_ath + 1e-9:
                tp_cycles.append((date, ath_pre))
                last_tp_cycle_ath = ath_pre
                print(f"{COLOR_CYAN}[STATE RESET]{COLOR_RESET} {date.date()} new ATH={ath_pre:.2f}")
            # 1) Try gold principal
            gold_used = gold_vault.withdraw_principal(date, executed)

            # 2) Try TP pool next
            tp_available = p.low_marker_pool
            tp_used = min(tp_available, executed - gold_used)
            p.low_marker_pool -= tp_used

            # 3) Remaining is new personal money
            new_money = executed - gold_used - tp_used

            # Update portfolio source totals
            p.gold_used += gold_used
            p.tp_used += tp_used
            p.new_money_used += new_money

            # Update this BUY log row
            p.trade_log[-1]["from_gold"] = float(gold_used)
            p.trade_log[-1]["from_tp_pool"] = float(tp_used)
            p.trade_log[-1]["from_new_money"] = float(new_money)

            print(f"{COLOR_GREEN}[BUY]{COLOR_RESET} {date.date()} {executed}@{price:.2f} "
                  f"(gold={gold_used:.2f}, tp={tp_used:.2f}, new={new_money:.2f})")


    # -------------------------
    # TAKE PROFIT PROCESSING - STRICT SEQUENTIAL TP1 → TP2 → TP3
    # -------------------------

    buy_dates = [
        pd.Timestamp(row["date"])
        for row in p.trade_log
        if row["type"] == "BUY"
    ]
    tp_start = max(resume_from + pd.Timedelta(days=1), pd.Timestamp(start_date))

    future_dates = close[close.index >= tp_start].index
    if fresh_portfolio:
        p.tp.reset()
    tp_cycle_idx = -1

    for check_date in future_dates:
        check_price = close.loc[check_date]

        while tp_cycle_idx + 1 < len(tp_cycles) and tp_cycles[tp_cycle_idx + 1][0] <= check_date:
            tp_cycle_idx += 1
            _, cycle_ath = tp_cycles[tp_cycle_idx]
            p.tp.reset()
            p.tp.set_ath(cycle_ath)

        if p.tp.ath_level is None:
            continue

        # If no normal shares, skip
        normal_shares = sum(c.shares for c in p.cycles if not c.heavy)
        if normal_shares <= EPSILON_SHARES:
            continue

        # Determine which TP phase to check
        if not p.tp1_done:
            phase = "TP1"
            multiplier = tp1_mult
        elif not p.tp2_done:
            phase = "TP2"
            multiplier = tp2_mult
        elif not p.tp3_done:
            phase = "TP3"
            multiplier = tp3_mult
        else:
            # all TPs done → wait for next ATH
            continue

        # Check if TP target is met
        target = p.tp.ath_level * multiplier

        if check_price >= target:
            prof = p.sell_fraction(check_date, check_price, TP_FRAC, f"TP_{phase}")
            print(f"{COLOR_MAGENTA}[TP]{COLOR_RESET} {check_date.date()} sell@{check_price:.2f}, prof={prof:.2f}")

            if phase == "TP1":
                p.tp1_done = True
            elif phase == "TP2":
                p.tp2_done = True
            elif phase == "TP3":
                p.tp3_done = True

            continue

    # -------------------------
    # HEAVY LOW-MARKER BUYS
    # -------------------------
    heavy_counts = getattr(p, "heavy_counts", {})

    for d in marker_dates:
        d = pd.Timestamp(d)
        price = close.loc[d]

        cid = cycle_by_date[d]
        if heavy_counts.get(cid, 0) >= max_heavy_per_cycle:
            # already did 3 heavy buys in this ATH cycle
            continue

        executed = p.heavy_buy(d, price, heavy_base)
        if executed > 0:
            heavy_counts[cid] = heavy_counts.get(cid, 0) + 1

            # 1) Gold principal first
            gold_used = gold_vault.withdraw_principal(d, executed)

            # 2) TP pool
            tp_available = p.low_marker_pool
            tp_used = min(tp_available, executed - gold_used)
            p.low_marker_pool -= tp_used

            # 3) New money
            new_money = executed - gold_used - tp_used

            # Portfolio-level totals
            p.gold_used += gold_used
            p.tp_used += tp_used
            p.new_money_used += new_money

            # Update heavy buy log row (last row)
            p.trade_log[-1]["from_gold"] = float(gold_used)
            p.trade_log[-1]["from_tp_pool"] = float(tp_used)
            p.trade_log[-1]["from_new_money"] = float(new_money)

            print(
                f"{COLOR_BLUE}[HEAVY]{COLOR_RESET} {d.date()} {executed}@{price:.2f} "
                f"(gold={gold_used:.2f}, tp={tp_used:.2f}, new={new_money:.2f})"
            )

    # -------------------------
    # FINAL VALUATION
    # -------------------------
    last_date = close.index[-1]
    p.heavy_counts = heavy_counts
    last_price = float(close.iloc[-1])

    held = p.current_value(last_price)
    final = held + p.profit_booked
    pnl = final - p.invested

    # -------------------------
    # BUILD PER-TICKER EVENT TABLE
    # -------------------------
    events = []
    events.extend(ath_events)
    events.extend(p.trade_log)

    df = pd.DataFrame(events)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

    result = {
        "ticker": ticker,
        "trade_df": df,
        "invested": p.invested,
        "profit": p.profit_booked,
        "held": held,
        "final_value": final,
        "pnl": pnl,
        "gold_used": p.gold_used,
        "tp_used": p.tp_used,
        "new_money_used": p.new_money_used,
        "last_date": last_date,
        "portfolio": p,
        "ath_events": ath_events
    }

    return result


# ---------------------------------------------------------------------------
# 8. BUILD CLEAN TABULAR REPORT
# ---------------------------------------------------------------------------

def build_ticker_report(res):
    """
    Converts 1 ticker’s result into a formatted table for printing.
    """
    t = res["ticker"]
    df = res["trade_df"]

    print("\n" + "="*70)
    print(f"{COLOR_WHITE}TICKER REPORT — {t}{COLOR_RESET}")
    print("="*70)

    if df.empty:
        print("No events.")
        return

    # Apply colors to TYPE column
    def colortype(x):
        if x == "BUY":
            return COLOR_GREEN + x + COLOR_RESET
        if x == "SELL":
            return COLOR_RED + x + COLOR_RESET
        if x == "HEAVY_BUY":
            return COLOR_BLUE + x + COLOR_RESET
        if x == "ATH_EVENT":
            return COLOR_YELLOW + x + COLOR_RESET
        return x

    dfp = df.copy()
    dfp["type"] = dfp["type"].astype(str).apply(colortype)

    # Tidy display columns
    show = ["type", "date", "price", "amount", "shares",
            "shares_sold", "realized", "profit", "reason",
            "from_gold", "from_tp_pool", "from_new_money"]

    for col in show:
        if col not in dfp.columns:
            dfp[col] = ""

    print(dfp[show].to_string(index=False))


# ---------------------------------------------------------------------------
# 9. GLOBAL PORTFOLIO SUMMARY (ALL TICKERS)
# ---------------------------------------------------------------------------

def build_global_summary(all_results):
    """
    Builds:
    - Per-ticker table
    - Combined totals
    - Combined XIRR
    """
    rows = []
    for r in all_results:
        rows.append({
            "ticker": r["ticker"],
            "invested": r["invested"],
            "profit_booked": r["profit"],
            "held_value": r["held"],
            "final_value": r["final_value"],
            "pnl": r["pnl"],
            "gold_used": r["gold_used"],
            "tp_used": r["tp_used"],
            "new_money_used": r["new_money_used"]
        })

    df = pd.DataFrame(rows)

    total_inv = df["invested"].sum()
    total_profit = df["profit_booked"].sum()
    total_held = df["held_value"].sum()
    final_value = total_profit + total_held
    pnl = final_value - total_inv

    # Build combined cashflows for XIRR ---------------------
    combined_cf = []
    for r in all_results:
        tdf = r["trade_df"]
        for _, row in tdf.iterrows():
            if row["type"] in ("BUY", "HEAVY_BUY"):
                combined_cf.append((row["date"], -row["amount"]))
            elif row["type"] == "SELL":
                combined_cf.append((row["date"], row["realized"]))

    # Final terminal cashflow
    max_date = max(r["last_date"] for r in all_results)
    combined_cf.append((pd.Timestamp(max_date), total_held))

    xirr_dec = compute_xirr(combined_cf)
    xirr_pct = xirr_dec * 100 if not np.isnan(xirr_dec) else np.nan

    print("\n" + "="*80)
    print(f"{COLOR_WHITE}GLOBAL PORTFOLIO SUMMARY — ALL TICKERS{COLOR_RESET}")
    print("="*80)
    print(df.to_string(index=False))
    print("-"*80)
    print(f"Total invested : {total_inv:,.2f}")
    print(f"Total profit booked : {total_profit:,.2f}")
    print(f"Total held value : {total_held:,.2f}")
    print(f"Total final value : {final_value:,.2f}")
    print(f"Total PnL : {pnl:,.2f}")
    print(f"Portfolio XIRR : {xirr_pct:,.2f}%")
    print("="*80)

    print(f"Total gold used : {df['gold_used'].sum():,.2f}")
    print(f"Total TP profits reused : {df['tp_used'].sum():,.2f}")
    print(f"Total new money invested : {df['new_money_used'].sum():,.2f}")

    return df, xirr_pct


# ---------------------------------------------------------------------------
# 10. GOLD VAULT SUMMARY (PRINCIPAL-ONLY, LIFO WITH DATE FILTER)
# ---------------------------------------------------------------------------

def gold_summary(gold_vault):
    total_principal = gold_vault.total_principal
    principal_left = gold_vault.principal_left()
    value = gold_vault.current_value()
    profit = value - principal_left

    print("\n" + "="*80)
    print(f"{COLOR_YELLOW}GOLD VAULT SUMMARY — PRINCIPAL-ONLY (LIFO){COLOR_RESET}")
    print("="*80)
    print(f"Gold ticker : {GOLD_TICKER}")
    print(f"Total principal deposited : {total_principal:,.2f}")
    print(f"Principal still spendable (LIFO): {principal_left:,.2f}")
    print(f"Gold current market value : {value:,.2f}")
    print(f"Gold profit (value - principal) : {profit:,.2f}")
    print("="*80)


# ---------------------------------------------------------------------------
# 11. CSV / EXCEL EXPORTS
# ---------------------------------------------------------------------------

def export_results(all_results, gold_vault, run_dir):
    """
    Saves per-ticker and combined reports.
    """
    os.makedirs(run_dir, exist_ok=True)

    # Ticker summaries
    rows = []
    trade_rows = []

    for r in all_results:
        rows.append({
            "ticker": r["ticker"],
            "invested": r["invested"],
            "profit_booked": r["profit"],
            "held_value": r["held"],
            "final_value": r["final_value"],
            "pnl": r["pnl"],
            "gold_used": r["gold_used"],
            "tp_used": r["tp_used"],
            "new_money_used": r["new_money_used"]
        })

        df = r["trade_df"].copy()
        df["ticker"] = r["ticker"]
        trade_rows.append(df)

    summary_df = pd.DataFrame(rows)
    trades_df = pd.concat(trade_rows, ignore_index=True)

    summary_path = os.path.join(run_dir, "summary.csv")
    trades_path = os.path.join(run_dir, "trades.csv")
    summary_xlsx_path = os.path.join(run_dir, "summary.xlsx")
    trades_xlsx_path = os.path.join(run_dir, "trades.xlsx")
    summary_df.to_csv(summary_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    summary_df.to_excel(summary_xlsx_path, index=False)
    trades_df.to_excel(trades_xlsx_path, index=False)

    state_df = pd.DataFrame([{
        "gold_total_principal": gold_vault.total_principal,
        "gold_principal_left": gold_vault.principal_left(),
        "gold_value": gold_vault.current_value(),
        "gold_profit": gold_vault.current_value() - gold_vault.principal_left()
    }])
    state_csv = os.path.join(run_dir, "state_snapshot.csv")
    state_xlsx = os.path.join(run_dir, "state_snapshot.xlsx")
    state_df.to_csv(state_csv, index=False)
    state_df.to_excel(state_xlsx, index=False)

    print(f"{COLOR_GREEN}CSV and Excel exported: summary.*, trades.*{COLOR_RESET}")


# ---------------------------------------------------------------------------
# 12. MAIN EXECUTION ENTRY
# ---------------------------------------------------------------------------

def main():
    print(f"{COLOR_YELLOW}Setting up GOLD VAULT (LIFO PRINCIPAL ONLY)...{COLOR_RESET}")
    gold_close = load_close(GOLD_TICKER, STRATEGY_START_DATE)
    checkpoint = load_checkpoint()
    if checkpoint:
        gold_vault = checkpoint.get("gold_vault", GoldVault(gold_close))
        gold_vault.prices = gold_close
        ticker_states = checkpoint.get("tickers", {})
        current = pd.Timestamp(checkpoint.get("gold_next_deposit", STRATEGY_START_DATE))
    else:
        gold_vault = GoldVault(gold_close)
        ticker_states = {}
        current = pd.Timestamp(STRATEGY_START_DATE)

    # Monthly deposits -------------------------------------
    last_gold = gold_close.index[-1]

    while current <= last_gold:
        elig = gold_close[gold_close.index >= current]
        if not elig.empty:
            dep_date = elig.index[0]
            gold_vault.deposit(dep_date, MONTHLY_GOLD_DEPOSIT)

        y = current.year + (1 if current.month == 12 else 0)
        m = 1 if current.month == 12 else current.month + 1
        current = pd.Timestamp(year=y, month=m, day=1)
    next_gold_deposit = current

    os.makedirs(LOG_DIR, exist_ok=True)
    run_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(LOG_DIR, f"run_{run_stamp}")

    print(f"{COLOR_WHITE}\nRUNNING FULL STRATEGY...\n{COLOR_RESET}")

    all_results = []
    new_ticker_states = {}
    processed_tickers = set()
    for t in TICKERS:
        params = resolve_parameters(t)
        existing = ticker_states.get(t) if ticker_states else None
        print(f"{COLOR_CYAN}\nProcessing {t}...{COLOR_RESET}")
        res = run_ticker_strategy(t, gold_vault, params, STRATEGY_START_DATE, existing_state=existing)
        if res is None:
            continue
        all_results.append(res)
        processed_tickers.add(t)
        snapshot = {k: v for k, v in res.items() if k != "portfolio"}
        new_ticker_states[t] = {
            "portfolio": res["portfolio"],
            "last_processed": res["last_date"],
            "ath_events": res["ath_events"],
            "snapshot": snapshot
        }
        build_ticker_report(res)

    # Include previously held tickers even if not processed this run
    for t, state in (ticker_states or {}).items():
        if t in processed_tickers:
            continue
        snap = state.get("snapshot")
        if snap:
            all_results.append(snap)
        if t not in new_ticker_states:
            new_ticker_states[t] = state

    # GLOBAL summaries
    build_global_summary(all_results)
    gold_summary(gold_vault)

    # Exports
    export_results(all_results, gold_vault, run_dir)

    checkpoint_state = {
        "gold_vault": gold_vault,
        "gold_next_deposit": next_gold_deposit,
        "tickers": new_ticker_states,
        "last_run": pd.Timestamp.utcnow()
    }
    save_checkpoint(checkpoint_state)

    print(f"{COLOR_GREEN}\nALL DONE — FULL STRATEGY COMPLETED.{COLOR_RESET}")


# ---------------------------------------------------------------------------
# 13. PRINT FULL STRATEGY SUMMARY AT END OF SCRIPT
# ---------------------------------------------------------------------------

def print_strategy_summary():
    print("\n" + "="*100)
    print(f"{COLOR_WHITE}STRATEGY SUMMARY — FULL LOGIC OVERVIEW{COLOR_RESET}")
    print("="*100)

    print(f"""\
ENTRY LOGIC (B1 LADDER)
-----------------------
1. Track ATH (All-Time-High) for each ticker.
2. When price drops below ATH by:
       - 15% → BUY 250 USD
       - 20% → BUY 500 USD
       - 25% → BUY 750 USD
3. Buy only if price is ABOVE the 200-day EMA (trend filter).
4. Normal buy cap: max 3,000 USD invested per ticker.
5. All normal BUYs are funded from GOLD VAULT principal (if available).

ATH EVENT & STATE MACHINE
-------------------------
1. Only 1 ATH_EVENT per month is printed (clean output).
2. When a new ATH is detected:
       → RESET TP STATE MACHINE
       → TP phase resets to TP1.
       → Tracks ATH 'ath_pre' for TP target computation.

TAKE PROFIT STATE MACHINE (TP1 → TP2 → TP3 → LOCKED)
-----------------------------------------------------
1. TP1 triggers at price ≥ ATH * 1.20.
2. TP2 triggers at price ≥ ATH * 1.40.
3. TP3 triggers at price ≥ ATH * 1.60.
4. Each TP sells 15% of ALL normal shares (heavy cycles ignored).
5. After TP3 → No more sells until a NEW ATH resets state.
6. Profits from sells accumulate in LOW_MARKER_POOL (not reused for normal buys).

HEAVY LOW-MARKER BUY LOGIC
---------------------------
Triggered when ALL conditions are met:
    • Drawdown ≥ 30%
    • Price ≤ EMA200 * 0.85
    • Volume ≥ 1.3 × 60-day volume average
    • logATR ≥ 1.3 × 60-day logATR average
    • CLV ≥ 0.40
    • RSI(14) ≤ 40

If triggered:
    • BUY (1000 USD + accumulated low_marker_pool)
    • This is a HEAVY BUY:
           - Never sold.
           - NOT restricted by 3,000 USD cap.
    • Funded entirely by GOLD principal if available.

GOLD VAULT (LIFO PRINCIPAL-ONLY WITH DATE VALIDATION)
-------------------------------------------------------
• Monthly deposit: 500 USD into GLD.
• Stored into lots with (date, shares, principal_remaining).
• For any BUY (normal or heavy):
      → Withdraw principal using LIFO order.
      → Only lots with date ≤ buy-date are eligible.
      → Only withdraw UP TO principal_remaining.
      → PROFIT in gold is never touched.
• Gold value grows independently and is shown in final report.

PORTFOLIO VALUATION (PER TICKER)
---------------------------------
For each ticker we report:
    invested — total BUY amounts
    profit_booked — realized gains from TP sells
    held_value — final value of remaining shares
    final_value — held_value + profit_booked
    pnl — final_value - invested

REPORTING OUTPUT
----------------
The script prints:
    ✓ Per-ticker trade table with colors
    ✓ ATH_EVENT markers
    ✓ BUY, SELL, HEAVY_BUY rows
    ✓ Realized profit for each sell
    ✓ Per-ticker summary
    ✓ Global portfolio summary
    ✓ Gold vault summary
    ✓ CSV + Excel exports for:
            • summary.csv/summary.xlsx
            • trades.csv/trades.xlsx

This summary provides a complete overview of the strategy behavior.

""")

    print("="*100)
    print(f"{COLOR_GREEN}END OF STRATEGY SUMMARY{COLOR_RESET}")
    print("="*100 + "\n")


# ---------------------------------------------------------------------------
# 14. RUN MAIN WHEN EXECUTED DIRECTLY
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
