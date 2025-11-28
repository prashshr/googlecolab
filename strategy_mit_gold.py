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
MONTHLY_GOLD_DEPOSIT = 50000.0

# Strategy start date
STRATEGY_START_DATE = "2016-01-01"

# Data loading start date (historical data)
HISTORICAL_START_DATE = "2014-01-01"

# List of tickers to process
#TICKERS = [
#    "NVDA", "MSFT", "PLTR", "TSLA", "AMZN",
#    "ASML", "GOOG", "META", "AVGO", "AAPL"
#]

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
    "ADANIGREEN.NS", "TATAPOWER.NS", "POLYCAB.NS", "DMART.NS", "TATAELXSI.NS"
]


# B1 ladder thresholds (from ATH)
B1_THRESHOLDS = [0.15, 0.20, 0.25]

# B1 base amounts based on drawdown
B1_BASE_AMOUNTS = {0.25: 100000, 0.20: 50000, 0.15: 25000}

# Maximum normal investment cap per ticker
MAX_NORM_CAP = 200000.0

# Take-profit fraction per phase
TP_FRAC = 0.15

# Heavy buy base amount
HEAVY_BASE = 250000.0

# Maximum heavy buys per ATH cycle
MAX_HEAVY_PER_CYCLE = 3

# Low-marker conditions
LOW_MARKER_DRAWDOWN_THRESH = 0.30
LOW_MARKER_EMA_MULT = 0.85
LOW_MARKER_VOL_MULT = 1.3
LOW_MARKER_ATR_MULT = 1.3
LOW_MARKER_CLV_THRESH = 0.40
LOW_MARKER_RSI_THRESH = 40

# Trend filter thresholds
TREND_FILTER_DEEP_CRASH = 0.40
TREND_FILTER_DIP_25 = 0.25
TREND_FILTER_DIP_15_TO_20 = 0.15
TREND_FILTER_EMA_EXTENDED = 1.30

# TP multipliers
TP1_MULT = 1.20
TP2_MULT = 1.40
TP3_MULT = 1.60

# Technical indicators periods
RSI_PERIOD = 14
EMA_PERIOD = 200
VOL_PERIOD = 60
ATR_PERIOD = 60

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
        if self.phase == "TP1":
            return TP1_MULT
        if self.phase == "TP2":
            return TP2_MULT
        if self.phase == "TP3":
            return TP3_MULT
        return None


class Portfolio:
    """
    - Holds cycles (normal + heavy)
    - Executes buys, sells
    - Tracks low-marker pool
    - Tracks full trade log
    - Uses TP state machine per ticker
    """
    def __init__(self, ticker):
        self.cycles = []
        self.trade_log = []
        self.ticker = ticker

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

def run_ticker_strategy(ticker, gold_vault, start_date=STRATEGY_START_DATE):
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

    # EMA200 (SMA used here exactly like original)
    ema200 = close.rolling(EMA_PERIOD).mean()
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
    condA = drawdown >= LOW_MARKER_DRAWDOWN_THRESH
    condB = close <= ema200 * LOW_MARKER_EMA_MULT
    condC = volume >= vol60 * LOW_MARKER_VOL_MULT
    condD = log_tr >= atr60 * LOW_MARKER_ATR_MULT
    condE = clv >= LOW_MARKER_CLV_THRESH
    condF = rsi14 <= LOW_MARKER_RSI_THRESH

    low_marker = condA & condB & condC & condD & condE & condF
    marker_dates = [
        d for d in low_marker[low_marker].index
        if pd.Timestamp(d) >= start_ts
    ]

    # B1 SIGNALS --------------------------------------------
    signals = [
        (d, price, ath_pre)
        for (d, price, ath_pre) in detect_bottoms_b1(close, B1_THRESHOLDS)
        if pd.Timestamp(d) >= start_ts
    ]

    # PORTFOLIO for this ticker
    p = Portfolio(ticker)

    # Filter ATH events
    ath_events = detect_ath_events(close, signals)
    tp_cycles = []
    last_tp_cycle_ath = -np.inf

    # -------------------------
    # MAIN STRATEGY LOOP - PROCESS BUYS THEN TPS
    # -------------------------
    
    for (date, price, ath_pre) in signals:
        date = pd.Timestamp(date)

        # Trend filter
        dd = (ath_pre - price) / ath_pre

        if dd >= TREND_FILTER_DEEP_CRASH:
            trend_ok = True
        elif dd >= TREND_FILTER_DIP_25:
            trend_ok = True
        elif dd >= TREND_FILTER_DIP_15_TO_20:
            trend_ok = price <= ema200.loc[date] * TREND_FILTER_EMA_EXTENDED
        else:
            trend_ok = False

        if not trend_ok:
            continue

        # Determine base amount from DD
        base_amt = B1_BASE_AMOUNTS.get(dd // 0.05 * 0.05, B1_BASE_AMOUNTS[0.15])

        executed = p.buy(date, price, base_amt, MAX_NORM_CAP)
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
    tp_start = min(buy_dates) if buy_dates else pd.Timestamp(STRATEGY_START_DATE)

    future_dates = close[close.index >= tp_start].index
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
            multiplier = TP1_MULT
        elif not p.tp2_done:
            phase = "TP2"
            multiplier = TP2_MULT
        elif not p.tp3_done:
            phase = "TP3"
            multiplier = TP3_MULT
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
    heavy_counts = {}  # cycle_id -> heavy buys used

    for d in marker_dates:
        d = pd.Timestamp(d)
        price = close.loc[d]

        cid = cycle_by_date[d]
        if heavy_counts.get(cid, 0) >= MAX_HEAVY_PER_CYCLE:
            # already did 3 heavy buys in this ATH cycle
            continue

        executed = p.heavy_buy(d, price, HEAVY_BASE)
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
        "last_date": last_date
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

def export_results(all_results):
    """
    Saves per-ticker and combined reports.
    """

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

    summary_df.to_csv("summary.csv", index=False)
    trades_df.to_csv("trades.csv", index=False)

    summary_df.to_excel("summary.xlsx", index=False)
    trades_df.to_excel("trades.xlsx", index=False)

    print(f"{COLOR_GREEN}CSV and Excel exported: summary.*, trades.*{COLOR_RESET}")


# ---------------------------------------------------------------------------
# 12. MAIN EXECUTION ENTRY
# ---------------------------------------------------------------------------

def main():
    print(f"{COLOR_YELLOW}Setting up GOLD VAULT (LIFO PRINCIPAL ONLY)...{COLOR_RESET}")
    gold_close = load_close(GOLD_TICKER, STRATEGY_START_DATE)
    gold_vault = GoldVault(gold_close)

    # Monthly deposits -------------------------------------
    current = pd.Timestamp(STRATEGY_START_DATE)
    last_gold = gold_close.index[-1]

    while current <= last_gold:
        elig = gold_close[gold_close.index >= current]
        if not elig.empty:
            dep_date = elig.index[0]
            gold_vault.deposit(dep_date, MONTHLY_GOLD_DEPOSIT)

        y = current.year + (1 if current.month == 12 else 0)
        m = 1 if current.month == 12 else current.month + 1
        current = pd.Timestamp(year=y, month=m, day=1)

    print(f"{COLOR_WHITE}\nRUNNING FULL STRATEGY...\n{COLOR_RESET}")

    all_results = []
    for t in TICKERS:
        print(f"{COLOR_CYAN}\nProcessing {t}...{COLOR_RESET}")
        res = run_ticker_strategy(t, gold_vault, STRATEGY_START_DATE)
        if res is None:
            continue
        all_results.append(res)
        build_ticker_report(res)

    # GLOBAL summaries
    build_global_summary(all_results)
    gold_summary(gold_vault)

    # Exports
    export_results(all_results)

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
