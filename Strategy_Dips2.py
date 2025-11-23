import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ==========================================================
# 0. CONFIG – EDITABLE PARAMETERS
# ==========================================================

# Data / universe
START_DATE    = "2021-01-01"
TICKERS       = ["NVDA", "MSFT", "PLTR", "TSLA", "AMZN",
                 "ASML", "GOOG", "META", "AVGO", "AAPL"]

# Entry ladder (normal B1.2)
DD_THRESHOLDS = [0.10, 0.15, 0.20]          # 15%, 20%, 25% from ATH
TP_LEVELS     = [1.20, 1.40, 1.60]          # 1.2x, 1.4x, 1.6x ATH TP levels
TP_FRACS      = [0.15, 0.15, 0.15]          # 15% of normal shares at each TP

# Normal buys
BUY_AMOUNTS   = {                            # base size per ladder level
    0: 250.0,   # for 1st threshold
    1: 250.0,   # for 2nd threshold
    2: 250.0,   # for 3rd threshold
}
MAX_NORMAL_CAP = 3000.0                      # cap for normal buys only

# Heavy low-marker buy
HEAVY_BUY_BASE          = 1000.0             # 1000 USD + accumulated pool
LOW_MARKER_LOOKBACK     = 60                 # days for vol / logTR avg
LOW_MARKER_DD_MIN       = 0.30               # ≥ 30% below ATH
LOW_MARKER_EMA_MULT     = 0.85               # price ≤ EMA200 * 0.85
LOW_MARKER_VOL_MULT     = 1.3                # vol ≥ 1.3 × 60d_avg
LOW_MARKER_ATR_MULT     = 1.3                # logTR ≥ 1.3 × 60d_avg
LOW_MARKER_CLV_MIN      = 0.40               # CLV ≥ 0.40
LOW_MARKER_RSI_MAX      = 40.0               # RSI ≤ 40

# Technicals
EMA_WINDOW   = 200
RSI_PERIOD   = 14

# Printing / colors
ENABLE_COLORS = True

# Columns for the event table (Format A)
EVENT_COLUMNS = [
    "type", "date", "price", "amount",
    "shares", "shares_sold", "realized", "profit", "reason",
]

# ==========================================================
# ANSI colors (simple)
# ==========================================================

if ENABLE_COLORS:
    COLOR_RESET  = "\033[0m"
    COLOR_YELLOW = "\033[93m"
    COLOR_GREEN  = "\033[92m"
    COLOR_RED    = "\033[91m"
    COLOR_CYAN   = "\033[96m"
else:
    COLOR_RESET  = ""
    COLOR_YELLOW = ""
    COLOR_GREEN  = ""
    COLOR_RED    = ""
    COLOR_CYAN   = ""


def color_type(event_type: str) -> str:
    """Color only the event type cell, depending on event."""
    if event_type == "ATH_EVENT":
        return f"{COLOR_YELLOW}{event_type}{COLOR_RESET}"
    if event_type == "BUY":
        return f"{COLOR_GREEN}{event_type}{COLOR_RESET}"
    if event_type == "HEAVY_BUY":
        return f"{COLOR_CYAN}{event_type}{COLOR_RESET}"
    if event_type == "SELL":
        return f"{COLOR_RED}{event_type}{COLOR_RESET}"
    return event_type


# ==========================================================
# 1. Robust loader for Close prices (split-adjusted, tz-naive)
# ==========================================================

def load_close(ticker, start=START_DATE):
    df = yf.download(
        ticker,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="column"
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    if "Close" not in df.columns or df.empty:
        hist = yf.Ticker(ticker).history(start=start, auto_adjust=True)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(-1)
        if "Close" not in hist.columns or hist.empty:
            raise ValueError(f"No usable Close data for {ticker}")
        close = hist["Close"]
    else:
        close = df["Close"]

    close = close.astype(float).dropna()

    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)

    return close


# ==========================================================
# 1b. OHLCV loader for technicals (auto_adjusted, tz-naive)
# ==========================================================

def load_ohlcv(ticker, start=START_DATE):
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

    if df.empty or not set(needed).issubset(df.columns):
        hist = yf.Ticker(ticker).history(start=start, auto_adjust=True)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(-1)
        if hist.empty or not set(needed).issubset(hist.columns):
            missing = [c for c in needed if c not in hist.columns]
            raise ValueError(
                f"Missing columns {missing} for {ticker} from both download() and history()"
            )
        df = hist[needed].astype(float)
    else:
        df = df[needed].astype(float)

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    return df


# ==========================================================
# 1c. RSI helper
# ==========================================================

def compute_rsi(close, period=RSI_PERIOD):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ==========================================================
# 2. B1.2 signal detection (drawdowns from ATH)
# ==========================================================

def detect_bottoms_b1(close, thresholds):
    """
    thresholds = list of drawdown percentages from ATH
    Returns list of (date, price, ath_pre_correction, threshold_index)
    """
    ath = -np.inf
    hit = {}
    signals = []

    for date, price in close.items():
        price = float(price)

        if price > ath:
            ath = price
            hit = {thr: False for thr in thresholds}
            continue

        if ath <= 0:
            continue

        dd = (ath - price) / ath

        for i, thr in enumerate(thresholds):
            if not hit[thr] and dd >= thr:
                signals.append((date, price, ath, i))
                hit[thr] = True

    return signals


# ==========================================================
# 3. Portfolio engine (normal + heavy buys, low-marker pool)
# ==========================================================

class Cycle:
    def __init__(self, buy_price, shares, is_heavy=False):
        self.buy_price = buy_price
        self.shares = shares
        self.is_heavy = is_heavy   # heavy-buy cycles are never sold


class Portfolio:
    def __init__(self):
        self.cycles = []
        self.trade_log = []
        self.profit_booked = 0.0
        self.invested = 0.0
        self.low_marker_pool = 0.0  # profits waiting for heavy buy

    def buy(self, date, price, base_amount, max_cap, reason):
        """Normal buy: base_amount only, respects max_cap."""
        amount = base_amount

        if self.invested + amount > max_cap:
            amount = max(0.0, max_cap - self.invested)

        if amount <= 0:
            return

        shares = amount / price
        self.cycles.append(Cycle(price, shares, is_heavy=False))
        self.invested += amount

        self.trade_log.append({
            "type": "BUY",
            "date": pd.Timestamp(date).to_pydatetime().replace(tzinfo=None),
            "price": float(price),
            "amount": float(amount),
            "shares": float(shares),
            "shares_sold": 0.0,
            "realized": 0.0,
            "profit": 0.0,
            "reason": reason,
        })

    def heavy_buy(self, date, price, base_amount, reason):
        """Heavy low-marker buy: base_amount + low_marker_pool, never sold."""
        amount = base_amount + self.low_marker_pool
        self.low_marker_pool = 0.0

        if amount <= 0:
            return

        shares = amount / price
        self.cycles.append(Cycle(price, shares, is_heavy=True))
        self.invested += amount

        self.trade_log.append({
            "type": "HEAVY_BUY",
            "date": pd.Timestamp(date).to_pydatetime().replace(tzinfo=None),
            "price": float(price),
            "amount": float(amount),
            "shares": float(shares),
            "shares_sold": 0.0,
            "realized": 0.0,
            "profit": 0.0,
            "reason": reason,
        })

    def sell_fraction(self, date, price, fraction, reason):
        """
        Sells 'fraction' of TOTAL normal shares (is_heavy=False),
        pro-rata across normal cycles. Heavy cycles are untouched.
        """
        total_shares = sum(c.shares for c in self.cycles if not c.is_heavy)
        if total_shares <= 0:
            return 0.0

        target = total_shares * fraction
        remaining = target
        realized_total = 0.0
        profit_total = 0.0

        for c in self.cycles:
            if c.is_heavy:
                continue
            if remaining <= 0 or c.shares <= 0:
                continue

            sell_here = min(c.shares, remaining)
            realized = sell_here * price
            cost = sell_here * c.buy_price
            profit = realized - cost

            c.shares -= sell_here
            remaining -= sell_here

            realized_total += realized
            profit_total += profit

        self.low_marker_pool += profit_total
        self.profit_booked += profit_total

        self.trade_log.append({
            "type": "SELL",
            "date": pd.Timestamp(date).to_pydatetime().replace(tzinfo=None),
            "price": float(price),
            "amount": 0.0,
            "shares": 0.0,
            "shares_sold": float(target - remaining),
            "realized": float(realized_total),
            "profit": float(profit_total),
            "reason": reason,
        })

        self.cycles = [c for c in self.cycles if c.shares > 1e-12]
        return realized_total

    def current_value(self, price):
        return sum(c.shares * price for c in self.cycles)


# ==========================================================
# 4. XIRR
# ==========================================================

def compute_xirr(cashflows):
    if not cashflows:
        return np.nan

    flows = []
    for d, amt in cashflows:
        d = pd.Timestamp(d).to_pydatetime().replace(tzinfo=None)
        flows.append((d, amt))

    flows = sorted(flows, key=lambda x: x[0])
    t0 = flows[0][0]

    def npv(rate):
        s = 0.0
        for d, amt in flows:
            years = (d - t0).days / 365.0
            s += amt / ((1 + rate) ** years)
        return s

    low, high = -0.999, 5.0
    npv_low = npv(low)
    npv_high = npv(high)

    if npv_low * npv_high > 0:
        return np.nan

    for _ in range(100):
        mid = (low + high) / 2.0
        val = npv(mid)
        if abs(val) < 1e-6:
            return mid
        if npv_low * val < 0:
            high = mid
            npv_high = val
        else:
            low = mid
            npv_low = val
    return mid


# ==========================================================
# 5. Strategy runner (B1.2 + low-marker heavy buys)
# ==========================================================

def run_strategy_b1_2(ticker, start=START_DATE):
    # Price and technicals
    close = load_close(ticker, start)
    ohlcv = load_ohlcv(ticker, start).reindex(close.index)

    high = ohlcv["High"]
    low = ohlcv["Low"]
    volume = ohlcv["Volume"]

    ema200 = close.rolling(EMA_WINDOW).mean()
    rsi14 = compute_rsi(close, period=RSI_PERIOD)

    # Log-based True Range
    log_close = np.log(close)
    log_high = np.log(high)
    log_low = np.log(low)
    log_prev_close = log_close.shift(1)

    lr1 = (log_high - log_low).abs()
    lr2 = (log_high - log_prev_close).abs()
    lr3 = (log_low - log_prev_close).abs()
    log_tr = pd.concat([lr1, lr2, lr3], axis=1).max(axis=1)

    vol_mean = volume.rolling(LOW_MARKER_LOOKBACK).mean()
    logtr_mean = log_tr.rolling(LOW_MARKER_LOOKBACK).mean()

    # Drawdown, CLV
    ath_series = close.cummax()
    drawdown = (ath_series - close) / ath_series

    day_range = (high - low).replace(0, np.nan)
    clv = (close - low) / day_range

    # Low-marker conditions
    condA = drawdown >= LOW_MARKER_DD_MIN
    condB = close <= ema200 * LOW_MARKER_EMA_MULT
    condC = volume >= vol_mean * LOW_MARKER_VOL_MULT
    condD = log_tr >= logtr_mean * LOW_MARKER_ATR_MULT
    condE = clv >= LOW_MARKER_CLV_MIN
    condF = rsi14 <= LOW_MARKER_RSI_MAX

    low_marker = condA & condB & condC & condD & condE & condF
    marker_dates = low_marker[low_marker].index
    marker_count = int(low_marker.sum())

    # Ladder signals
    signals = detect_bottoms_b1(close, DD_THRESHOLDS)
    p = Portfolio()

    # Normal ladder logic
    for date, price, ath_pre, thr_idx in signals:
        if pd.isna(ema200.loc[date]) or price < ema200.loc[date]:
            continue

        base_amt = BUY_AMOUNTS.get(thr_idx, 250.0)
        dd = (ath_pre - price) / ath_pre
        dd_pct = int(round(dd * 100))
        reason_buy = f"NORMAL_BUY_DD_{dd_pct}%"

        p.buy(date, price, base_amt, MAX_NORMAL_CAP, reason_buy)

        future = close[close.index > date]

        for i, (tp_mult, frac) in enumerate(zip(TP_LEVELS, TP_FRACS), start=1):
            target = ath_pre * tp_mult
            hit = future[future >= target]
            if hit.empty:
                break
            hit_date = hit.index[0]
            hit_price = hit.iloc[0]

            reason_sell = f"SELL_TP{i}"
            p.sell_fraction(hit_date, hit_price, frac, reason_sell)
            future = future[future.index > hit_date]

    # Heavy low-marker buys
    for d in marker_dates:
        price = close.loc[d]
        p.heavy_buy(d, price, HEAVY_BUY_BASE, reason="HEAVY_BUY_LOW_MARKER")

    # ATH events timeline
    ath_events = []
    ath = -np.inf
    for date, price in close.items():
        price = float(price)
        if price > ath:
            ath = price
            ath_events.append({
                "type": "ATH_EVENT",
                "date": pd.Timestamp(date).to_pydatetime().replace(tzinfo=None),
                "price": price,
                "amount": 0.0,
                "shares": 0.0,
                "shares_sold": 0.0,
                "realized": 0.0,
                "profit": 0.0,
                "reason": "NEW_ATH",
            })

    # Final valuation + XIRR
    last_price = float(close.iloc[-1])
    last_date = close.index[-1]

    held_value = p.current_value(last_price)
    final_value = held_value + p.profit_booked
    total_pnl = final_value - p.invested

    cashflows = []
    for log in p.trade_log:
        if log["type"] in ("BUY", "HEAVY_BUY"):
            cashflows.append((log["date"], -log["amount"]))
        elif log["type"] == "SELL":
            cashflows.append((log["date"], log["realized"]))
    if held_value > 0:
        cashflows.append(
            (pd.Timestamp(last_date).to_pydatetime().replace(tzinfo=None), held_value)
        )

    xirr_dec = compute_xirr(cashflows)
    xirr_pct = xirr_dec * 100.0 if pd.notna(xirr_dec) else np.nan

    return {
        "ticker": ticker,
        "num_signals": len(signals),
        "invested": p.invested,
        "profit_booked": p.profit_booked,
        "held_value": held_value,
        "final_value": final_value,
        "total_pnl": total_pnl,
        "xirr": xirr_pct,
        "trade_log": p.trade_log,
        "ath_events": ath_events,
        "marker_count": marker_count,
        "last_date": pd.Timestamp(last_date).to_pydatetime().replace(tzinfo=None),
    }


# ==========================================================
# 6. Run for all tickers + exports + portfolio stats
# ==========================================================

if __name__ == "__main__":
    print("Running B1.2 backtest...\n")
    results = []
    for t in TICKERS:
        print(f"Processing {t} ...")
        r = run_strategy_b1_2(t)
        results.append(r)

    # Summary per ticker
    summary = pd.DataFrame([{
        "ticker": r["ticker"],
        "num_signals": r["num_signals"],
        "invested": r["invested"],
        "profit_booked": r["profit_booked"],
        "held_value": r["held_value"],
        "final_value": r["final_value"],
        "total_pnl": r["total_pnl"],
        "xirr": r["xirr"],  # in %
    } for r in results])

    print("\n======== FINAL SUMMARY (B1.2) ========\n")
    print(summary.to_string(index=False))

    # Low-marker counts
    marker_counts = pd.DataFrame([{
        "ticker": r["ticker"],
        "low_marker_triggers": r["marker_count"],
    } for r in results])

    print("\n====== LOW MARKER COUNTS ======\n")
    print(marker_counts.to_string(index=False))

    # Trade log dataframe (for CSV / Excel)
    trade_rows = []
    for r in results:
        for log in r["trade_log"]:
            row = {"ticker": r["ticker"]}
            row.update(log)
            trade_rows.append(row)

    trade_df = pd.DataFrame(trade_rows)
    if "date" in trade_df.columns:
        trade_df["date"] = pd.to_datetime(trade_df["date"]).dt.tz_localize(None)

    # ----- Portfolio-wide stats (combined cashflow) -----
    total_invested = sum(r["invested"] for r in results)
    total_profit_booked = sum(r["profit_booked"] for r in results)
    total_held_value = sum(r["held_value"] for r in results)
    total_final_value = total_profit_booked + total_held_value
    total_pnl = total_final_value - total_invested

    combined_cf = []
    for r in results:
        for log in r["trade_log"]:
            if log["type"] in ("BUY", "HEAVY_BUY"):
                combined_cf.append((log["date"], -log["amount"]))
            elif log["type"] == "SELL":
                combined_cf.append((log["date"], log["realized"]))

    if results:
        portfolio_end = max(r["last_date"] for r in results)
        combined_cf.append((portfolio_end, total_held_value))
        portfolio_xirr_dec = compute_xirr(combined_cf)
        portfolio_xirr_pct = portfolio_xirr_dec * 100.0 if pd.notna(portfolio_xirr_dec) else np.nan
    else:
        portfolio_xirr_pct = np.nan

    print("\n====== PORTFOLIO-WIDE STATS (COMBINED CASHFLOW) ======\n")
    print(f"Total invested     : {total_invested:,.2f}")
    print(f"Total profit_booked: {total_profit_booked:,.2f}")
    print(f"Total held_value   : {total_held_value:,.2f}")
    print(f"Total final_value  : {total_final_value:,.2f}")
    print(f"Total PnL          : {total_pnl:,.2f}")
    if pd.notna(portfolio_xirr_pct):
        print(f"Portfolio XIRR     : {portfolio_xirr_pct:,.2f}%")
    else:
        print("Portfolio XIRR     : NaN")

    # ----- CSV / Excel exports -----
    summary.to_csv("strategy_results_b1_2.csv", index=False)
    trade_df.to_csv("trade_logs_b1_2.csv", index=False)
    summary.to_excel("strategy_results_b1_2.xlsx", index=False)
    trade_df.to_excel("trade_logs_b1_2.xlsx", index=False)
    marker_counts.to_csv("low_marker_counts_b1_2.csv", index=False)

    print("\nCSV/Excel reports saved:")
    print(" - strategy_results_b1_2.csv / .xlsx")
    print(" - trade_logs_b1_2.csv / .xlsx")
    print(" - low_marker_counts_b1_2.csv")

    # ----- EVENT TABLES (Format A, chronological: oldest TOP, newest BOTTOM) -----
    for r in results:
        # merge ATH events + trade_log, then sort by date ascending
        rows = []
        rows.extend(r["ath_events"])
        rows.extend(r["trade_log"])
        events_df = pd.DataFrame(rows)

        if events_df.empty:
            continue

        events_df["date"] = pd.to_datetime(events_df["date"])
        events_df.sort_values("date", inplace=True)  # oldest at top

        # Ensure all columns exist and ordered
        for col in EVENT_COLUMNS:
            if col not in events_df.columns:
                events_df[col] = 0.0 if col not in ("type", "date", "reason") else ""

        df_print = events_df[EVENT_COLUMNS].copy()

        # Color just the 'type' column
        df_print["type"] = df_print["type"].astype(str).apply(color_type)

        print(f"\n----- EVENTS FOR {r['ticker']} -----\n")
        print(df_print.to_string(index=False))

    # ----- Strategy text -----
    print("""

NOTE: Prices are split- and dividend-adjusted (yfinance auto_adjust=True)

========= STRATEGY B1.2 DESCRIPTION (UPDATED) =========

ENTRY LOGIC (B1.2):
  1) Track all-time-high (ATH) for each ticker.
  2) When price drops 15% below ATH: first BUY.
  3) When price drops 20% below ATH: second BUY.
  4) When price drops 25% below ATH: third BUY.
  5) Buy sizes for these ladders are configured via BUY_AMOUNTS.
  6) Only buy if price > 200-day EMA (trend filter, no catching falling knives).
  7) Hard cap: max invested per ticker = MAX_NORMAL_CAP (normal buys only).
  8) Realized profit from normal SELLs is accumulated in a low-marker pool
     and only deployed into heavy low-marker buys.

LOW-MARKER HEAVY BUY:
  - On any day where ALL of the following hold:
      * Drawdown_from_ATH ≥ LOW_MARKER_DD_MIN
      * Price ≤ EMA200 × LOW_MARKER_EMA_MULT
      * Volume_today ≥ LOW_MARKER_VOL_MULT × Volume_60d_avg
      * logATR_today ≥ LOW_MARKER_ATR_MULT × logATR_60d_avg
      * (Close - Low) / (High - Low) ≥ LOW_MARKER_CLV_MIN
      * RSI(14) ≤ LOW_MARKER_RSI_MAX
    then:
      * Execute a heavy BUY of (HEAVY_BUY_BASE + accumulated low-marker pool).
      * Heavy-buy cycles are never sold and do not affect the normal cap.

EXIT / PROFIT-TAKING:
  - For each normal BUY, define take-profit levels at:
      * TP_LEVELS (e.g. 1.20, 1.40, 1.60 × ATH before correction)
  - At each target hit, sell TP_FRACS (e.g. 15%) of total normal shares.
  - Remaining normal shares plus all heavy-buy shares are held until the end.
  - Realized profit from normal sells feeds into the low-marker pool.

NOTES:
  - Ticker-level XIRR is expressed in percent.
  - Portfolio-wide XIRR is computed from combined cashflows:
      * All BUY / HEAVY_BUY amounts as negative cashflows,
      * All SELL realized amounts as positive cashflows,
      * One final positive cashflow equal to total held_value at the global end date.
""")
