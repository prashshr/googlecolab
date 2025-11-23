import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ==========================================================
# 0. USER-CONFIGURABLE PARAMETERS
# ==========================================================

# Data / universe
DATA_START = "2021-01-01"
TICKERS = ["NVDA", "MSFT", "PLTR", "TSLA", "AMZN", "ASML", "CRWD", "META", "AVGO", "NOW"]

# Enable ANSI colors in console output
ENABLE_COLORS = True

# Ladder (normal B1.2) drawdown thresholds from ATH
LADDER_THRESHOLDS = [0.15, 0.20, 0.25]      # 15%, 20%, 25%

# Normal buy sizes per ladder step (before any special logic)
FIRST_BUY_AMOUNT = 250.0
SECOND_BUY_AMOUNT = 250.0
THIRD_BUY_AMOUNT = 250.0

# Take-profit levels and fractions (for normal buys)
TP_LEVELS = [1.20, 1.40, 1.60]              # x ATH
TP_FRACTIONS = [0.15, 0.15, 0.15]           # 15% each TP (of normal shares)

# Trend filter
EMA_PERIOD = 200

# Hard capital cap for normal buys (per ticker)
MAX_CAP_NORMAL = 3000.0

# Heavy low-marker buy base size (per marker), before profit pool
HEAVY_BASE_AMOUNT = 1000.0

# Low-marker (capitulation) conditions (relaxed version)
LOWMARKER_DD_MIN = 0.30           # Drawdown_from_ATH ≥ 30%
LOWMARKER_EMA_MULT = 0.85         # Price ≤ 200d_EMA * 0.85
VOL_LOOKBACK = 60
VOL_MULT = 1.3                    # Volume_today ≥ 1.3 × Vol_60d_avg
LOGTR_LOOKBACK = 60
LOGTR_MULT = 1.3                  # logATR_today ≥ 1.3 × logATR_60d_avg
CLV_MIN = 0.40                    # (Close - Low)/(High - Low) ≥ 0.40
RSI_PERIOD = 14
RSI_MAX = 40.0                    # RSI(14) ≤ 40

# XIRR solver bounds
XIRR_LOW = -0.999
XIRR_HIGH = 5.0

# ==========================================================
# ANSI COLORS (controlled by ENABLE_COLORS)
# ==========================================================

if ENABLE_COLORS:
    COLOR_RESET = "\033[0m"
    COLOR_ATH = "\033[33m"   # Yellow
    COLOR_BUY = "\033[32m"   # Green
    COLOR_SELL = "\033[31m"  # Red
else:
    COLOR_RESET = ""
    COLOR_ATH = ""
    COLOR_BUY = ""
    COLOR_SELL = ""


# ==========================================================
# 1. Robust loader for Close prices (split-adjusted, tz-naive)
# ==========================================================

def load_close(ticker, start=DATA_START):
    df = yf.download(
        ticker,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="column"
    )

    # Flatten possible MultiIndex
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

def load_ohlcv(ticker, start=DATA_START):
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
            raise ValueError(f"Missing columns {missing} for {ticker} from both download() and history()")

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
                 example: [0.15, 0.20, 0.25]
    Returns list of (date, price, ath_pre_correction)
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

        for thr in thresholds:
            if not hit[thr] and dd >= thr:
                signals.append((date, price, ath))
                hit[thr] = True

    return signals


# ==========================================================
# 3. Portfolio engine (low-marker profit pool, heavy buys)
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

        # profits from normal sells accumulate here
        # and are deployed only at the next low-marker heavy buy.
        self.low_marker_pool = 0.0

    def buy(self, date, price, base_amount, max_cap):
        """
        Normal buy:
        - Uses ONLY base_amount (e.g. 250).
        - Does NOT use the low_marker_pool.
        - Respects max_cap for normal buys.
        """
        amount = base_amount

        if self.invested + amount > max_cap:
            amount = max(0, max_cap - self.invested)

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
            "shares_sold": None,
            "realized": None,
            "profit": None,
            "is_heavy": False,
        })

    def heavy_buy(self, date, price, base_amount):
        """
        Heavy capitulation buy:
        - Amount = base_amount (e.g. 1000 USD) + accumulated low_marker_pool.
        - Heavy-buy cycles are never sold (is_heavy=True).
        - Counts toward 'invested' for reporting.
        """
        amount = base_amount + self.low_marker_pool
        self.low_marker_pool = 0.0  # reset pool after deployment

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
            "shares_sold": None,
            "realized": None,
            "profit": None,
            "is_heavy": True,
        })

    def sell_fraction(self, date, price, fraction):
        """
        Sells 'fraction' of TOTAL *normal* shares (is_heavy=False),
        pro-rata across normal cycles.

        Heavy-buy cycles (is_heavy=True) are NEVER sold.
        Realized profit from these sells goes into low_marker_pool.
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
                continue  # do not touch heavy cycles

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

        # profit from normal sells is booked and saved for next low-marker heavy buy
        self.low_marker_pool += profit_total
        self.profit_booked += profit_total

        self.trade_log.append({
            "type": "SELL",
            "date": pd.Timestamp(date).to_pydatetime().replace(tzinfo=None),
            "price": float(price),
            "amount": None,
            "shares": None,
            "shares_sold": float(target - remaining),
            "realized": float(realized_total),
            "profit": float(profit_total),
            "is_heavy": False,
        })

        # remove empty cycles
        self.cycles = [c for c in self.cycles if c.shares > 1e-12]
        return realized_total

    def current_value(self, price):
        # heavy + normal cycles; all are part of portfolio value
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

    low, high = XIRR_LOW, XIRR_HIGH
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
# Helper: build chronological event log (ATH + trades)
# ==========================================================

def build_event_log(close, trade_log):
    """
    Build a chronological event log:
    - ATH_EVENT (yellow)
    - BUY (green)
    - HEAVY_BUY (green)
    - SELL (red)

    Returns list of dicts with the 9 columns:
      date, event, price, amount, shares, shares_sold,
      total_shares_after, cum_invested, cum_profit_booked
    """
    # 1) ATH events
    ath_events = []
    ath = -np.inf
    for date, price in close.items():
        price = float(price)
        if price > ath:
            ath = price
            ath_events.append({
                "date": pd.Timestamp(date).to_pydatetime().replace(tzinfo=None),
                "event": "ATH_EVENT",
                "price": price,
                "amount": np.nan,
                "shares": np.nan,
                "shares_sold": np.nan,
                "profit": 0.0,
            })

    # 2) Trade events
    trade_events = []
    for log in trade_log:
        e = {
            "date": log["date"],
            "event": log["type"],  # BUY / HEAVY_BUY / SELL
            "price": log["price"],
            "amount": log["amount"] if log["amount"] is not None else np.nan,
            "shares": log["shares"] if log["shares"] is not None else np.nan,
            "shares_sold": log["shares_sold"] if log["shares_sold"] is not None else np.nan,
            "profit": log["profit"] if log["profit"] is not None else 0.0,
        }
        trade_events.append(e)

    # 3) Combine and sort chronologically
    all_events = ath_events + trade_events
    all_events.sort(key=lambda e: (e["date"], 0 if e["event"] == "ATH_EVENT" else 1))

    # 4) Walk through and build cumulative state
    events_out = []
    cum_shares = 0.0
    cum_invested = 0.0
    cum_profit = 0.0

    for e in all_events:
        ev = e["event"]

        if ev in ("BUY", "HEAVY_BUY"):
            if not np.isnan(e["shares"]):
                cum_shares += e["shares"]
            if not np.isnan(e["amount"]):
                cum_invested += e["amount"]

        elif ev == "SELL":
            if not np.isnan(e["shares_sold"]):
                cum_shares -= e["shares_sold"]
            cum_profit += e["profit"]

        # ATH_EVENT: no state change

        events_out.append({
            "date": e["date"],
            "event": ev,
            "price": e["price"],
            "amount": e["amount"],
            "shares": e["shares"],
            "shares_sold": e["shares_sold"],
            "total_shares_after": cum_shares,
            "cum_invested": cum_invested,
            "cum_profit_booked": cum_profit,
        })

    return events_out


# ==========================================================
# 5. Strategy runner (with log-ATR marker + relaxed thresholds)
# ==========================================================

def run_strategy_b1_2(ticker, start=DATA_START):
    # Base price series
    close = load_close(ticker, start)

    # OHLCV for technicals (aligned to close index)
    ohlcv = load_ohlcv(ticker, start)
    ohlcv = ohlcv.reindex(close.index)

    high = ohlcv["High"]
    low = ohlcv["Low"]
    volume = ohlcv["Volume"]

    # 200-day EMA (currently SMA via rolling mean)
    ema200 = close.rolling(EMA_PERIOD).mean()

    # RSI
    rsi14 = compute_rsi(close, period=RSI_PERIOD)

    # -------- Upgrade D: log-based True Range & ATR-style measure --------
    log_close = np.log(close)
    log_high = np.log(high)
    log_low = np.log(low)
    log_prev_close = log_close.shift(1)

    lr1 = (log_high - log_low).abs()
    lr2 = (log_high - log_prev_close).abs()
    lr3 = (log_low - log_prev_close).abs()
    log_tr = pd.concat([lr1, lr2, lr3], axis=1).max(axis=1)

    vol_60 = volume.rolling(VOL_LOOKBACK).mean()
    log_tr_60 = log_tr.rolling(LOGTR_LOOKBACK).mean()

    # Drawdown from ATH
    ath_series = close.cummax()
    drawdown = (ath_series - close) / ath_series

    # Intraday close-location value (CLV)
    day_range = (high - low).replace(0, np.nan)
    clv = (close - low) / day_range  # can be NaN when no range

    # -------- RELAXED LOW-MARKER CONDITIONS --------
    condA = drawdown >= LOWMARKER_DD_MIN
    condB = close <= ema200 * LOWMARKER_EMA_MULT
    condC = volume >= VOL_MULT * vol_60
    condD = log_tr >= LOGTR_MULT * log_tr_60
    condE = clv >= CLV_MIN
    condF = rsi14 <= RSI_MAX

    low_marker = condA & condB & condC & condD & condE & condF
    marker_dates = low_marker[low_marker].index
    marker_count = int(low_marker.sum())

    # -------- Normal B1.2 ladder logic --------
    thresholds = LADDER_THRESHOLDS
    tp_levels = TP_LEVELS
    tp_fracs = TP_FRACTIONS

    signals = detect_bottoms_b1(close, thresholds)
    p = Portfolio()

    for date, price, ath_pre in signals:
        # Trend filter
        if pd.isna(ema200.loc[date]) or price < ema200.loc[date]:
            continue

        # Buy size selection based on DD
        dd = (ath_pre - price) / ath_pre
        if dd >= thresholds[2]:
            base_amt = THIRD_BUY_AMOUNT
        elif dd >= thresholds[1]:
            base_amt = SECOND_BUY_AMOUNT
        else:
            base_amt = FIRST_BUY_AMOUNT

        # Normal ladder buy (uses only base_amt, not pool)
        p.buy(date, price, base_amt, MAX_CAP_NORMAL)

        # Take-profits based on ATH * multiples
        future = close[close.index > date]

        for tp_mult, frac in zip(tp_levels, tp_fracs):
            target = ath_pre * tp_mult
            hit = future[future >= target]
            if hit.empty:
                break
            hit_date = hit.index[0]
            hit_price = hit.iloc[0]

            p.sell_fraction(hit_date, hit_price, frac)
            future = future[future.index > hit_date]

    # -------- Heavy low-marker buys --------
    for d in marker_dates:
        price = close.loc[d]
        p.heavy_buy(d, price, HEAVY_BASE_AMOUNT)

    # -------- Final valuation + XIRR --------
    last_price = float(close.iloc[-1])
    last_date = close.index[-1]

    held_value = p.current_value(last_price)
    final_value = held_value + p.profit_booked
    total_pnl = final_value - p.invested

    # Per-ticker cashflows (for per-ticker XIRR)
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

    xirr_decimal = compute_xirr(cashflows)
    xirr_pct = xirr_decimal * 100.0 if pd.notna(xirr_decimal) else np.nan

    # -------- Build chronological event log (ATH + trades) --------
    events = build_event_log(close, p.trade_log)

    return {
        "ticker": ticker,
        "num_signals": len(signals),
        "invested": p.invested,
        "profit_booked": p.profit_booked,
        "held_value": held_value,
        "final_value": final_value,
        "total_pnl": total_pnl,
        "xirr": xirr_pct,      # percentage
        "trade_log": p.trade_log,
        "marker_count": marker_count,
        "last_date": pd.Timestamp(last_date).to_pydatetime().replace(tzinfo=None),
        "events": events,
    }


# ==========================================================
# Helper: pretty-print event tables with colors
# ==========================================================

def fmt_float(x, width, decimals=2):
    if pd.isna(x):
        return " " * width
    fmt = f"{{:>{width}.{decimals}f}}"
    return fmt.format(x)

def print_events_for_ticker(ticker, events):
    if not events:
        print(f"\n===== EVENTS FOR {ticker} =====")
        print("(no events)")
        return

    print(f"\n===== EVENTS FOR {ticker} =====\n")

    # 9 columns: date, event, price, amount, shares, shares_sold,
    #            total_shares_after, cum_invested, cum_profit_booked
    header = (
        f"{'date':<19} "
        f"{'event':<10} "
        f"{'price':>10} "
        f"{'amount':>10} "
        f"{'shares':>10} "
        f"{'sold':>10} "
        f"{'tot_shares':>12} "
        f"{'cum_invest':>12} "
        f"{'cum_profit':>12}"
    )
    print(header)
    print("-" * len(header))

    for e in events:
        d = e["date"].strftime("%Y-%m-%d")
        ev = e["event"]
        price = fmt_float(e["price"], 10, 2)
        amount = fmt_float(e["amount"], 10, 2)
        shares = fmt_float(e["shares"], 10, 4)
        sold = fmt_float(e["shares_sold"], 10, 4)
        tot_shares = fmt_float(e["total_shares_after"], 12, 4)
        cum_inv = fmt_float(e["cum_invested"], 12, 2)
        cum_profit = fmt_float(e["cum_profit_booked"], 12, 2)

        line = (
            f"{d:<19} "
            f"{ev:<10} "
            f"{price} "
            f"{amount} "
            f"{shares} "
            f"{sold} "
            f"{tot_shares} "
            f"{cum_inv} "
            f"{cum_profit}"
        )

        # Color by event type
        if ev == "ATH_EVENT":
            color = COLOR_ATH
        elif ev in ("BUY", "HEAVY_BUY"):
            color = COLOR_BUY
        elif ev == "SELL":
            color = COLOR_SELL
        else:
            color = ""

        if ENABLE_COLORS and color:
            print(color + line + COLOR_RESET)
        else:
            print(line)


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

    summary = pd.DataFrame([{
        "ticker": r["ticker"],
        "num_signals": r["num_signals"],
        "invested": r["invested"],
        "profit_booked": r["profit_booked"],
        "held_value": r["held_value"],
        "final_value": r["final_value"],
        "total_pnl": r["total_pnl"],
        "xirr_%": r["xirr"],  # percentage
    } for r in results])

    print("\n======== FINAL SUMMARY (B1.2) ========\n")
    print(summary.to_string(index=False))

    # ----- LOW-MARKER COUNTS -----
    marker_counts = pd.DataFrame([{
        "ticker": r["ticker"],
        "low_marker_triggers": r["marker_count"],
    } for r in results])

    print("\n====== LOW MARKER COUNTS ======\n")
    print(marker_counts.to_string(index=False))

    # ----- Build trade log dataframe -----
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

    # Combined cashflows (treat as one portfolio)
    combined_cf = []
    for r in results:
        for log in r["trade_log"]:
            if log["type"] in ("BUY", "HEAVY_BUY"):
                combined_cf.append((log["date"], -log["amount"]))
            elif log["type"] == "SELL":
                combined_cf.append((log["date"], log["realized"]))

    # Single terminal cashflow at the global end date
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

    # ----- Print per-ticker event timelines -----
    for r in results:
        print_events_for_ticker(r["ticker"], r["events"])

    print("""
========= STRATEGY B1.2 DESCRIPTION (UPDATED) =========

ENTRY LOGIC (B1.2):
  1) Track all-time-high (ATH) for each ticker.
  2) When price drops 15% below ATH: first BUY.
  3) When price drops 20% below ATH: second BUY.
  4) When price drops 25% below ATH: third BUY.
  5) Buy sizes (per ATH regime) = 250, 500, 750 USD (see FIRST/SECOND/THIRD_BUY_AMOUNT).
  6) Only buy if price > 200-day EMA (trend filter, no catching falling knives).
  7) Hard cap: max invested per ticker = 3000 USD for normal buys (MAX_CAP_NORMAL).
  8) Realized profit from normal SELLs is accumulated in a low-marker pool
     (not recycled into the next normal buy).

LOW-MARKER HEAVY BUY (with log-ATR and relaxed thresholds):
  - On any day where ALL of the following hold:
      * Drawdown_from_ATH ≥ 30%
      * Price ≤ 200d_EMA * 0.85
      * Volume_today ≥ 1.3 × Volume_60d_avg
      * logATR_today ≥ 1.3 × logATR_60d_avg
      * (Close - Low) / (High - Low) ≥ 0.4
      * RSI(14) ≤ 40
    then:
      * Execute a heavy BUY of (HEAVY_BASE_AMOUNT + accumulated low-marker pool).
      * Heavy-buy cycles are never sold and do not affect the 3000 USD cap
        for normal buys, though the invested amount is counted in 'invested'.

EXIT / PROFIT-TAKING:
  - For each normal BUY, define take-profit levels at:
      * 1.20 × ATH before correction
      * 1.40 × ATH before correction
      * 1.60 × ATH before correction
  - At each target hit, sell these fractions of TOTAL *normal* shares:
      * 15% of total shares
      * 15% of total shares
      * 15% of total shares
  - Remaining normal shares plus all heavy-buy shares are held until the end.
  - Realized profit from these normal sells feeds into the low-marker pool.

NOTES:
  - 'xirr_%' in the summary is expressed in PERCENT (e.g. 76.71 = 76.71%).
  - Portfolio-wide XIRR is computed using a combined cashflow method:
      * all BUY / HEAVY_BUY as negative cashflows,
      * all SELL realized amounts as positive cashflows,
      * one final positive cashflow equal to the total held_value at the end date.
""")
