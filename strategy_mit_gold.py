import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

GOLD_TICKER = "GLD"
MONTHLY_GOLD_DEPOSIT = 500.0

# ANSI colors for pretty printing
GOLD_COLOR = "\033[33m"
RESET_COLOR = "\033[0m"

# ==========================================================
# 1. Robust loader for Close prices (split-adjusted, tz-naive)
# ==========================================================

def load_close(ticker, start="2015-01-01"):
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

def load_ohlcv(ticker, start="2015-01-01"):
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

    # If download is empty or missing columns, fall back to history()
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

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ==========================================================
# 2. B1 signal detection (15%,20%,25% drawdowns from ATH)
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
# 3. Portfolio engine (no reinvest on normal sells)
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
        # and are only deployed at the next low-marker heavy buy.
        self.low_marker_pool = 0.0

    def buy(self, date, price, base_amount, max_cap):
        """
        Normal buy:
        - Uses ONLY base_amount (250, 500, 750).
        - Does NOT use the low_marker_pool.
        - Respects max_cap (normal cap).
        Returns the actually executed amount (may be 0 if at cap).
        """
        amount = base_amount

        # respect hard cap for normal buys
        if self.invested + amount > max_cap:
            amount = max(0, max_cap - self.invested)

        if amount <= 0:
            return 0.0

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
        })
        return amount

    def heavy_buy(self, date, price, base_amount):
        """
        Heavy capitulation buy:
        - Amount = base_amount (e.g. 1000 USD) + accumulated low_marker_pool.
        - Never sold (is_heavy=True).
        - DOES count toward 'invested' (for reporting).
        Returns the total amount invested in this heavy buy.
        """
        amount = base_amount + self.low_marker_pool
        self.low_marker_pool = 0.0  # reset pool after deployment

        if amount <= 0:
            return 0.0

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
        })
        return amount

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
# 5. Gold vault (Option A: principal-only, FIFO)
# ==========================================================

class GoldLot:
    def __init__(self, date, shares, principal):
        self.date = pd.Timestamp(date).to_pydatetime().replace(tzinfo=None)
        self.shares = float(shares)
        self.principal_remaining = float(principal)


class GoldVault:
    """
    Single global vault:
      - Each month: deposit fixed USD into gold.
      - For stock buys: we try to fund from principal in FIFO order.
      - We NEVER touch profit: we only withdraw up to principal_remaining, and only
        if the lot's current market value can cover it. Profit remains in leftover shares.
    """
    def __init__(self, gold_prices: pd.Series):
        if getattr(gold_prices.index, "tz", None) is not None:
            gold_prices = gold_prices.tz_localize(None)
        self.gold_prices = gold_prices.dropna()
        self.lots = []
        self.total_principal_deposited = 0.0

    def _price_asof(self, date):
        date = pd.Timestamp(date).to_pydatetime().replace(tzinfo=None)
        s = self.gold_prices
        # asof-like behavior: last available price on or before date
        s2 = s[s.index <= pd.Timestamp(date)]
        if s2.empty:
            # if nothing before, just take earliest
            return float(s.iloc[0])
        return float(s2.iloc[-1])

    def deposit(self, date, amount):
        price = self._price_asof(date)
        shares = amount / price
        lot = GoldLot(date, shares, amount)
        self.lots.append(lot)
        self.total_principal_deposited += amount

        # Pretty print in "gold" color
        print(f"{GOLD_COLOR}[GOLD DEPOSIT] {date.date()} : +${amount:.2f} at {price:.2f} => {shares:.4f} shares{RESET_COLOR}")

    def withdraw_principal(self, date, needed_amount):
        """
        Try to free `needed_amount` of principal (USD) using FIFO lots with date <= given date.
        - We only use up to lot.principal_remaining and up to lot_value.
        - If price < original, we may not be able to recover full principal.
        Returns the actual amount freed from gold (<= needed_amount).
        """
        if needed_amount <= 0:
            return 0.0

        price = self._price_asof(date)
        remaining = needed_amount
        freed = 0.0

        # FIFO: oldest lots first
        for lot in sorted(self.lots, key=lambda x: x.date):
            if lot.date > pd.Timestamp(date):
                # cannot use future deposits
                continue
            if remaining <= 0:
                break
            if lot.shares <= 0 or lot.principal_remaining <= 0:
                continue

            lot_value = lot.shares * price
            if lot_value <= 0:
                continue

            # Maximum cash we can free from this lot without touching profit
            max_cash_from_lot = min(lot_value, lot.principal_remaining)
            if max_cash_from_lot <= 0:
                continue

            take_cash = min(remaining, max_cash_from_lot)
            shares_to_sell = take_cash / price
            if shares_to_sell > lot.shares:
                shares_to_sell = lot.shares
                take_cash = shares_to_sell * price

            lot.shares -= shares_to_sell
            lot.principal_remaining -= take_cash
            remaining -= take_cash
            freed += take_cash

        # Clean empty lots
        self.lots = [
            lot for lot in self.lots
            if lot.shares > 1e-10 or lot.principal_remaining > 1e-6
        ]

        if freed > 0:
            print(f"{GOLD_COLOR}[GOLD WITHDRAW] {date.date()} : -${freed:.2f} principal at {price:.2f}{RESET_COLOR}")
        return freed

    def current_value(self):
        if self.gold_prices.empty:
            return 0.0
        last_price = float(self.gold_prices.iloc[-1])
        total_shares = sum(lot.shares for lot in self.lots)
        return total_shares * last_price

    def principal_remaining(self):
        return sum(lot.principal_remaining for lot in self.lots)


# ==========================================================
# 6. Strategy runner (with log-ATR marker + relaxed thresholds + gold)
# ==========================================================

def run_strategy_b1_2(ticker, gold_vault=None, start="2021-01-01"):
    # Base price series
    close = load_close(ticker, start)

    # OHLCV for technicals (aligned to close index)
    ohlcv = load_ohlcv(ticker, start)
    ohlcv = ohlcv.reindex(close.index)

    high = ohlcv["High"]
    low = ohlcv["Low"]
    volume = ohlcv["Volume"]

    # 200-day EMA (using SMA as in original)
    ema200 = close.rolling(200).mean()

    # RSI(14)
    rsi14 = compute_rsi(close, period=14)

    # -------- Upgrade D: log-based True Range & ATR-style measure --------
    log_close = np.log(close)
    log_high = np.log(high)
    log_low = np.log(low)
    log_prev_close = log_close.shift(1)

    lr1 = (log_high - log_low).abs()
    lr2 = (log_high - log_prev_close).abs()
    lr3 = (log_low - log_prev_close).abs()
    log_tr = pd.concat([lr1, lr2, lr3], axis=1).max(axis=1)

    vol_60 = volume.rolling(60).mean()
    log_tr_60 = log_tr.rolling(60).mean()

    # Drawdown from ATH
    ath_series = close.cummax()
    drawdown = (ath_series - close) / ath_series

    # Intraday close-location value (CLV)
    day_range = (high - low).replace(0, np.nan)
    clv = (close - low) / day_range  # can be NaN when no range

    # -------- RELAXED LOW-MARKER CONDITIONS --------
    condA = drawdown >= 0.30
    condB = close <= ema200 * 0.85
    condC = volume >= 1.3 * vol_60
    condD = log_tr >= 1.3 * log_tr_60
    condE = clv >= 0.40
    condF = rsi14 <= 40.0

    low_marker = condA & condB & condC & condD & condE & condF
    marker_dates = low_marker[low_marker].index
    marker_count = int(low_marker.sum())

    # -------- Normal B1.2 ladder logic --------
    thresholds = [0.15, 0.20, 0.25]        # drawdown levels from ATH
    tp_levels  = [1.20, 1.40, 1.60]        # take-profit levels (x ATH)
    tp_fracs   = [0.15, 0.15, 0.15]        # 15% of normal shares each TP

    signals = detect_bottoms_b1(close, thresholds)
    p = Portfolio()

    MAX_CAP = 3000.0  # cap only for normal buys

    for date, price, ath_pre in signals:
        # Trend filter
        if pd.isna(ema200.loc[date]) or price < ema200.loc[date]:
            continue

        # Buy size selection: 250 / 500 / 750 based on DD
        dd = (ath_pre - price) / ath_pre
        if dd >= 0.25:
            base_amt = 750.0
        elif dd >= 0.20:
            base_amt = 500.0
        else:
            base_amt = 250.0

        executed_amt = p.buy(date, price, base_amt, MAX_CAP)

        # Use gold principal for the executed amount (if vault is provided)
        if gold_vault is not None and executed_amt > 0:
            gold_vault.withdraw_principal(date, executed_amt)

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
    HEAVY_BASE = 1000.0  # base amount for each low-marker
    for d in marker_dates:
        price = close.loc[d]
        executed_amt = p.heavy_buy(d, price, HEAVY_BASE)
        if gold_vault is not None and executed_amt > 0:
            gold_vault.withdraw_principal(d, executed_amt)

    # -------- Final valuation + XIRR (stocks only) --------
    last_price = float(close.iloc[-1])
    last_date = close.index[-1]

    held_value = p.current_value(last_price)
    final_value = held_value + p.profit_booked
    total_pnl = final_value - p.invested

    # Per-ticker cashflows (for per-ticker XIRR), stocks-only
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

    return {
        "ticker": ticker,
        "num_signals": len(signals),
        "invested": p.invested,
        "profit_booked": p.profit_booked,
        "held_value": held_value,
        "final_value": final_value,
        "total_pnl": total_pnl,
        "xirr": xirr_pct,      # store as percentage now
        "trade_log": p.trade_log,
        "marker_count": marker_count,
        "last_date": pd.Timestamp(last_date).to_pydatetime().replace(tzinfo=None),
    }


# ==========================================================
# 7. Run for all tickers + exports + portfolio + gold stats
# ==========================================================

if __name__ == "__main__":
    start_date = "2021-01-01"
    tickers = ["NVDA","MSFT","PLTR","TSLA","AMZN","ASML","CRWD","META","AVGO","NOW"]

    # --- Set up global GOLD vault with monthly deposits (Option A) ---
    print(f"{GOLD_COLOR}Setting up global gold vault using {GOLD_TICKER}...{RESET_COLOR}")
    gold_close = load_close(GOLD_TICKER, start=start_date)
    gold_vault = GoldVault(gold_close)

    # Monthly deposits from start_date to last available gold date
    current = pd.Timestamp(start_date)
    last_gold_date = gold_close.index[-1]

    while current <= last_gold_date:
        # first trading day on or after the 1st of the month
        eligible = gold_close[gold_close.index >= current]
        if not eligible.empty:
            dep_date = eligible.index[0]
            gold_vault.deposit(dep_date, MONTHLY_GOLD_DEPOSIT)
        # move to first day of next month
        year = current.year + (1 if current.month == 12 else 0)
        month = 1 if current.month == 12 else current.month + 1
        current = pd.Timestamp(year=year, month=month, day=1)

    print("Running B1.2 backtest with global GOLD vault...\n")
    results = []
    for t in tickers:
        print(f"Processing {t} ...")
        r = run_strategy_b1_2(t, gold_vault=gold_vault, start=start_date)
        results.append(r)

    summary = pd.DataFrame([{
        "ticker": r["ticker"],
        "num_signals": r["num_signals"],
        "invested": r["invested"],
        "profit_booked": r["profit_booked"],
        "held_value": r["held_value"],
        "final_value": r["final_value"],
        "total_pnl": r["total_pnl"],
        "xirr": r["xirr"],  # already in %
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
    if "date" in trade_df.columns and not trade_df.empty:
        trade_df["date"] = pd.to_datetime(trade_df["date"]).dt.tz_localize(None)

    # ----- Portfolio-wide stats (stocks only, combined cashflow) -----
    total_invested = sum(r["invested"] for r in results)
    total_profit_booked = sum(r["profit_booked"] for r in results)
    total_held_value = sum(r["held_value"] for r in results)
    total_final_value = total_profit_booked + total_held_value
    total_pnl = total_final_value - total_invested

    # Combined cashflows (treat as one stock portfolio, ignore gold for XIRR)
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

    print("\n====== PORTFOLIO-WIDE STATS (STOCKS ONLY, COMBINED CASHFLOW) ======\n")
    print(f"Total invested     : {total_invested:,.2f}")
    print(f"Total profit_booked: {total_profit_booked:,.2f}")
    print(f"Total held_value   : {total_held_value:,.2f}")
    print(f"Total final_value  : {total_final_value:,.2f}")
    print(f"Total PnL          : {total_pnl:,.2f}")
    if pd.notna(portfolio_xirr_pct):
        print(f"Portfolio XIRR     : {portfolio_xirr_pct:,.2f}%")
    else:
        print("Portfolio XIRR     : NaN")

    # ----- GOLD VAULT STATS -----
    gold_value = gold_vault.current_value()
    gold_principal_left = gold_vault.principal_remaining()
    print("\n====== GOLD VAULT (OPTION A: PRINCIPAL-ONLY) ======\n")
    print(f"Gold ticker                     : {GOLD_TICKER}")
    print(f"Total principal deposited       : {gold_vault.total_principal_deposited:,.2f}")
    print(f"Principal still unused/spendable: {gold_principal_left:,.2f}")
    print(f"Gold current market value       : {gold_value:,.2f}")
    print(f"Gold 'pure profit' (value - principal_remaining): {gold_value - gold_principal_left:,.2f}")

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

    print("""
========= STRATEGY B1.2 DESCRIPTION (WITH GLOBAL GOLD VAULT, OPTION A) =========

ENTRY LOGIC (B1.2):
  1) Track all-time-high (ATH) for each ticker.
  2) When price drops 15% below ATH: first BUY.
  3) When price drops 20% below ATH: second BUY.
  4) When price drops 25% below ATH: third BUY.
  5) Buy sizes (per ATH regime) = 250, 500, 750 USD.
  6) Only buy if price > 200-day EMA (trend filter, no catching falling knives).
  7) Hard cap: max invested per ticker = 3000 USD for normal buys.
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
      * Execute a heavy BUY of (1000 USD + accumulated low-marker pool).
      * Heavy-buy cycles are never sold and do not affect the 3000 USD cap
        for normal buys, though the invested amount is still counted in
        'invested' for reporting.

GLOBAL GOLD VAULT OVERLAY (Option A: principal-only, single vault):
  - Ticker used: GLD (gold ETF).
  - From the global start date, on the first trading day of each month:
      * Deposit 500 USD into gold (converted to GLD shares, recorded as a lot).
  - For every stock BUY or HEAVY_BUY:
      * Attempt to fund the executed amount from the gold vault, FIFO by lot date.
      * For each lot, we only withdraw up to:
            min(lot_value_today, lot.principal_remaining)
        so we NEVER use gold returns (profit) to fund stock buys.
      * If gold principal is insufficient, the remainder is conceptually funded
        by new external cash (not tracked separately in this script).
  - Any leftover gold shares (including all accumulated profit) remain invested
    for the full backtest and are reported separately as the GOLD VAULT value.

EXIT / PROFIT-TAKING (stocks):
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
  - Per-ticker 'xirr' here is computed from STOCK cashflows only
    (BUY / HEAVY_BUY / SELL + final marked-to-market of held shares).
  - Portfolio-wide XIRR is also STOCK-only. The gold overlay is reported
    separately so you can see how much principal remains unused and how
    much long-term gold profit has accumulated.
""")
