"""
High-level circuit breakers (market crashes, drawdowns, etc.).
"""

from .exceptions import GuardrailRejection

ACCOUNT_DRAWDOWN_LIMIT = 0.15
MARKET_CRASH_LIMIT = 0.05


def check_global_market_risk(order):
    if order.get("simulated"):
        return

    account_dd = order.get("account_drawdown")
    if account_dd is not None and account_dd > ACCOUNT_DRAWDOWN_LIMIT:
        raise GuardrailRejection("Account drawdown exceeds limit.")

    market_dd = order.get("market_drawdown")
    if market_dd is not None and market_dd > MARKET_CRASH_LIMIT:
        raise GuardrailRejection("Market crash circuit breaker triggered.")
