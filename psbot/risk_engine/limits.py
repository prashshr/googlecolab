"""
Limit guardrails: caps on position sizes, investment amount, etc.
"""

from datetime import datetime

from .exceptions import GuardrailRejection

MAX_INVESTMENT_PER_TICKER = 5000.0
MAX_SHARES_PER_TICKER = 10000.0
MAX_HEAVY_BUYS_PER_CYCLE = 3


def check_position_limits(order):
    """
    Ensure the order keeps position size within configured bounds.
    """
    ticker = order.get("ticker")
    side = order.get("side")
    qty = float(order.get("qty", 0.0) or 0.0)
    current = float(order.get("current_position_shares", 0.0) or 0.0)
    max_shares = float(order.get("max_shares", MAX_SHARES_PER_TICKER))

    if not ticker or qty < 0:
        raise GuardrailRejection("Invalid ticker/quantity payload.")

    if side == "BUY" and current + qty > max_shares + 1e-9:
        raise GuardrailRejection(
            f"{ticker}: share cap exceeded ({current + qty:.2f} > {max_shares:.2f})."
        )

    if side == "SELL" and qty - current > 1e-6:
        raise GuardrailRejection(
            f"{ticker}: attempting to sell more than held ({qty:.2f} > {current:.2f})."
        )


def check_investment_limits(order):
    """
    Prevent capital usage beyond per-ticker investment caps.
    """
    ticker = order.get("ticker")
    amount = float(order.get("amount", 0.0) or 0.0)
    current_inv = float(order.get("current_invested", 0.0) or 0.0)
    max_inv = float(order.get("max_investment", MAX_INVESTMENT_PER_TICKER))

    if order.get("side") != "BUY":
        return

    if current_inv + amount > max_inv + 1e-9:
        raise GuardrailRejection(
            f"{ticker}: investment cap exceeded ({current_inv + amount:.2f} > {max_inv:.2f})."
        )
