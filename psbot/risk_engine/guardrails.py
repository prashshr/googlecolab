"""
Central guardrail controller that fans out to individual safety modules.
"""

from datetime import datetime

from . import limits, data_checks, duplicates, time_filters, broker_health, circuit_breakers
from .exceptions import GuardrailRejection


def before_order(order, mode="paper"):
    """
    Run every safety module before an order is passed to any broker.

    Parameters
    ----------
    order : dict
        Order payload with keys such as ticker, side, qty, price, reason, timestamp.
    mode : str
        'paper', 'live', or 'disabled'. Disabled raises immediately.
    """
    if order is None:
        raise GuardrailRejection("Empty order payload.")

    normalized_mode = (mode or "paper").lower()
    order = dict(order)
    order.setdefault("timestamp", datetime.utcnow())

    if normalized_mode not in {"paper", "live", "disabled"}:
        raise GuardrailRejection(f"Unknown trading mode '{mode}'.")

    if normalized_mode == "disabled":
        raise GuardrailRejection("Trading disabled by TRADING_MODE flag.")

    # Execute all guardrails. Each module raises GuardrailRejection on failure.
    limits.check_position_limits(order)
    limits.check_investment_limits(order)

    data_checks.check_price_validity(order)
    data_checks.check_stale_data(order)

    duplicates.check_duplicate(order)

    time_filters.check_market_session(order)

    broker_health.check_api_stability(order)

    circuit_breakers.check_global_market_risk(order)

    # If we arrive here, record the order for duplicate tracking.
    duplicates.record_order(order)
    return True
