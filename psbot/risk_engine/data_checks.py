"""
Data-quality guardrails.
"""

from datetime import datetime, timezone

from .exceptions import GuardrailRejection

MAX_SPREAD = 0.02  # 2%
MAX_PRICE_JUMP = 0.10
STALE_THRESHOLD_SECONDS = 600


def check_price_validity(order):
    price = float(order.get("price", 0.0) or 0.0)
    if price <= 0:
        raise GuardrailRejection(f"{order.get('ticker')}: invalid price {price}.")

    if order.get("simulated"):
        return

    bid = order.get("bid")
    ask = order.get("ask")
    if bid and ask and bid > 0:
        spread = (ask - bid) / bid
        if spread > MAX_SPREAD:
            raise GuardrailRejection(
                f"{order.get('ticker')}: spread {spread:.2%} exceeds limit."
            )

    last_price = order.get("last_price")
    if last_price:
        jump = abs(price - last_price) / last_price
        if jump > MAX_PRICE_JUMP:
            raise GuardrailRejection(
                f"{order.get('ticker')}: price jump {jump:.2%} exceeds limit."
            )


def check_stale_data(order):
    if order.get("simulated"):
        return
    ts = order.get("timestamp")
    if not ts:
        return
    now = datetime.now(timezone.utc)
    if isinstance(ts, datetime):
        order_time = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    else:
        order_time = datetime.fromisoformat(str(ts))
    age = (now - order_time).total_seconds()
    if age > STALE_THRESHOLD_SECONDS:
        raise GuardrailRejection(
            f"{order.get('ticker')}: market data stale ({age:.0f}s old)."
        )
