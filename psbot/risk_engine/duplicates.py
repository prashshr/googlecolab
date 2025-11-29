"""
Duplicate-order prevention.
"""

from collections import deque
from datetime import datetime, timedelta

from .exceptions import GuardrailRejection

RECENT_ORDER_WINDOW = timedelta(seconds=0)
_recent_orders = deque()


def _prune(now):
    while _recent_orders and now - _recent_orders[0][0] > RECENT_ORDER_WINDOW:
        _recent_orders.popleft()


def check_duplicate(order):
    now = datetime.utcnow()
    key = (order.get("ticker"), order.get("side"), order.get("reason"))
    _prune(now)
    for ts, existing_key in _recent_orders:
        if existing_key == key:
            raise GuardrailRejection(
                f"{order.get('ticker')}: duplicate {order.get('side')} ({order.get('reason')})."
            )


def record_order(order):
    now = datetime.utcnow()
    key = (order.get("ticker"), order.get("side"), order.get("reason"))
    _recent_orders.append((now, key))
    _prune(now)
