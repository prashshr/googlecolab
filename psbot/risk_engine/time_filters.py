"""
Session-based filters (avoid open/close, etc.).
"""

from datetime import time, datetime, timedelta

from .exceptions import GuardrailRejection

OPEN_BUFFER_MINUTES = 5
CLOSE_BUFFER_MINUTES = 10


def _is_within_block(order_time, open_time, close_time, block_start, block_end):
    return block_start <= order_time <= block_end


def check_market_session(order):
    if order.get("simulated"):
        return

    ts = order.get("timestamp")
    if not isinstance(ts, datetime):
        return

    local_time = ts.time()
    # Generic US session 09:30 - 16:00
    open_time = time(9, 30)
    close_time = time(16, 0)

    open_dt = ts.replace(hour=open_time.hour, minute=open_time.minute, second=0, microsecond=0)
    open_block_end = (open_dt + timedelta(minutes=OPEN_BUFFER_MINUTES)).time()
    close_dt = ts.replace(hour=close_time.hour, minute=close_time.minute, second=0, microsecond=0)
    close_block_start = (close_dt - timedelta(minutes=CLOSE_BUFFER_MINUTES)).time()

    if local_time < open_time or local_time > close_time:
        raise GuardrailRejection("Market closed.")

    if local_time <= open_block_end:
        raise GuardrailRejection("Blocked during opening buffer window.")

    if local_time >= close_block_start:
        raise GuardrailRejection("Blocked during closing buffer window.")
