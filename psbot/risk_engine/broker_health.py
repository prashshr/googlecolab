"""
Broker/API health checks.
"""

from collections import deque
from datetime import datetime, timedelta

from .exceptions import GuardrailRejection

ERROR_LIMIT = 3
WINDOW = timedelta(seconds=60)
_error_timestamps = deque()


def report_error():
    """External hook to report broker/API errors."""
    now = datetime.utcnow()
    _error_timestamps.append(now)
    _prune(now)


def _prune(now):
    while _error_timestamps and now - _error_timestamps[0] > WINDOW:
        _error_timestamps.popleft()


def check_api_stability(order=None):
    if order and order.get("simulated"):
        return
    now = datetime.utcnow()
    _prune(now)
    if len(_error_timestamps) >= ERROR_LIMIT:
        raise GuardrailRejection("Broker/API unstable: error threshold exceeded.")
