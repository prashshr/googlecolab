"""
Shared exceptions for the risk engine.
"""


class GuardrailRejection(Exception):
    """Raised when a guardrail blocks an order."""
