"""Evaluation utilities for trained IV surface models."""

from typing import Any


def compute_metrics(model: Any, data: Any) -> dict[str, float]:
    """Calculate error metrics and diagnostics."""
    raise NotImplementedError("Implement evaluation metrics")


def check_no_arbitrage(model: Any, data: Any) -> Any:
    """Verify no-arbitrage conditions on model outputs."""
    raise NotImplementedError("Implement no-arbitrage verification")
