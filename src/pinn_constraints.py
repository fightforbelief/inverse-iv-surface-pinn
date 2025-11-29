"""Constraints for physics-informed and no-arbitrage penalties."""

from typing import Any


def bs_pde_residual(model_outputs: Any, strikes: Any, maturities: Any) -> Any:
    """Compute residuals of the Black–Scholes PDE for the PINN."""
    raise NotImplementedError("Implement PDE residual calculation")


def no_arbitrage_penalty(model_outputs: Any) -> Any:
    """Evaluate no-arbitrage conditions such as convexity and monotonicity."""
    raise NotImplementedError("Implement no-arbitrage checks")
