"""Black–Scholes pricing utilities (torch-friendly)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F
from scipy import optimize, stats

TensorLike = Union[torch.Tensor, float]


@dataclass
class BlackScholesInputs:
    spot: TensorLike
    strike: TensorLike
    time_to_maturity: TensorLike
    rate: TensorLike
    dividend: TensorLike = 0.0
    option_type: Union[str, torch.Tensor] = "C"  # 'C'/'P' or tensor(+1 call, -1 put)
    sigma: Optional[TensorLike] = None


def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF implemented with erf for autograd compatibility."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def validate_inputs(params: BlackScholesInputs) -> None:
    """Basic validation to avoid invalid pricing calls."""
    if isinstance(params.time_to_maturity, torch.Tensor):
        if torch.any(params.time_to_maturity < 0):
            raise ValueError("time_to_maturity must be non-negative")
    else:
        if params.time_to_maturity < 0:
            raise ValueError("time_to_maturity must be non-negative")

    if isinstance(params.strike, torch.Tensor):
        if torch.any(params.strike <= 0):
            raise ValueError("strike must be positive")
    else:
        if params.strike <= 0:
            raise ValueError("strike must be positive")


def bs_price(
    *,
    spot: TensorLike,
    strike: TensorLike,
    time_to_maturity: TensorLike,
    rate: TensorLike,
    dividend: TensorLike = 0.0,
    sigma: TensorLike,
    option_type: Union[str, torch.Tensor] = "C",
) -> torch.Tensor:
    """
    Black–Scholes price for European options (torch-compatible).

    Args:
        spot: Spot price S
        strike: Strike price K
        time_to_maturity: Year fraction T
        rate: Risk-free rate r
        dividend: Continuous dividend yield q
        sigma: Implied volatility
        option_type: 'C'/'P' or tensor (+1 for call, -1 for put)
    """
    # Ensure tensors for autograd
    S = torch.as_tensor(spot, dtype=torch.float32)
    K = torch.as_tensor(strike, dtype=torch.float32)
    T = torch.as_tensor(time_to_maturity, dtype=torch.float32)
    r = torch.as_tensor(rate, dtype=torch.float32)
    q = torch.as_tensor(dividend, dtype=torch.float32)
    vol = torch.as_tensor(sigma, dtype=torch.float32)

    # Numerical safety
    eps = 1e-8
    T = torch.clamp(T, min=eps)
    vol = torch.clamp(vol, min=eps)

    d1 = (torch.log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * torch.sqrt(T))
    d2 = d1 - vol * torch.sqrt(T)

    call = S * torch.exp(-q * T) * _norm_cdf(d1) - K * torch.exp(-r * T) * _norm_cdf(d2)
    put = K * torch.exp(-r * T) * _norm_cdf(-d2) - S * torch.exp(-q * T) * _norm_cdf(-d1)

    if isinstance(option_type, torch.Tensor):
        cp_sign = torch.sign(option_type)  # +1 for call, -1 for put
        return torch.where(cp_sign >= 0, call, put)

    if str(option_type).upper() == "C":
        return call
    return put


def bs_price_numpy(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    rate: float,
    dividend: float = 0.0,
    sigma: float = 0.2,
    option_type: str = "C",
) -> float:
    """NumPy/float version (for root finding and quick checks)."""
    if time_to_maturity <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * sigma * sigma) * time_to_maturity) / (
        sigma * math.sqrt(time_to_maturity)
    )
    d2 = d1 - sigma * math.sqrt(time_to_maturity)
    if option_type.upper() == "C":
        return spot * math.exp(-dividend * time_to_maturity) * stats.norm.cdf(d1) - strike * math.exp(
            -rate * time_to_maturity
        ) * stats.norm.cdf(d2)
    return strike * math.exp(-rate * time_to_maturity) * stats.norm.cdf(-d2) - spot * math.exp(
        -dividend * time_to_maturity
    ) * stats.norm.cdf(-d1)


def implied_volatility(
    *,
    price: float,
    spot: float,
    strike: float,
    time_to_maturity: float,
    rate: float,
    dividend: float = 0.0,
    option_type: str = "C",
    tol: float = 1e-6,
    max_iter: int = 100,
    sigma_bounds: tuple[float, float] = (1e-4, 5.0),
) -> Optional[float]:
    """Solve for sigma via Brent's method; returns None if it fails."""
    if price <= 0 or time_to_maturity <= 0:
        return None

    def objective(sig: float) -> float:
        return bs_price_numpy(
            spot=spot,
            strike=strike,
            time_to_maturity=time_to_maturity,
            rate=rate,
            dividend=dividend,
            sigma=sig,
            option_type=option_type,
        ) - price

    try:
        return optimize.brentq(objective, sigma_bounds[0], sigma_bounds[1], xtol=tol, maxiter=max_iter)
    except Exception:
        return None
