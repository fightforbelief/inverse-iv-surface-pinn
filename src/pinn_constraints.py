"""No-arbitrage and PINN-style constraint penalties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from black_scholes import bs_price


@dataclass
class ConstraintConfig:
    """Hyperparameters controlling constraint strength."""

    strike_bucket: float = 0.5  # bucket width for calendar check
    maturity_bucket_days: int = 1  # rounding granularity (days) for butterfly
    min_points_per_bucket: int = 3
    lambda_calendar: float = 1.0
    lambda_butterfly: float = 1.0


def _bucketize_strikes(strikes: torch.Tensor, bucket: float) -> torch.Tensor:
    """Round strikes into coarse buckets to align maturities for monotonicity."""
    return torch.round(strikes / bucket) * bucket


def calendar_arbitrage_penalty(
    *,
    sigma: torch.Tensor,
    strikes: torch.Tensor,
    maturities: torch.Tensor,
    bucket_width: float = 0.5,
) -> torch.Tensor:
    """
    Penalize decreasing total variance across maturities for similar strikes.
    Uses grouped strikes (bucketized) to compare total variance monotonicity.
    """
    device = sigma.device
    strikes_b = _bucketize_strikes(strikes, bucket_width)
    penalty_terms = []

    for k_val in strikes_b.unique():
        mask = strikes_b == k_val
        if mask.sum() < 2:
            continue
        t = maturities[mask]
        w = sigma[mask] ** 2 * t
        sorted_t, idx = torch.sort(t)
        sorted_w = w[idx]
        diffs = sorted_w[:-1] - sorted_w[1:]  # should be <= 0
        penalty_terms.append(F.relu(diffs))

    if len(penalty_terms) == 0:
        return torch.tensor(0.0, device=device)

    return torch.cat(penalty_terms).mean()


def butterfly_arbitrage_penalty(
    *,
    sigma: torch.Tensor,
    strikes: torch.Tensor,
    maturities: torch.Tensor,
    rates: torch.Tensor,
    dividends: torch.Tensor,
    spots: torch.Tensor,
    cp_flags: torch.Tensor,
    maturity_bucket_days: int = 1,
    min_points: int = 3,
) -> torch.Tensor:
    """
    Penalize negative convexity of option prices vs strike for each maturity bucket.

    Uses discrete second differences of Black–Scholes prices with model sigmas.
    """
    device = sigma.device

    # Align maturities into buckets (in days)
    days = torch.round(maturities * 365.0 / maturity_bucket_days) * maturity_bucket_days

    penalty_terms = []

    for d in days.unique():
        mask = days == d
        if mask.sum() < min_points:
            continue

        K = strikes[mask]
        idx = torch.argsort(K)
        K_sorted = K[idx]

        # Compute prices as calls
        sigma_sorted = sigma[mask][idx]
        T_sorted = maturities[mask][idx]
        r_sorted = rates[mask][idx]
        q_sorted = dividends[mask][idx]
        S_sorted = spots[mask][idx]

        prices_call = bs_price(
            spot=S_sorted,
            strike=K_sorted,
            time_to_maturity=T_sorted,
            rate=r_sorted,
            dividend=q_sorted,
            sigma=sigma_sorted,
            option_type=torch.ones_like(K_sorted),  # call
        )

        if len(prices_call) < min_points:
            continue

        # Finite difference second derivative for convexity
        K1, K2, K3 = K_sorted[:-2], K_sorted[1:-1], K_sorted[2:]
        C1, C2, C3 = prices_call[:-2], prices_call[1:-1], prices_call[2:]

        d1 = (C2 - C1) / (K2 - K1 + 1e-8)
        d2 = (C3 - C2) / (K3 - K2 + 1e-8)
        second_diff = d2 - d1
        penalty_terms.append(F.relu(-second_diff))  # convexity: second_diff >= 0

    if len(penalty_terms) == 0:
        return torch.tensor(0.0, device=device)

    return torch.cat(penalty_terms).mean()


def compute_constraint_loss(
    *,
    sigma: torch.Tensor,
    strikes: torch.Tensor,
    maturities: torch.Tensor,
    rates: torch.Tensor,
    dividends: torch.Tensor,
    spots: torch.Tensor,
    cp_flags: torch.Tensor,
    config: ConstraintConfig,
) -> torch.Tensor:
    """Aggregate calendar and butterfly penalties with weights."""
    cal_pen = calendar_arbitrage_penalty(
        sigma=sigma,
        strikes=strikes,
        maturities=maturities,
        bucket_width=config.strike_bucket,
    )
    fly_pen = butterfly_arbitrage_penalty(
        sigma=sigma,
        strikes=strikes,
        maturities=maturities,
        rates=rates,
        dividends=dividends,
        spots=spots,
        cp_flags=cp_flags,
        maturity_bucket_days=config.maturity_bucket_days,
        min_points=config.min_points_per_bucket,
    )

    return config.lambda_calendar * cal_pen + config.lambda_butterfly * fly_pen, {
        "calendar_penalty": cal_pen.detach(),
        "butterfly_penalty": fly_pen.detach(),
    }
