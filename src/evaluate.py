"""Evaluation utilities for trained IV surface models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from black_scholes import bs_price
from visualize import (
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
)


@dataclass
class EvalConfig:
    """Configuration for evaluation routines."""

    input_cols: Sequence[str] = ("log_moneyness", "T")
    target_col: str = "mid_price"
    device: Optional[str] = None
    moneyness_tolerance: float = 0.02
    maturity_tolerance_days: int = 2


def _to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    return t.to(device) if isinstance(t, torch.Tensor) else torch.as_tensor(t, device=device)


def predict_prices_and_sigma(
    model: torch.nn.Module,
    df: pd.DataFrame,
    config: EvalConfig = EvalConfig(),
) -> pd.DataFrame:
    """Return a copy of df with added sigma_pred and price_pred columns."""
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()

    # Ensure numeric inputs
    df_num = df.copy()
    needed_cols = list(config.input_cols) + ["K", "T", "r", "q", "S", config.target_col]
    for c in needed_cols:
        if c in df_num.columns:
            df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    df_num = df_num.dropna(subset=list(config.input_cols) + ["K", "T", "r", config.target_col])

    x = torch.tensor(df_num[list(config.input_cols)].to_numpy(dtype=np.float32), device=device)
    K = torch.tensor(df_num["K"].to_numpy(dtype=np.float32), device=device)
    T = torch.tensor(df_num["T"].to_numpy(dtype=np.float32), device=device)
    r = torch.tensor(df_num["r"].to_numpy(dtype=np.float32), device=device)
    q = torch.tensor(df_num["q"].to_numpy(dtype=np.float32)) if "q" in df_num else torch.zeros(len(df_num), device=device)
    cp_map = {"C": 1.0, "P": -1.0}
    cp = torch.tensor([cp_map.get(v, 1.0) for v in df_num["cp_flag"]], dtype=torch.float32, device=device)
    S = torch.tensor(df_num["S"].to_numpy(dtype=np.float32), device=device) if "S" in df_num else K / torch.exp(x[:, 0])

    with torch.no_grad():
        sigma_pred = model(x).squeeze(-1)
        price_pred = bs_price(
            spot=S,
            strike=K,
            time_to_maturity=T,
            rate=r,
            dividend=q,
            sigma=sigma_pred,
            option_type=cp,
        ).squeeze(-1)

    df_out = df_num.copy()
    df_out["sigma_pred"] = sigma_pred.cpu().numpy()
    df_out["price_pred"] = price_pred.cpu().numpy()
    return df_out


def compute_error_metrics(df_pred: pd.DataFrame, *, target_col: str = "mid_price") -> Dict[str, float]:
    """Compute MAE/MSE/RMSE/MAPE between predicted price and target."""
    y_true = df_pred[target_col].to_numpy(dtype=np.float64)
    y_pred = df_pred["price_pred"].to_numpy(dtype=np.float64)

    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    mape = float(np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8))) * 100

    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape}


def no_arbitrage_summary(
    df_pred: pd.DataFrame,
    *,
    moneyness_tolerance: float = 0.02,
    maturity_tolerance_days: int = 2,
) -> Dict[str, float]:
    """Summarize calendar and butterfly arbitrage violations."""
    cal = check_calendar_arbitrage(df_pred, iv_col="sigma_pred", moneyness_tolerance=moneyness_tolerance)
    fly = check_butterfly_arbitrage(df_pred, price_col="price_pred", maturity_tolerance=maturity_tolerance_days)

    total_rows = len(df_pred)
    cal_count = len(cal)
    fly_count = len(fly)

    return {
        "calendar_violations": float(cal_count),
        "calendar_violation_ratio": float(cal_count) / max(total_rows, 1),
        "butterfly_violations": float(fly_count),
        "butterfly_violation_ratio": float(fly_count) / max(total_rows, 1),
    }


def evaluate_model(
    model: torch.nn.Module,
    df: pd.DataFrame,
    config: EvalConfig = EvalConfig(),
) -> Dict[str, float]:
    """
    High-level evaluation: predict sigma/price, compute error metrics and arbitrage stats.

    Returns a merged metrics dictionary.
    """
    df_pred = predict_prices_and_sigma(model, df, config=config)
    error_metrics = compute_error_metrics(df_pred, target_col=config.target_col)
    arb_metrics = no_arbitrage_summary(
        df_pred,
        moneyness_tolerance=config.moneyness_tolerance,
        maturity_tolerance_days=config.maturity_tolerance_days,
    )
    return {**error_metrics, **arb_metrics}
