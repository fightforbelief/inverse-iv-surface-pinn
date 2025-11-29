"""Training loop for IV surface fitting using price MSE loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from black_scholes import bs_price
from pinn_constraints import ConstraintConfig, compute_constraint_loss


class IVPriceDataset(Dataset):
    """Torch dataset for price-fitting with predicted sigma."""

    def __init__(
        self,
        *,
        features: torch.Tensor,
        prices: torch.Tensor,
        strikes: torch.Tensor,
        maturities: torch.Tensor,
        rates: torch.Tensor,
        dividends: torch.Tensor,
        cp_flags: torch.Tensor,
        spots: Optional[torch.Tensor] = None,
    ) -> None:
        self.x = features.float()
        self.y = prices.float()
        self.K = strikes.float()
        self.T = maturities.float()
        self.r = rates.float()
        self.q = dividends.float()
        self.cp = cp_flags.float()
        self.S = spots.float() if spots is not None else None

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": self.x[idx],
            "price": self.y[idx],
            "K": self.K[idx],
            "T": self.T[idx],
            "r": self.r[idx],
            "q": self.q[idx],
            "cp": self.cp[idx],
            "S": self.S[idx] if self.S is not None else None,
        }


@dataclass
class TrainingConfig:
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    log_every: int = 10
    device: Optional[str] = None  # "cuda" / "cpu"
    target_col: str = "mid_price"
    input_cols: Sequence[str] = ("log_moneyness", "T")
    spot_constant: Optional[float] = None  # if not provided in dataset
    # Constraint weights and knobs
    lambda_calendar: float = 0.0
    lambda_butterfly: float = 0.0
    strike_bucket: float = 0.5
    maturity_bucket_days: int = 1
    min_points_per_bucket: int = 3


def build_dataset_from_df(df, config: TrainingConfig) -> IVPriceDataset:
    """Create a dataset from a pandas DataFrame."""
    import numpy as np  # local import to keep torch-only top-level

    features = torch.tensor(df[list(config.input_cols)].values, dtype=torch.float32)
    prices = torch.tensor(df[config.target_col].values, dtype=torch.float32)

    strikes = torch.tensor(df["K"].values, dtype=torch.float32)
    maturities = torch.tensor(df["T"].values, dtype=torch.float32)
    rates = torch.tensor(df["r"].values, dtype=torch.float32)
    dividends = torch.tensor(df["q"].values if "q" in df.columns else np.zeros(len(df)), dtype=torch.float32)

    cp_map = {"C": 1.0, "P": -1.0}
    cp_flags = torch.tensor([cp_map.get(val, 1.0) for val in df["cp_flag"]], dtype=torch.float32)

    spots = (
        torch.tensor(df["S"].values, dtype=torch.float32)
        if "S" in df.columns
        else torch.full((len(df),), float(config.spot_constant or 0.0))
    )

    return IVPriceDataset(
        features=features,
        prices=prices,
        strikes=strikes,
        maturities=maturities,
        rates=rates,
        dividends=dividends,
        cp_flags=cp_flags,
        spots=spots,
    )


def price_mse_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Compute price MSE between BS(model sigma) and market mid price."""
    _, pred_price, target_price, _, _, _, _, _, _ = forward_batch(model, batch, device=device)
    return F.mse_loss(pred_price, target_price)


def forward_batch(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run model and Black–Scholes pricing for a batch."""
    x = batch["x"].to(device)
    target_price = batch["price"].to(device)
    K = batch["K"].to(device)
    T = batch["T"].to(device)
    r = batch["r"].to(device)
    q = batch["q"].to(device)
    cp = batch["cp"].to(device)

    S_tensor = batch.get("S")
    S = S_tensor.to(device) if isinstance(S_tensor, torch.Tensor) else None

    if S is None:
        log_m = x[..., 0]
        S = K / torch.exp(log_m)

    sigma_pred = model(x).squeeze(-1)
    pred_price = bs_price(
        spot=S,
        strike=K,
        time_to_maturity=T,
        rate=r,
        dividend=q,
        sigma=sigma_pred,
        option_type=cp,
    ).squeeze(-1)

    return sigma_pred, pred_price, target_price, S, K, T, r, q, cp


def train(
    model: nn.Module,
    train_data: Dataset,
    *,
    config: TrainingConfig = TrainingConfig(),
) -> Dict[str, float]:
    """
    Price-fitting training loop with optional no-arbitrage constraints.

    Returns:
        A metrics dictionary with final training and constraint losses.
    """
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

    constraint_cfg = ConstraintConfig(
        strike_bucket=config.strike_bucket,
        maturity_bucket_days=config.maturity_bucket_days,
        min_points_per_bucket=config.min_points_per_bucket,
        lambda_calendar=config.lambda_calendar,
        lambda_butterfly=config.lambda_butterfly,
    )

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_price_loss = 0.0
        epoch_constraint = 0.0
        sample_count = 0
        for batch in loader:
            optimizer.zero_grad()
            sigma_pred, pred_price, target_price, S, K, T, r, q, cp = forward_batch(model, batch, device=device)
            price_loss = F.mse_loss(pred_price, target_price)

            constraint_loss = torch.tensor(0.0, device=device)
            if config.lambda_calendar > 0 or config.lambda_butterfly > 0:
                constraint_loss, _ = compute_constraint_loss(
                    sigma=sigma_pred,
                    strikes=K,
                    maturities=T,
                    rates=r,
                    dividends=q,
                    spots=S,
                    cp_flags=cp,
                    config=constraint_cfg,
                )

            loss = price_loss + constraint_loss
            loss.backward()
            optimizer.step()

            batch_size = batch["x"].shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_price_loss += price_loss.item() * batch_size
            epoch_constraint += constraint_loss.item() * batch_size
            sample_count += batch_size

        avg_loss = epoch_loss / max(sample_count, 1)
        avg_price = epoch_price_loss / max(sample_count, 1)
        avg_constraint = epoch_constraint / max(sample_count, 1)
        if (epoch + 1) % config.log_every == 0 or epoch == 0:
            print(
                f"[Epoch {epoch+1:04d}] "
                f"train_loss={avg_loss:.6f} "
                f"price={avg_price:.6f} "
                f"constraint={avg_constraint:.6f}"
            )

    return {"train_loss": avg_loss, "price_loss": avg_price, "constraint_loss": avg_constraint}


def save_checkpoint(model: nn.Module, path: str) -> None:
    """Persist model weights."""
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: str, *, map_location: Optional[str] = None) -> nn.Module:
    """Load weights into an existing model instance."""
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return model
