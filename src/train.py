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
    lr_step_size: int = 0  # 0 disables StepLR
    lr_gamma: float = 0.5
    weight_decay: float = 0.0
    batch_size: int = 256
    log_every: int = 10
    val_every: int = 10
    early_stop_patience: int = 0  # 0 disables early stopping
    early_stop_min_delta: float = 1e-4
    checkpoint_path: Optional[str] = None
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
    val_data: Optional[Dataset] = None,
) -> Dict[str, float]:
    """
    Price-fitting training loop with optional no-arbitrage constraints, validation,
    LR scheduling, early stopping, and checkpointing.

    Returns:
        A metrics dictionary with final training/constraint losses (and validation if provided).
    """
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size) if val_data is not None else None

    scheduler = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
        if config.lr_step_size and config.lr_step_size > 0
        else None
    )

    constraint_cfg = ConstraintConfig(
        strike_bucket=config.strike_bucket,
        maturity_bucket_days=config.maturity_bucket_days,
        min_points_per_bucket=config.min_points_per_bucket,
        lambda_calendar=config.lambda_calendar,
        lambda_butterfly=config.lambda_butterfly,
    )

    best_val = float("inf")
    patience_counter = 0
    best_state = None

    def _eval_loader(dl: DataLoader) -> Dict[str, float]:
        return evaluate(model, dl, device=device, constraint_cfg=constraint_cfg, lambda_calendar=config.lambda_calendar, lambda_butterfly=config.lambda_butterfly)

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
        if scheduler is not None:
            scheduler.step()

        # Logging
        do_log = (epoch + 1) % config.log_every == 0 or epoch == 0
        val_metrics = None
        if val_loader is not None and ((epoch + 1) % config.val_every == 0 or epoch == 0):
            val_metrics = _eval_loader(val_loader)

        if do_log:
            msg = (
                f"[Epoch {epoch+1:04d}] "
                f"train_loss={avg_loss:.6f} "
                f"price={avg_price:.6f} "
                f"constraint={avg_constraint:.6f}"
            )
            if val_metrics:
                msg += f" | val_loss={val_metrics['loss']:.6f} price={val_metrics['price_loss']:.6f} constraint={val_metrics['constraint_loss']:.6f}"
            print(msg)

        # Early stopping on validation loss
        if val_metrics is not None:
            current_val = val_metrics["loss"]
            if current_val + config.early_stop_min_delta < best_val:
                best_val = current_val
                patience_counter = 0
                best_state = model.state_dict()
                if config.checkpoint_path:
                    save_checkpoint(model, config.checkpoint_path)
            else:
                patience_counter += 1
                if config.early_stop_patience and patience_counter >= config.early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1} (best val_loss={best_val:.6f})")
                    break

    # Restore best state if captured
    if best_state is not None:
        model.load_state_dict(best_state)

    result = {"train_loss": avg_loss, "price_loss": avg_price, "constraint_loss": avg_constraint}
    if val_metrics is not None:
        result.update({f"val_{k}": v for k, v in val_metrics.items()})
    return result


def save_checkpoint(model: nn.Module, path: str) -> None:
    """Persist model weights."""
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: str, *, map_location: Optional[str] = None) -> nn.Module:
    """Load weights into an existing model instance."""
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    device: torch.device,
    constraint_cfg: ConstraintConfig,
    lambda_calendar: float,
    lambda_butterfly: float,
) -> Dict[str, float]:
    """Evaluate losses on a dataset (price + constraints)."""
    model.eval()
    total = 0
    total_price = 0.0
    total_constraint = 0.0
    total_samples = 0

    for batch in data_loader:
        sigma_pred, pred_price, target_price, S, K, T, r, q, cp = forward_batch(model, batch, device=device)
        price_loss = F.mse_loss(pred_price, target_price, reduction="sum")
        constraint_loss = torch.tensor(0.0, device=device)
        if lambda_calendar > 0 or lambda_butterfly > 0:
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
            constraint_loss = constraint_loss * K.shape[0]  # sum-style

        total += (price_loss + constraint_loss).item()
        total_price += price_loss.item()
        total_constraint += constraint_loss.item()
        total_samples += K.shape[0]

    return {
        "loss": total / max(total_samples, 1),
        "price_loss": total_price / max(total_samples, 1),
        "constraint_loss": total_constraint / max(total_samples, 1),
    }
