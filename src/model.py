"""Neural network architectures for approximating the IV surface σ(K, T).

This module provides a configurable MLP that maps two inputs
(e.g., log-moneyness and time-to-maturity) to a positive implied
volatility value. The design favors smoothness (SILU/GELU),
optionally applies dropout for mild regularization, and clamps the
output to stay positive. The class is torch.nn.Module-compatible so
it can be used directly in PINN-style training loops with autograd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    """Return an activation module by name."""
    name = name.lower()
    if name in {"relu"}:
        return nn.ReLU()
    if name in {"gelu"}:
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name in {"tanh"}:
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


@dataclass
class IVSurfaceModelConfig:
    """
    Configuration for the IV surface network.

    Args:
        input_dim: Number of input features (default 2: log-moneyness, T).
        hidden_dims: Sizes of hidden layers.
        activation: Hidden activation function name.
        dropout: Dropout probability applied after each hidden layer.
        output_activation: Activation applied to final output.
        min_sigma: Floor for sigma to keep it strictly positive.
        max_sigma: Optional cap for sigma to avoid exploding values.
        weight_init: Weight init strategy ('xavier' or 'kaiming').
    """

    input_dim: int = 2
    hidden_dims: Sequence[int] = field(default_factory=lambda: (128, 128, 64))
    activation: str = "silu"
    dropout: float = 0.0
    output_activation: str = "softplus"
    min_sigma: float = 1e-4
    max_sigma: Optional[float] = None
    weight_init: str = "xavier"


class IVSurfaceMLP(nn.Module):
    """Feedforward network approximating σ(K, T)."""

    def __init__(self, config: Optional[IVSurfaceModelConfig] = None) -> None:
        super().__init__()
        self.config = config or IVSurfaceModelConfig()

        layers: List[nn.Module] = []
        input_dim = self.config.input_dim

        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(_get_activation(self.config.activation))
            if self.config.dropout and self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

        if self.config.output_activation.lower() not in {"softplus", "exp"}:
            raise ValueError("output_activation must be one of: 'softplus', 'exp'")

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for better training stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.config.weight_init == "kaiming":
                    nn.init.kaiming_uniform_(module.weight, a=0.2, nonlinearity="leaky_relu")
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, 2) containing normalized inputs.

        Returns:
            Tensor of shape (batch, 1) with positive implied volatilities.
        """
        raw_out = self.net(x)

        if self.config.output_activation == "softplus":
            sigma = torch.nn.functional.softplus(raw_out) + self.config.min_sigma
        else:  # "exp"
            sigma = torch.exp(raw_out) + self.config.min_sigma

        if self.config.max_sigma is not None:
            sigma = torch.clamp(sigma, max=self.config.max_sigma)

        return sigma

    @torch.no_grad()
    def predict_sigma(self, log_moneyness: torch.Tensor, maturity: torch.Tensor) -> torch.Tensor:
        """
        Convenience prediction wrapper using separate input tensors.

        Args:
            log_moneyness: Tensor of shape (batch,) or (batch, 1).
            maturity: Tensor of shape (batch,) or (batch, 1).
        """
        x = torch.stack((log_moneyness.flatten(), maturity.flatten()), dim=-1)
        return self.forward(x)

    def parameter_count(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
