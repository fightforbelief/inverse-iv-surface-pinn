"""Neural network architecture for approximating the IV surface."""

from typing import Any


class IVSurfaceModel:
    """Placeholder model definition."""

    def __init__(self, config: Any | None = None) -> None:
        self.config = config

    def forward(self, strikes: Any, maturities: Any) -> Any:
        """Compute implied volatilities for given strikes and maturities."""
        raise NotImplementedError("Implement forward pass")

    def parameters(self) -> list[Any]:
        """Return model parameters for optimization."""
        raise NotImplementedError("Expose model parameters")
