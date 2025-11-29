"""Training loop for the IV surface PINN."""

from typing import Any


def train(model: Any, data: Any, *, epochs: int = 1, learning_rate: float = 1e-3) -> None:
    """Run the training process with price and constraint losses."""
    raise NotImplementedError("Implement training loop")


def save_checkpoint(model: Any, path: str) -> None:
    """Persist model weights and training state."""
    raise NotImplementedError("Implement checkpoint saving")


def load_checkpoint(path: str) -> Any:
    """Load model weights and training state."""
    raise NotImplementedError("Implement checkpoint loading")
