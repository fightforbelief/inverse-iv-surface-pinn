import numpy as np
import random
import torch
import os
import json
from typing import Any, Dict


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_config(config: Dict[str, Any], path: str):
    """
    Save configuration dictionary to JSON file.
    
    Args:
        config: Configuration dictionary
        path: Output file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        path: Configuration file path
    
    Returns:
        Configuration dictionary
    """
    with open(path, 'r') as f:
        config = json.load(f)
    return config


def ensure_dir(directory: str):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True)


def linear_interpolation(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """
    Linear interpolation wrapper.
    
    Args:
        x: x-coordinates at which to evaluate
        xp: x-coordinates of data points
        fp: y-coordinates of data points
    
    Returns:
        Interpolated values
    """
    return np.interp(x, xp, fp)


def compute_relative_error(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Compute relative error.
    
    Args:
        predicted: Predicted values
        actual: Actual values
    
    Returns:
        Relative errors
    """
    return np.abs(predicted - actual) / (np.abs(actual) + 1e-8)


def compute_metrics(predicted: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    """
    Compute various error metrics.
    
    Args:
        predicted: Predicted values
        actual: Actual values
    
    Returns:
        Dictionary of metrics
    """
    mse = np.mean((predicted - actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predicted - actual))
    mape = np.mean(compute_relative_error(predicted, actual)) * 100
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape)
    }
