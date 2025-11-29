import numpy as np
import pandas as pd
from scipy.optimize import brentq
from typing import Tuple, Optional
import warnings


def clean_data(df: pd.DataFrame, min_price: float = 0.01, max_price: Optional[float] = None) -> pd.DataFrame:
    """
    Clean option data by removing invalid entries.
    
    Args:
        df: Option dataframe with required columns
        min_price: Minimum valid option price
        max_price: Maximum valid option price (optional)
    
    Returns:
        Cleaned dataframe
    """
    cleaned_df = df.copy()
    
    # Remove rows with missing critical values
    critical_cols = ['S', 'K', 'T', 'r', 'mid_price']
    cleaned_df = cleaned_df.dropna(subset=critical_cols)
    
    # Remove expired or invalid maturity options
    cleaned_df = cleaned_df[cleaned_df['T'] > 0]
    cleaned_df = cleaned_df[cleaned_df['days_to_maturity'] > 0]
    
    # Remove invalid strikes
    cleaned_df = cleaned_df[cleaned_df['K'] > 0]
    
    # Remove invalid prices
    cleaned_df = cleaned_df[cleaned_df['mid_price'] >= min_price]
    if max_price is not None:
        cleaned_df = cleaned_df[cleaned_df['mid_price'] <= max_price]
    
    # Remove invalid bid-offer spreads
    cleaned_df = cleaned_df[cleaned_df['best_bid'] > 0]
    cleaned_df = cleaned_df[cleaned_df['best_offer'] > cleaned_df['best_bid']]
    
    # Remove options with zero or negative volume/open interest (optional filter)
    # cleaned_df = cleaned_df[(cleaned_df['volume'] > 0) | (cleaned_df['open_interest'] > 0)]
    
    # Remove extreme moneyness (deep ITM or OTM)
    cleaned_df = cleaned_df[
        (cleaned_df['moneyness'] >= 0.7) & 
        (cleaned_df['moneyness'] <= 1.3)
    ]
    
    # Remove very short-dated options (< 7 days)
    cleaned_df = cleaned_df[cleaned_df['days_to_maturity'] >= 7]
    
    # Remove very long-dated options (> 2 years)
    cleaned_df = cleaned_df[cleaned_df['days_to_maturity'] <= 730]
    
    return cleaned_df.reset_index(drop=True)


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = 'C'
) -> float:
    """
    Calculate Black-Scholes option price.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility
        q: Dividend yield
        option_type: 'C' for call, 'P' for put
    
    Returns:
        Option price
    """
    from scipy.stats import norm
    
    if T <= 0 or sigma <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.upper() == 'C':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    return price


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = 'C',
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> Optional[float]:
    """
    Calculate implied volatility using Brent's method.
    
    Args:
        market_price: Observed market price
        S: Spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        q: Dividend yield
        option_type: 'C' for call, 'P' for put
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
    
    Returns:
        Implied volatility or None if calculation fails
    """
    if market_price <= 0 or T <= 0:
        return None
    
    # Check intrinsic value
    if option_type.upper() == 'C':
        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    
    if market_price < intrinsic:
        return None
    
    # Define objective function
    def objective(sigma):
        if sigma <= 0:
            return market_price
        try:
            bs_price = black_scholes_price(S, K, T, r, sigma, q, option_type)
            return bs_price - market_price
        except:
            return market_price
    
    try:
        # Find IV using Brent's method
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iv = brentq(objective, 0.001, 5.0, maxiter=max_iterations, xtol=tolerance)
        return iv
    except:
        return None


def compute_implied_volatilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute implied volatility for each option in the dataframe.
    
    Args:
        df: Option dataframe with S, K, T, r, q, mid_price, cp_flag columns
    
    Returns:
        Dataframe with additional 'computed_iv' column
    """
    result_df = df.copy()
    
    ivs = []
    for idx, row in result_df.iterrows():
        iv = implied_volatility(
            market_price=row['mid_price'],
            S=row['S'],
            K=row['K'],
            T=row['T'],
            r=row['r'],
            q=row['q'],
            option_type=row['cp_flag']
        )
        ivs.append(iv)
    
    result_df['computed_iv'] = ivs
    
    # Remove options where IV computation failed
    result_df = result_df.dropna(subset=['computed_iv'])
    
    # Filter unreasonable IVs
    result_df = result_df[
        (result_df['computed_iv'] > 0.01) & 
        (result_df['computed_iv'] < 2.0)
    ]
    
    return result_df.reset_index(drop=True)


def normalize_features(
    df: pd.DataFrame,
    feature_cols: list,
    method: str = 'standard'
) -> Tuple[pd.DataFrame, dict]:
    """
    Normalize features for model input.
    
    Args:
        df: Dataframe with features
        feature_cols: List of feature column names
        method: 'standard' for standardization or 'minmax' for min-max scaling
    
    Returns:
        Normalized dataframe and normalization parameters dictionary
    """
    normalized_df = df.copy()
    params = {}
    
    for col in feature_cols:
        if col not in df.columns:
            continue
            
        if method == 'standard':
            mean = df[col].mean()
            std = df[col].std()
            normalized_df[col] = (df[col] - mean) / (std + 1e-8)
            params[col] = {'mean': mean, 'std': std, 'method': 'standard'}
        elif method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            normalized_df[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)
            params[col] = {'min': min_val, 'max': max_val, 'method': 'minmax'}
    
    return normalized_df, params


def denormalize_features(
    values: np.ndarray,
    col_name: str,
    params: dict
) -> np.ndarray:
    """
    Denormalize features using stored parameters.
    
    Args:
        values: Normalized values
        col_name: Feature column name
        params: Normalization parameters dictionary
    
    Returns:
        Denormalized values
    """
    if col_name not in params:
        return values
    
    col_params = params[col_name]
    
    if col_params['method'] == 'standard':
        return values * col_params['std'] + col_params['mean']
    elif col_params['method'] == 'minmax':
        return values * (col_params['max'] - col_params['min']) + col_params['min']
    
    return values


def create_model_dataset(
    df: pd.DataFrame,
    input_features: list = ['log_moneyness', 'T'],
    target: str = 'computed_iv',
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
    """
    Create input-output arrays for model training.
    
    Args:
        df: Prepared dataframe
        input_features: List of input feature column names
        target: Target column name
        normalize: Whether to normalize features
    
    Returns:
        X (inputs), y (targets), and normalization parameters
    """
    # Extract features and target
    X = df[input_features].values
    y = df[target].values
    
    norm_params = None
    
    if normalize:
        # Normalize inputs
        df_normalized, norm_params = normalize_features(df, input_features, method='standard')
        X = df_normalized[input_features].values
    
    return X, y, norm_params


def filter_by_liquidity(
    df: pd.DataFrame,
    min_volume: int = 10,
    min_open_interest: int = 100
) -> pd.DataFrame:
    """
    Filter options by liquidity criteria.
    
    Args:
        df: Option dataframe
        min_volume: Minimum volume
        min_open_interest: Minimum open interest
    
    Returns:
        Filtered dataframe
    """
    filtered_df = df.copy()
    
    if 'volume' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['volume'] >= min_volume) | 
            (filtered_df['open_interest'] >= min_open_interest)
        ]
    
    return filtered_df.reset_index(drop=True)


def add_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add additional engineered features.
    
    Args:
        df: Option dataframe
    
    Returns:
        Dataframe with additional features
    """
    result_df = df.copy()
    
    # Forward moneyness
    result_df['forward_moneyness'] = result_df['K'] / (
        result_df['S'] * np.exp((result_df['r'] - result_df['q']) * result_df['T'])
    )
    
    # Time to maturity in different scales
    result_df['sqrt_T'] = np.sqrt(result_df['T'])
    result_df['log_T'] = np.log(result_df['T'] + 1e-8)
    
    # Moneyness categories
    result_df['is_atm'] = (
        (result_df['moneyness'] >= 0.95) & 
        (result_df['moneyness'] <= 1.05)
    ).astype(int)
    
    result_df['is_otm'] = (
        ((result_df['cp_flag'] == 'C') & (result_df['moneyness'] > 1.05)) |
        ((result_df['cp_flag'] == 'P') & (result_df['moneyness'] < 0.95))
    ).astype(int)
    
    result_df['is_itm'] = (
        ((result_df['cp_flag'] == 'C') & (result_df['moneyness'] < 0.95)) |
        ((result_df['cp_flag'] == 'P') & (result_df['moneyness'] > 1.05))
    ).astype(int)
    
    return result_df
