import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional


def load_raw_data(
    raw_data_path: str,
    ticker_name: str = "SPY",
    date: str = "2023-01-03"
) -> Dict[str, pd.DataFrame]:
    """
    Load raw option and security data from parquet files.
    
    Args:
        raw_data_path: Path to raw data directory
        ticker_name: Ticker symbol
        date: Date in format 'YYYY-MM-DD'
    
    Returns:
        Dictionary containing all loaded dataframes
    """
    date_name = date.replace("-", "")
    
    data = {}
    
    # Load option prices
    op_path = os.path.join(raw_data_path, f"{ticker_name}{date_name}_option_price.parquet")
    if os.path.exists(op_path):
        data['option_price'] = pd.read_parquet(op_path)
    
    # Load security prices
    sp_path = os.path.join(raw_data_path, f"{ticker_name}{date_name}_security_price.parquet")
    if os.path.exists(sp_path):
        data['security_price'] = pd.read_parquet(sp_path)
    
    # Load volatility surface
    vs_path = os.path.join(raw_data_path, f"{ticker_name}{date_name}_volatility_surface.parquet")
    if os.path.exists(vs_path):
        data['volatility_surface'] = pd.read_parquet(vs_path)
    
    # Load standard option prices
    sop_path = os.path.join(raw_data_path, f"{ticker_name}{date_name}_stdoption_price.parquet")
    if os.path.exists(sop_path):
        data['stdoption_price'] = pd.read_parquet(sop_path)
    
    # Load distribution projection
    dp_path = os.path.join(raw_data_path, f"{ticker_name}{date_name}_distr_proj.parquet")
    if os.path.exists(dp_path):
        data['distr_proj'] = pd.read_parquet(dp_path)
    
    # Load zero curve
    zc_path = os.path.join(raw_data_path, f"{date_name}_zero_curve.parquet")
    if os.path.exists(zc_path):
        data['zero_curve'] = pd.read_parquet(zc_path)
    
    # Load security descriptor
    sc_path = os.path.join(raw_data_path, f"{ticker_name}_securd.parquet")
    if os.path.exists(sc_path):
        data['securd'] = pd.read_parquet(sc_path)
    
    return data


def get_spot_price(security_df: pd.DataFrame) -> float:
    """
    Extract spot price from security price dataframe.
    
    Args:
        security_df: Security price dataframe
    
    Returns:
        Spot price (close price)
    """
    if len(security_df) == 0:
        raise ValueError("Security price dataframe is empty")
    return float(security_df['close'].iloc[0])


def get_risk_free_rate(zero_curve_df: pd.DataFrame, days: float) -> float:
    """
    Interpolate risk-free rate for given days to maturity.
    
    Args:
        zero_curve_df: Zero curve dataframe with 'days' and 'rate' columns
        days: Days to maturity
    
    Returns:
        Interpolated annual risk-free rate (in decimal, e.g., 0.04 for 4%)
    """
    if len(zero_curve_df) == 0:
        raise ValueError("Zero curve dataframe is empty")
    
    # Convert rate from percentage to decimal
    zero_curve_df = zero_curve_df.copy()
    zero_curve_df['rate'] = zero_curve_df['rate'] / 100.0
    
    # Sort by days
    zero_curve_df = zero_curve_df.sort_values('days')
    
    # Interpolate
    rate = np.interp(days, zero_curve_df['days'].values, zero_curve_df['rate'].values)
    
    return float(rate)


def get_dividend_yield(distr_proj_df: pd.DataFrame, reference_date: str, maturity_date: str) -> float:
    """
    Calculate annualized dividend yield from distribution projection.
    
    Args:
        distr_proj_df: Distribution projection dataframe
        reference_date: Current date string 'YYYY-MM-DD'
        maturity_date: Option maturity date string 'YYYY-MM-DD'
    
    Returns:
        Annualized dividend yield (in decimal)
    """
    if len(distr_proj_df) == 0:
        return 0.0
    
    ref_dt = pd.to_datetime(reference_date)
    mat_dt = pd.to_datetime(maturity_date)
    
    # Filter dividends between reference and maturity
    distr_proj_df = distr_proj_df.copy()
    distr_proj_df['exdate'] = pd.to_datetime(distr_proj_df['exdate'])
    
    relevant_divs = distr_proj_df[
        (distr_proj_df['exdate'] > ref_dt) & 
        (distr_proj_df['exdate'] <= mat_dt)
    ]
    
    if len(relevant_divs) == 0:
        return 0.0
    
    # Sum dividends and annualize
    total_div = relevant_divs['amount'].sum()
    days_to_mat = (mat_dt - ref_dt).days
    
    if days_to_mat <= 0:
        return 0.0
    
    # Annualize dividend yield
    annualized_yield = total_div * (365.0 / days_to_mat)
    
    return float(annualized_yield)


def prepare_option_data(
    option_df: pd.DataFrame,
    spot_price: float,
    date: str,
    zero_curve_df: pd.DataFrame,
    distr_proj_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Prepare option data by adding necessary features for modeling.
    
    Args:
        option_df: Option price dataframe
        spot_price: Current spot price
        date: Current date 'YYYY-MM-DD'
        zero_curve_df: Zero curve dataframe
        distr_proj_df: Distribution projection dataframe (optional)
    
    Returns:
        Dataframe with additional features: S, K, T, r, q, mid_price, moneyness
    """
    df = option_df.copy()
    
    # Convert dates to datetime
    df['date'] = pd.to_datetime(df['date'])
    df['exdate'] = pd.to_datetime(df['exdate'])
    
    # Calculate days to maturity and years to maturity
    df['days_to_maturity'] = (df['exdate'] - df['date']).dt.days
    df['T'] = df['days_to_maturity'] / 365.0
    
    # Add spot price
    df['S'] = spot_price
    
    # Strike price (convert from cents to dollars)
    df['K'] = df['strike_price'] / 1000.0
    
    # Calculate mid price
    df['mid_price'] = (df['best_bid'] + df['best_offer']) / 2.0
    
    # Get risk-free rate for each option
    df['r'] = df['days_to_maturity'].apply(
        lambda days: get_risk_free_rate(zero_curve_df, days)
    )
    
    # Get dividend yield for each option
    if distr_proj_df is not None and len(distr_proj_df) > 0:
        df['q'] = df.apply(
            lambda row: get_dividend_yield(
                distr_proj_df, 
                row['date'].strftime('%Y-%m-%d'), 
                row['exdate'].strftime('%Y-%m-%d')
            ),
            axis=1
        )
    else:
        df['q'] = 0.0
    
    # Calculate moneyness
    df['moneyness'] = df['K'] / df['S']
    df['log_moneyness'] = np.log(df['moneyness'])
    
    return df


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets.
    
    Args:
        df: Dataframe to split
        train_ratio: Ratio of training data
        random_state: Random seed
    
    Returns:
        Training and testing dataframes
    """
    np.random.seed(random_state)
    
    # Shuffle indices
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    
    # Split
    n_train = int(len(df) * train_ratio)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    
    return train_df, test_df
