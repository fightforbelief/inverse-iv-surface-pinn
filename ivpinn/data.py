"""
Data acquisition and preprocessing module for option market data.

This module provides interfaces to fetch option data from multiple sources:
- Yahoo Finance (yfinance) for quick prototyping
- WRDS OptionMetrics for high-quality institutional data

It also includes data cleaning, filtering, and transformation utilities.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from datetime import datetime, timedelta
import warnings


# ============================================================================
# WRDS OptionMetrics Interface
# ============================================================================

def fetch_optionmetrics_wrds(
    conn,
    date: str,
    symbol: str = "SPY",
    return_raw: bool = False
) -> pd.DataFrame:
    """
    Fetch option chain data from WRDS OptionMetrics database.
    
    This function queries the OptionMetrics database for a specific underlying
    symbol on a given date, retrieving all available option contracts.
    
    Parameters
    ----------
    conn : wrds.Connection
        Active WRDS database connection object
    date : str
        Date in format 'YYYY-MM-DD'
    symbol : str, optional
        Ticker symbol (default: "SPY")
    return_raw : bool, optional
        If True, return raw data without preprocessing (default: False)
    
    Returns
    -------
    pd.DataFrame
        Option chain data with columns:
        - date: observation date
        - exdate: expiration date
        - secid: security ID
        - optionid: option contract ID
        - cp_flag: 'C' for call, 'P' for put
        - K: strike price (in dollars, converted from cents)
        - best_bid: best bid price
        - best_offer: best offer price
        - volume: trading volume
        - open_interest: open interest
        - impl_volatility: market implied volatility (if available)
        - mid: mid price (bid+offer)/2
        - T: time to maturity in years
    
    Examples
    --------
    >>> import wrds
    >>> db = wrds.Connection(wrds_username='your_username')
    >>> df = fetch_optionmetrics_wrds(db, '2023-01-03', 'SPY')
    >>> print(f"Fetched {len(df)} option contracts")
    >>> db.close()
    
    Notes
    -----
    - Requires active WRDS subscription and valid credentials
    - OptionMetrics data is available from 1996 onwards
    - Strike prices are stored in cents in OptionMetrics, converted to dollars
    - Mid price calculated as simple average of bid and offer
    """
    # Step 1: Get security ID for the ticker
    secid_query = f"""
        SELECT secid, ticker 
        FROM optionm.optionmnames 
        WHERE ticker = '{symbol}'
        AND optionid IS NOT NULL
    """
    
    name_df = conn.raw_sql(secid_query)
    
    if name_df.empty:
        raise ValueError(f"No security found for ticker: {symbol}")
    
    secid_list = name_df['secid'].dropna().unique().tolist()
    
    if len(secid_list) == 0:
        raise ValueError(f"No valid secid found for ticker: {symbol}")
    
    # Step 2: Query option price data
    year = date[:4]
    date_str = f"'{date}'"
    
    option_dfs = []
    for secid in secid_list:
        query = f"""
            SELECT date, exdate, secid, optionid, cp_flag,
                   strike_price, best_bid, best_offer,
                   volume, open_interest, impl_volatility
            FROM optionm.opprcd{year}
            WHERE secid = {secid}
            AND date = {date_str}
        """
        df = conn.raw_sql(query)
        if not df.empty:
            option_dfs.append(df)
    
    if not option_dfs:
        raise ValueError(f"No option data found for {symbol} on {date}")
    
    # Combine all dataframes
    op_df = pd.concat(option_dfs, ignore_index=True)
    
    if return_raw:
        return op_df
    
    # Step 3: Preprocess data
    # Convert dates
    op_df['date'] = pd.to_datetime(op_df['date'])
    op_df['exdate'] = pd.to_datetime(op_df['exdate'])
    
    # Convert strike price from cents to dollars
    op_df['K'] = op_df['strike_price'] / 1000.0
    op_df.drop(columns=['strike_price'], inplace=True)
    
    # Calculate mid price
    op_df['mid'] = (op_df['best_bid'] + op_df['best_offer']) / 2.0
    
    # Calculate time to maturity in years
    op_df['T'] = (op_df['exdate'] - op_df['date']).dt.days / 365.0
    
    # Reorder columns
    cols = ['date', 'exdate', 'secid', 'optionid', 'cp_flag', 'K',
            'best_bid', 'best_offer', 'mid', 'volume', 'open_interest',
            'impl_volatility', 'T']
    op_df = op_df[cols]
    
    return op_df


def fetch_security_price_wrds(
    conn,
    date: str,
    symbol: str = "SPY"
) -> Dict[str, float]:
    """
    Fetch underlying security price from WRDS OptionMetrics.
    
    Parameters
    ----------
    conn : wrds.Connection
        Active WRDS database connection
    date : str
        Date in format 'YYYY-MM-DD'
    symbol : str, optional
        Ticker symbol (default: "SPY")
    
    Returns
    -------
    dict
        Dictionary with keys: 'S0' (spot price), 'close', 'volume'
    """
    # Get secid
    secid_query = f"""
        SELECT secid 
        FROM optionm.optionmnames 
        WHERE ticker = '{symbol}'
        LIMIT 1
    """
    secid_df = conn.raw_sql(secid_query)
    secid = float(secid_df['secid'].iloc[0])
    
    # Query security price
    year = date[:4]
    query = f"""
        SELECT date, close, volume
        FROM optionm.secprd{year}
        WHERE secid = {secid}
        AND date = '{date}'
    """
    
    price_df = conn.raw_sql(query)
    
    if price_df.empty:
        raise ValueError(f"No price data found for {symbol} on {date}")
    
    return {
        'S0': float(price_df['close'].iloc[0]),
        'close': float(price_df['close'].iloc[0]),
        'volume': float(price_df['volume'].iloc[0])
    }


def fetch_zero_curve_wrds(conn, date: str) -> pd.DataFrame:
    """
    Fetch risk-free rate zero curve from WRDS OptionMetrics.
    
    Parameters
    ----------
    conn : wrds.Connection
        Active WRDS database connection
    date : str
        Date in format 'YYYY-MM-DD'
    
    Returns
    -------
    pd.DataFrame
        Zero curve with columns 'days' and 'rate' (annualized %)
    """
    query = f"""
        SELECT days, rate
        FROM optionm.zerocd
        WHERE date = '{date}'
        ORDER BY days
    """
    
    zc_df = conn.raw_sql(query)
    
    if zc_df.empty:
        warnings.warn(f"No zero curve data for {date}, using default r=0.05")
        # Return default flat curve
        return pd.DataFrame({
            'days': [30, 91, 182, 365, 730],
            'rate': [5.0, 5.0, 5.0, 5.0, 5.0]
        })
    
    return zc_df


def get_rate_interpolator(zero_curve_df: pd.DataFrame):
    """
    Create rate interpolation function from zero curve.
    
    Parameters
    ----------
    zero_curve_df : pd.DataFrame
        Zero curve with 'days' and 'rate' columns
    
    Returns
    -------
    callable
        Function r(T) that takes years and returns annualized rate (%)
    """
    zc_days = zero_curve_df['days'].to_numpy(dtype=float)
    zc_rate = zero_curve_df['rate'].to_numpy(dtype=float)
    
    def r_of_T(T_years: np.ndarray) -> np.ndarray:
        """Interpolate rate for given time to maturity in years."""
        days = np.maximum(T_years, 0) * 365.0
        r = np.interp(days, zc_days, zc_rate, left=zc_rate[0], right=zc_rate[-1])
        return r / 100.0  # Convert from % to decimal
    
    return r_of_T


# ============================================================================
# Data Cleaning and Filtering
# ============================================================================

def clean_option_chain(
    df: pd.DataFrame,
    min_time_to_maturity: float = 0.02,  # ~7 days
    max_time_to_maturity: float = 2.0,   # 2 years
    min_moneyness: float = 0.8,
    max_moneyness: float = 1.2,
    min_volume: int = 0,
    min_open_interest: int = 0,
    only_calls: bool = True,
    remove_missing_prices: bool = True,
    S0: Optional[float] = None
) -> pd.DataFrame:
    """
    Clean and filter option chain data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw option chain data
    min_time_to_maturity : float
        Minimum T in years (default: 0.02 ≈ 7 days)
    max_time_to_maturity : float
        Maximum T in years (default: 2.0)
    min_moneyness : float
        Minimum K/S ratio (default: 0.8)
    max_moneyness : float
        Maximum K/S ratio (default: 1.2)
    min_volume : int
        Minimum trading volume (default: 0)
    min_open_interest : int
        Minimum open interest (default: 0)
    only_calls : bool
        If True, keep only call options (default: True)
    remove_missing_prices : bool
        If True, remove rows with missing bid/offer (default: True)
    S0 : float, optional
        Spot price for moneyness filter. If None, skip moneyness filter.
    
    Returns
    -------
    pd.DataFrame
        Cleaned option chain
    """
    df_clean = df.copy()
    
    # Filter by option type
    if only_calls:
        df_clean = df_clean[df_clean['cp_flag'] == 'C']
    
    # Remove missing prices
    if remove_missing_prices:
        df_clean = df_clean.dropna(subset=['best_bid', 'best_offer'])
        df_clean = df_clean[(df_clean['best_bid'] > 0) & (df_clean['best_offer'] > 0)]
    
    # Filter by time to maturity
    df_clean = df_clean[
        (df_clean['T'] >= min_time_to_maturity) &
        (df_clean['T'] <= max_time_to_maturity)
    ]
    
    # Filter by moneyness
    if S0 is not None:
        df_clean['moneyness'] = df_clean['K'] / S0
        df_clean = df_clean[
            (df_clean['moneyness'] >= min_moneyness) &
            (df_clean['moneyness'] <= max_moneyness)
        ]
    
    # Filter by volume and open interest
    if min_volume > 0:
        df_clean = df_clean[df_clean['volume'] >= min_volume]
    
    if min_open_interest > 0:
        df_clean = df_clean[df_clean['open_interest'] >= min_open_interest]
    
    # Remove duplicates (keep first occurrence)
    df_clean = df_clean.drop_duplicates(subset=['K', 'T'], keep='first')
    
    # Sort by T then K
    df_clean = df_clean.sort_values(['T', 'K']).reset_index(drop=True)
    
    return df_clean


def compute_mid_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mid price from bid and offer.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'best_bid' and 'best_offer' columns
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'mid' column added or updated
    """
    df = df.copy()
    df['mid'] = (df['best_bid'] + df['best_offer']) / 2.0
    return df


# ============================================================================
# Data Loading and Export
# ============================================================================

def load_clean_option_data(
    path: str,
    return_arrays: bool = True
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load cleaned option data from CSV file.
    
    Parameters
    ----------
    path : str
        Path to CSV file containing cleaned option data
    return_arrays : bool
        If True, return numpy arrays; otherwise return DataFrame only
    
    Returns
    -------
    S0 : float
        Spot price of underlying
    K_array : np.ndarray
        Strike prices
    T_array : np.ndarray
        Times to maturity (years)
    C_mkt_array : np.ndarray
        Market call option prices
    df : pd.DataFrame
        Full DataFrame for reference
    
    Examples
    --------
    >>> S0, K, T, C_mkt, df = load_clean_option_data('data/processed/spy_20230103.csv')
    >>> print(f"Loaded {len(K)} option contracts, S0=${S0:.2f}")
    """
    df = pd.read_csv(path)
    
    # Convert date columns if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    if 'exdate' in df.columns:
        df['exdate'] = pd.to_datetime(df['exdate'])
    
    if not return_arrays:
        return None, None, None, None, df
    
    # Extract arrays
    if 'S0' in df.columns:
        S0 = float(df['S0'].iloc[0])
    else:
        # Try to infer from metadata or use a placeholder
        S0 = None
        warnings.warn("S0 not found in CSV, returning None")
    
    K_array = df['K'].to_numpy()
    T_array = df['T'].to_numpy()
    C_mkt_array = df['mid'].to_numpy()
    
    return S0, K_array, T_array, C_mkt_array, df


def save_option_data(
    df: pd.DataFrame,
    path: str,
    S0: Optional[float] = None,
    metadata: Optional[Dict] = None
):
    """
    Save option data to CSV with metadata.
    
    Parameters
    ----------
    df : pd.DataFrame
        Option data to save
    path : str
        Output path for CSV file
    S0 : float, optional
        Spot price to include as column
    metadata : dict, optional
        Additional metadata to save in separate JSON file
    """
    df_save = df.copy()
    
    # Add S0 as column if provided
    if S0 is not None:
        df_save['S0'] = S0
    
    # Save CSV
    df_save.to_csv(path, index=False)
    
    # Save metadata if provided
    if metadata is not None:
        import json
        meta_path = path.replace('.csv', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)


# ============================================================================
# Statistics and Diagnostics
# ============================================================================

def print_data_summary(df: pd.DataFrame, S0: Optional[float] = None):
    """
    Print summary statistics of option data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Option chain data
    S0 : float, optional
        Spot price for moneyness calculation
    """
    print("=" * 70)
    print("Option Chain Data Summary")
    print("=" * 70)
    print(f"\nTotal contracts: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    if 'cp_flag' in df.columns:
        print(f"\nOption types:")
        print(df['cp_flag'].value_counts())
    
    print(f"\nStrike price (K):")
    print(f"  Range: ${df['K'].min():.2f} - ${df['K'].max():.2f}")
    print(f"  Mean: ${df['K'].mean():.2f}")
    
    print(f"\nTime to maturity (T):")
    print(f"  Range: {df['T'].min():.4f} - {df['T'].max():.4f} years")
    print(f"  Range: {df['T'].min()*365:.0f} - {df['T'].max()*365:.0f} days")
    print(f"  Unique maturities: {df['T'].nunique()}")
    
    print(f"\nMid price:")
    print(f"  Range: ${df['mid'].min():.2f} - ${df['mid'].max():.2f}")
    print(f"  Mean: ${df['mid'].mean():.2f}")
    
    if S0 is not None:
        moneyness = df['K'] / S0
        print(f"\nMoneyness (K/S0) with S0=${S0:.2f}:")
        print(f"  Range: {moneyness.min():.3f} - {moneyness.max():.3f}")
        print(f"  ATM contracts (0.98-1.02): {((moneyness >= 0.98) & (moneyness <= 1.02)).sum()}")
    
    if 'volume' in df.columns:
        print(f"\nTrading volume:")
        print(f"  Total: {df['volume'].sum():,.0f}")
        print(f"  Mean: {df['volume'].mean():.1f}")
        print(f"  Median: {df['volume'].median():.1f}")
    
    if 'open_interest' in df.columns:
        print(f"\nOpen interest:")
        print(f"  Total: {df['open_interest'].sum():,.0f}")
        print(f"  Mean: {df['open_interest'].mean():.1f}")
    
    print("\n" + "=" * 70)


# ============================================================================
# Yahoo Finance Interface (for quick prototyping)
# ============================================================================

def fetch_yahoo_options(
    symbol: str = "SPY",
    expiration_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch option chain from Yahoo Finance.
    
    NOTE: This is for prototyping only. For production, use OptionMetrics.
    
    Parameters
    ----------
    symbol : str
        Ticker symbol
    expiration_date : str, optional
        Expiration date in format 'YYYY-MM-DD'. If None, use nearest expiration.
    
    Returns
    -------
    pd.DataFrame
        Option chain data
    
    Requires
    --------
    yfinance : pip install yfinance
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")
    
    ticker = yf.Ticker(symbol)
    
    # Get expiration dates
    expirations = ticker.options
    
    if expiration_date is None:
        # Use nearest expiration
        exp_date = expirations[0]
    else:
        # Find closest match
        exp_date = min(expirations, key=lambda x: abs(
            pd.to_datetime(x) - pd.to_datetime(expiration_date)
        ))
    
    # Get option chain
    opt_chain = ticker.option_chain(exp_date)
    calls_df = opt_chain.calls
    
    # Rename columns to match our convention
    calls_df = calls_df.rename(columns={
        'strike': 'K',
        'lastPrice': 'mid',
        'bid': 'best_bid',
        'ask': 'best_offer',
        'impliedVolatility': 'impl_volatility'
    })
    
    # Add metadata
    calls_df['cp_flag'] = 'C'
    calls_df['date'] = pd.Timestamp.now().normalize()
    calls_df['exdate'] = pd.to_datetime(exp_date)
    calls_df['T'] = (calls_df['exdate'] - calls_df['date']).dt.days / 365.0
    
    # Get spot price
    history = ticker.history(period='1d')
    S0 = history['Close'].iloc[-1]
    calls_df['S0'] = S0
    
    return calls_df


if __name__ == "__main__":
    # Quick test of data structures
    print("Data module loaded successfully!")
    print("\nAvailable functions:")
    print("  - fetch_optionmetrics_wrds()")
    print("  - fetch_security_price_wrds()")
    print("  - fetch_zero_curve_wrds()")
    print("  - clean_option_chain()")
    print("  - load_clean_option_data()")
    print("  - fetch_yahoo_options() [for prototyping]")
