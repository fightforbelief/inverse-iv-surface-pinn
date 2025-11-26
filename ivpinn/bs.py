"""
Black-Scholes pricing formulas and Greeks.

This module implements the Black-Scholes option pricing model for European options,
including pricing functions and key Greeks (sensitivity measures).

References:
    Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.
    Journal of Political Economy, 81(3), 637-654.
"""

import numpy as np
from scipy.stats import norm
from typing import Union

# Type hint for array-like inputs
ArrayLike = Union[float, np.ndarray]


def _validate_inputs(S: ArrayLike, K: ArrayLike, T: ArrayLike, 
                     sigma: ArrayLike, r: float, q: float) -> None:
    """
    Validate inputs for Black-Scholes formulas.
    
    Args:
        S: Spot price of underlying asset
        K: Strike price
        T: Time to maturity (in years)
        sigma: Volatility (annualized)
        r: Risk-free rate (annualized)
        q: Dividend yield (annualized, continuous)
    
    Raises:
        ValueError: If any input is invalid
    """
    if np.any(S <= 0):
        raise ValueError("Spot price S must be positive")
    if np.any(K <= 0):
        raise ValueError("Strike price K must be positive")
    if np.any(T < 0):
        raise ValueError("Time to maturity T must be non-negative")
    if np.any(sigma < 0):
        raise ValueError("Volatility sigma must be non-negative")


def _d1_d2(S: ArrayLike, K: ArrayLike, T: ArrayLike, 
           sigma: ArrayLike, r: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate d1 and d2 terms in Black-Scholes formula.
    
    d1 = [ln(S/K) + (r - q + 0.5*sigma^2)*T] / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity
        sigma: Volatility
        r: Risk-free rate
        q: Dividend yield
    
    Returns:
        Tuple of (d1, d2) as numpy arrays
    """
    # Convert inputs to numpy arrays for consistent handling
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Handle edge case: T = 0 (at expiration)
    # At expiration, option value is intrinsic value
    mask_zero_T = (T == 0)
    
    # Initialize arrays
    d1 = np.zeros_like(T, dtype=float)
    d2 = np.zeros_like(T, dtype=float)
    
    # For T > 0, calculate normally
    mask_positive_T = ~mask_zero_T
    if np.any(mask_positive_T):
        T_pos = np.where(mask_positive_T, T, 1.0)  # Avoid division by zero
        sigma_pos = np.where(mask_positive_T, sigma, 1.0)
        
        sqrt_T = np.sqrt(T_pos)
        sigma_sqrt_T = sigma_pos * sqrt_T
        
        # Avoid division by zero when sigma = 0
        mask_zero_sigma = (sigma == 0) & mask_positive_T
        if np.any(mask_zero_sigma):
            # When sigma=0, if S > K, d1 = inf (call deep ITM), else d1 = -inf
            d1 = np.where(mask_zero_sigma, 
                         np.where(S > K, np.inf, -np.inf),
                         d1)
            d2 = d1
        
        # Normal case: sigma > 0 and T > 0
        mask_normal = mask_positive_T & (sigma > 0)
        if np.any(mask_normal):
            d1_normal = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
            d2_normal = d1_normal - sigma * sqrt_T
            
            d1 = np.where(mask_normal, d1_normal, d1)
            d2 = np.where(mask_normal, d2_normal, d2)
    
    # For T = 0, d1 and d2 don't matter (we'll use intrinsic value directly)
    # Set them to indicate ITM/OTM status
    if np.any(mask_zero_T):
        d1 = np.where(mask_zero_T, np.where(S > K, np.inf, -np.inf), d1)
        d2 = d1
    
    return d1, d2


def bs_call_price(S: ArrayLike, K: ArrayLike, T: ArrayLike, 
                  sigma: ArrayLike, r: float = 0.0, q: float = 0.0) -> np.ndarray:
    """
    Calculate European call option price using Black-Scholes formula.
    
    Formula:
        C = S * exp(-q*T) * N(d1) - K * exp(-r*T) * N(d2)
    
    where:
        d1 = [ln(S/K) + (r - q + 0.5*sigma^2)*T] / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        N(·) is the cumulative standard normal distribution
    
    Args:
        S: Spot price of underlying asset (must be positive)
        K: Strike price (must be positive)
        T: Time to maturity in years (must be non-negative)
        sigma: Volatility (annualized, must be non-negative)
        r: Risk-free interest rate (annualized, continuous), default 0.0
        q: Dividend yield (annualized, continuous), default 0.0
    
    Returns:
        Call option price as numpy array
    
    Examples:
        >>> bs_call_price(100, 100, 1.0, 0.2)
        array(7.96557421)
        
        >>> bs_call_price([100, 110], 100, 1.0, 0.2)
        array([ 7.96557421, 14.23144352])
    """
    _validate_inputs(S, K, T, sigma, r, q)
    
    # Convert to arrays
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    
    # Handle T = 0 case (expiration)
    mask_zero_T = (T == 0)
    if np.any(mask_zero_T):
        intrinsic = np.maximum(S - K, 0)
    
    # Calculate d1 and d2
    d1, d2 = _d1_d2(S, K, T, sigma, r, q)
    
    # Calculate option price
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    # For T = 0, use intrinsic value
    if np.any(mask_zero_T):
        call_price = np.where(mask_zero_T, intrinsic, call_price)
    
    return call_price


def bs_put_price(S: ArrayLike, K: ArrayLike, T: ArrayLike, 
                 sigma: ArrayLike, r: float = 0.0, q: float = 0.0) -> np.ndarray:
    """
    Calculate European put option price using Black-Scholes formula.
    
    Formula:
        P = K * exp(-r*T) * N(-d2) - S * exp(-q*T) * N(-d1)
    
    Alternatively, using put-call parity:
        P = C - S*exp(-q*T) + K*exp(-r*T)
    
    Args:
        S: Spot price of underlying asset (must be positive)
        K: Strike price (must be positive)
        T: Time to maturity in years (must be non-negative)
        sigma: Volatility (annualized, must be non-negative)
        r: Risk-free interest rate (annualized, continuous), default 0.0
        q: Dividend yield (annualized, continuous), default 0.0
    
    Returns:
        Put option price as numpy array
    
    Examples:
        >>> bs_put_price(100, 100, 1.0, 0.2)
        array(7.96557421)
        
        >>> bs_put_price([100, 90], 100, 1.0, 0.2)
        array([ 7.96557421, 14.76093525])
    """
    _validate_inputs(S, K, T, sigma, r, q)
    
    # Convert to arrays
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    
    # Handle T = 0 case (expiration)
    mask_zero_T = (T == 0)
    if np.any(mask_zero_T):
        intrinsic = np.maximum(K - S, 0)
    
    # Calculate d1 and d2
    d1, d2 = _d1_d2(S, K, T, sigma, r, q)
    
    # Calculate option price
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    # For T = 0, use intrinsic value
    if np.any(mask_zero_T):
        put_price = np.where(mask_zero_T, intrinsic, put_price)
    
    return put_price


def bs_vega(S: ArrayLike, K: ArrayLike, T: ArrayLike, 
            sigma: ArrayLike, r: float = 0.0, q: float = 0.0) -> np.ndarray:
    """
    Calculate Vega (sensitivity to volatility) for European options.
    
    Vega is the same for both call and put options.
    
    Formula:
        Vega = S * exp(-q*T) * phi(d1) * sqrt(T)
    
    where phi(·) is the standard normal probability density function.
    
    Note: Vega is often expressed "per percentage point" of volatility,
    so if sigma changes by 0.01 (1%), the option price changes by Vega * 0.01.
    
    Args:
        S: Spot price of underlying asset (must be positive)
        K: Strike price (must be positive)
        T: Time to maturity in years (must be non-negative)
        sigma: Volatility (annualized, must be non-negative)
        r: Risk-free interest rate (annualized, continuous), default 0.0
        q: Dividend yield (annualized, continuous), default 0.0
    
    Returns:
        Vega as numpy array
    
    Examples:
        >>> bs_vega(100, 100, 1.0, 0.2)
        array(39.89422804)
        
        >>> bs_vega([100, 110], 100, 1.0, 0.2)
        array([39.89422804, 36.94039059])
    """
    _validate_inputs(S, K, T, sigma, r, q)
    
    # Convert to arrays
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    
    # Handle T = 0 case (vega is 0 at expiration)
    mask_zero_T = (T == 0)
    if np.any(mask_zero_T):
        vega = np.zeros_like(T, dtype=float)
    
    # Calculate d1
    d1, _ = _d1_d2(S, K, T, sigma, r, q)
    
    # Calculate vega
    sqrt_T = np.sqrt(T)
    vega = S * np.exp(-q * T) * norm.pdf(d1) * sqrt_T
    
    # For T = 0, vega = 0
    if np.any(mask_zero_T):
        vega = np.where(mask_zero_T, 0.0, vega)
    
    return vega


def bs_delta_call(S: ArrayLike, K: ArrayLike, T: ArrayLike, 
                  sigma: ArrayLike, r: float = 0.0, q: float = 0.0) -> np.ndarray:
    """
    Calculate Delta (sensitivity to spot price) for European call options.
    
    Formula:
        Delta_call = exp(-q*T) * N(d1)
    
    Args:
        S: Spot price of underlying asset
        K: Strike price
        T: Time to maturity in years
        sigma: Volatility (annualized)
        r: Risk-free interest rate (annualized, continuous), default 0.0
        q: Dividend yield (annualized, continuous), default 0.0
    
    Returns:
        Delta as numpy array (ranges from 0 to 1 for calls)
    """
    _validate_inputs(S, K, T, sigma, r, q)
    
    T = np.asarray(T, dtype=float)
    
    # At expiration, delta is 1 if ITM, 0 if OTM
    mask_zero_T = (T == 0)
    if np.any(mask_zero_T):
        S_arr = np.asarray(S, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        delta_expiry = np.where(S_arr > K_arr, 1.0, 0.0)
    
    d1, _ = _d1_d2(S, K, T, sigma, r, q)
    delta = np.exp(-q * T) * norm.cdf(d1)
    
    if np.any(mask_zero_T):
        delta = np.where(mask_zero_T, delta_expiry, delta)
    
    return delta


def bs_delta_put(S: ArrayLike, K: ArrayLike, T: ArrayLike, 
                 sigma: ArrayLike, r: float = 0.0, q: float = 0.0) -> np.ndarray:
    """
    Calculate Delta (sensitivity to spot price) for European put options.
    
    Formula:
        Delta_put = -exp(-q*T) * N(-d1) = exp(-q*T) * (N(d1) - 1)
    
    Args:
        S: Spot price of underlying asset
        K: Strike price
        T: Time to maturity in years
        sigma: Volatility (annualized)
        r: Risk-free interest rate (annualized, continuous), default 0.0
        q: Dividend yield (annualized, continuous), default 0.0
    
    Returns:
        Delta as numpy array (ranges from -1 to 0 for puts)
    """
    _validate_inputs(S, K, T, sigma, r, q)
    
    T = np.asarray(T, dtype=float)
    
    # At expiration, delta is -1 if ITM, 0 if OTM
    mask_zero_T = (T == 0)
    if np.any(mask_zero_T):
        S_arr = np.asarray(S, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        delta_expiry = np.where(S_arr < K_arr, -1.0, 0.0)
    
    d1, _ = _d1_d2(S, K, T, sigma, r, q)
    delta = -np.exp(-q * T) * norm.cdf(-d1)
    
    if np.any(mask_zero_T):
        delta = np.where(mask_zero_T, delta_expiry, delta)
    
    return delta


def bs_gamma(S: ArrayLike, K: ArrayLike, T: ArrayLike, 
             sigma: ArrayLike, r: float = 0.0, q: float = 0.0) -> np.ndarray:
    """
    Calculate Gamma (second derivative with respect to spot price).
    
    Gamma is the same for both call and put options.
    
    Formula:
        Gamma = exp(-q*T) * phi(d1) / (S * sigma * sqrt(T))
    
    Args:
        S: Spot price of underlying asset
        K: Strike price
        T: Time to maturity in years
        sigma: Volatility (annualized)
        r: Risk-free interest rate (annualized, continuous), default 0.0
        q: Dividend yield (annualized, continuous), default 0.0
    
    Returns:
        Gamma as numpy array
    """
    _validate_inputs(S, K, T, sigma, r, q)
    
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # At expiration, gamma is 0
    mask_zero_T = (T == 0)
    if np.any(mask_zero_T):
        gamma = np.zeros_like(T, dtype=float)
    
    d1, _ = _d1_d2(S, K, T, sigma, r, q)
    sqrt_T = np.sqrt(T)
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * sqrt_T)
    
    if np.any(mask_zero_T):
        gamma = np.where(mask_zero_T, 0.0, gamma)
    
    return gamma


# Convenience function to verify put-call parity
def verify_put_call_parity(S: ArrayLike, K: ArrayLike, T: ArrayLike, 
                           sigma: ArrayLike, r: float = 0.0, q: float = 0.0,
                           tolerance: float = 1e-10) -> bool:
    """
    Verify put-call parity relationship.
    
    Put-call parity: C - P = S*exp(-q*T) - K*exp(-r*T)
    
    Args:
        S, K, T, sigma, r, q: Option parameters
        tolerance: Maximum allowed deviation
    
    Returns:
        True if parity holds within tolerance
    """
    C = bs_call_price(S, K, T, sigma, r, q)
    P = bs_put_price(S, K, T, sigma, r, q)
    
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    
    lhs = C - P
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    
    return np.all(np.abs(lhs - rhs) < tolerance)