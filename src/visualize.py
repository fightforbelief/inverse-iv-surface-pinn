import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.interpolate import griddata
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


def plot_volatility_smile(
    df: pd.DataFrame,
    maturity_days: List[int],
    iv_col: str = 'computed_iv',
    ax: Optional[plt.Axes] = None,
    option_type: Optional[str] = None
) -> plt.Axes:
    """Plot volatility smile curves for specific maturities."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(maturity_days)))
    
    for i, days in enumerate(maturity_days):
        mask = (df['days_to_maturity'] >= days - 2) & (df['days_to_maturity'] <= days + 2)
        
        if option_type:
            mask &= (df['cp_flag'] == option_type)
        
        subset = df[mask].copy()
        
        if len(subset) == 0:
            continue
        
        subset = subset.sort_values('moneyness')
        
        label = f'{days} days'
        if option_type:
            label += f' ({option_type})'
        
        ax.plot(
            subset['moneyness'],
            subset[iv_col],
            'o-',
            color=colors[i],
            label=label,
            alpha=0.7,
            markersize=4
        )
    
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.5, label='ATM')
    ax.set_xlabel('Moneyness (K/S)', fontsize=12)
    ax.set_ylabel('Implied Volatility', fontsize=12)
    ax.set_title('Volatility Smile', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_volatility_surface_3d(
    df: pd.DataFrame,
    iv_col: str = 'computed_iv',
    fig: Optional[plt.Figure] = None,
    interpolate: bool = True,
    grid_size: int = 50
) -> plt.Figure:
    """Plot 3D volatility surface."""
    if fig is None:
        fig = plt.figure(figsize=(14, 10))
    
    ax = fig.add_subplot(111, projection='3d')
    
    moneyness = df['moneyness'].values
    T = df['T'].values
    iv = df[iv_col].values
    
    if interpolate:
        m_min, m_max = moneyness.min(), moneyness.max()
        t_min, t_max = T.min(), T.max()
        
        m_grid = np.linspace(m_min, m_max, grid_size)
        t_grid = np.linspace(t_min, t_max, grid_size)
        M_grid, T_grid = np.meshgrid(m_grid, t_grid)
        
        points = np.column_stack([moneyness, T])
        IV_grid = griddata(points, iv, (M_grid, T_grid), method='cubic')
        
        surf = ax.plot_surface(
            M_grid, T_grid, IV_grid,
            cmap=cm.viridis,
            alpha=0.8,
            antialiased=True,
            edgecolor='none'
        )
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Implied Volatility')
    else:
        scatter = ax.scatter(
            moneyness, T, iv,
            c=iv,
            cmap=cm.viridis,
            s=20,
            alpha=0.6
        )
        fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='Implied Volatility')
    
    ax.set_xlabel('Moneyness (K/S)', fontsize=11)
    ax.set_ylabel('Time to Maturity (years)', fontsize=11)
    ax.set_zlabel('Implied Volatility', fontsize=11)
    ax.set_title('Implied Volatility Surface', fontsize=14, fontweight='bold')
    
    return fig


def plot_volatility_heatmap(
    df: pd.DataFrame,
    iv_col: str = 'computed_iv',
    figsize: Tuple[int, int] = (12, 8),
    grid_size: Tuple[int, int] = (30, 30)
) -> plt.Figure:
    """Plot volatility surface as heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    
    moneyness = df['moneyness'].values
    T = df['T'].values
    iv = df[iv_col].values
    
    m_bins = np.linspace(moneyness.min(), moneyness.max(), grid_size[0])
    t_bins = np.linspace(T.min(), T.max(), grid_size[1])
    M_grid, T_grid = np.meshgrid(m_bins, t_bins)
    
    points = np.column_stack([moneyness, T])
    IV_grid = griddata(points, iv, (M_grid, T_grid), method='cubic')
    
    im = ax.contourf(M_grid, T_grid, IV_grid, levels=20, cmap='viridis')
    fig.colorbar(im, ax=ax, label='Implied Volatility')
    
    contours = ax.contour(M_grid, T_grid, IV_grid, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    
    ax.set_xlabel('Moneyness (K/S)', fontsize=12)
    ax.set_ylabel('Time to Maturity (years)', fontsize=12)
    ax.set_title('Implied Volatility Surface (Heatmap)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_term_structure(
    df: pd.DataFrame,
    iv_col: str = 'computed_iv',
    moneyness_levels: List[float] = [0.9, 0.95, 1.0, 1.05, 1.1],
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot volatility term structure for different moneyness levels."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(moneyness_levels)))
    
    for i, m_level in enumerate(moneyness_levels):
        mask = (df['moneyness'] >= m_level - 0.02) & (df['moneyness'] <= m_level + 0.02)
        subset = df[mask].copy()
        
        if len(subset) == 0:
            continue
        
        grouped = subset.groupby('days_to_maturity')[iv_col].agg(['mean', 'std', 'count'])
        grouped = grouped[grouped['count'] >= 2]
        
        if len(grouped) == 0:
            continue
        
        grouped = grouped.sort_index()
        
        ax.plot(
            grouped.index,
            grouped['mean'],
            'o-',
            color=colors[i],
            label=f'M = {m_level:.2f}',
            alpha=0.7,
            markersize=5
        )
        
        ax.fill_between(
            grouped.index,
            grouped['mean'] - grouped['std'],
            grouped['mean'] + grouped['std'],
            color=colors[i],
            alpha=0.2
        )
    
    ax.set_xlabel('Days to Maturity', fontsize=12)
    ax.set_ylabel('Implied Volatility', fontsize=12)
    ax.set_title('Volatility Term Structure', fontsize=14, fontweight='bold')
    ax.legend(loc='best', title='Moneyness')
    ax.grid(True, alpha=0.3)
    
    return ax


def check_calendar_arbitrage(
    df: pd.DataFrame,
    iv_col: str = 'computed_iv',
    moneyness_tolerance: float = 0.02
) -> pd.DataFrame:
    """Check for calendar arbitrage violations (total variance non-monotonicity)."""
    violations = []
    
    moneyness_levels = df['moneyness'].unique()
    
    for m in moneyness_levels:
        mask = (df['moneyness'] >= m - moneyness_tolerance) & (df['moneyness'] <= m + moneyness_tolerance)
        subset = df[mask].copy()
        
        if len(subset) < 2:
            continue
        
        subset = subset.sort_values('T')
        subset['total_variance'] = subset[iv_col] ** 2 * subset['T']
        
        for i in range(len(subset) - 1):
            if subset.iloc[i + 1]['total_variance'] < subset.iloc[i]['total_variance']:
                violations.append({
                    'moneyness': m,
                    'T1': subset.iloc[i]['T'],
                    'T2': subset.iloc[i + 1]['T'],
                    'w1': subset.iloc[i]['total_variance'],
                    'w2': subset.iloc[i + 1]['total_variance'],
                    'violation': subset.iloc[i]['total_variance'] - subset.iloc[i + 1]['total_variance']
                })
    
    return pd.DataFrame(violations)


def check_butterfly_arbitrage(
    df: pd.DataFrame,
    price_col: str = 'mid_price',
    maturity_tolerance: int = 2
) -> pd.DataFrame:
    """Check for butterfly arbitrage violations (convexity)."""
    violations = []
    
    maturities = df['days_to_maturity'].unique()
    
    for mat in maturities:
        mask = (df['days_to_maturity'] >= mat - maturity_tolerance) & (df['days_to_maturity'] <= mat + maturity_tolerance)
        subset = df[mask].copy()
        
        for opt_type in ['C', 'P']:
            type_subset = subset[subset['cp_flag'] == opt_type].copy()
            
            if len(type_subset) < 3:
                continue
            
            type_subset = type_subset.sort_values('K')
            
            for i in range(len(type_subset) - 2):
                K1 = type_subset.iloc[i]['K']
                K2 = type_subset.iloc[i + 1]['K']
                K3 = type_subset.iloc[i + 2]['K']
                
                C1 = type_subset.iloc[i][price_col]
                C2 = type_subset.iloc[i + 1][price_col]
                C3 = type_subset.iloc[i + 2][price_col]
                
                if K2 - K1 < 0.01 or K3 - K2 < 0.01:
                    continue
                
                w1 = (K3 - K2) / (K3 - K1)
                w3 = (K2 - K1) / (K3 - K1)
                interpolated = w1 * C1 + w3 * C3
                
                if C2 > interpolated + 0.01:
                    violations.append({
                        'maturity_days': mat,
                        'option_type': opt_type,
                        'K1': K1,
                        'K2': K2,
                        'K3': K3,
                        'C1': C1,
                        'C2': C2,
                        'C3': C3,
                        'interpolated': interpolated,
                        'violation': C2 - interpolated
                    })
    
    return pd.DataFrame(violations)


def plot_arbitrage_analysis(
    calendar_violations: pd.DataFrame,
    butterfly_violations: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """Plot arbitrage violation analysis."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax1 = axes[0]
    if len(calendar_violations) > 0:
        ax1.scatter(
            calendar_violations['moneyness'],
            calendar_violations['violation'],
            s=50,
            alpha=0.6,
            c='red'
        )
        ax1.set_xlabel('Moneyness', fontsize=11)
        ax1.set_ylabel('Total Variance Decrease', fontsize=11)
        ax1.set_title(f'Calendar Arbitrage Violations\n({len(calendar_violations)} cases)', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No calendar arbitrage violations detected',
                ha='center', va='center', fontsize=12, transform=ax1.transAxes)
        ax1.set_title('Calendar Arbitrage Violations', fontsize=12, fontweight='bold')
    
    ax2 = axes[1]
    if len(butterfly_violations) > 0:
        ax2.scatter(
            butterfly_violations['K2'],
            butterfly_violations['violation'],
            s=50,
            alpha=0.6,
            c='orange'
        )
        ax2.set_xlabel('Strike Price (K2)', fontsize=11)
        ax2.set_ylabel('Convexity Violation', fontsize=11)
        ax2.set_title(f'Butterfly Arbitrage Violations\n({len(butterfly_violations)} cases)', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No butterfly arbitrage violations detected',
                ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        ax2.set_title('Butterfly Arbitrage Violations', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_bid_ask_spread_analysis(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """Analyze and plot bid-ask spreads."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    df = df.copy()
    df['spread'] = df['best_offer'] - df['best_bid']
    df['spread_pct'] = df['spread'] / df['mid_price'] * 100
    
    ax1 = axes[0]
    ax1.scatter(df['moneyness'], df['spread'], s=10, alpha=0.4)
    ax1.set_xlabel('Moneyness', fontsize=11)
    ax1.set_ylabel('Bid-Ask Spread ($)', fontsize=11)
    ax1.set_title('Spread vs Moneyness', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.scatter(df['days_to_maturity'], df['spread'], s=10, alpha=0.4)
    ax2.set_xlabel('Days to Maturity', fontsize=11)
    ax2.set_ylabel('Bid-Ask Spread ($)', fontsize=11)
    ax2.set_title('Spread vs Maturity', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    ax3.hist(df['spread_pct'], bins=50, alpha=0.7, edgecolor='black')
    ax3.axvline(df['spread_pct'].median(), color='red', linestyle='--', 
                label=f'Median: {df["spread_pct"].median():.2f}%')
    ax3.set_xlabel('Bid-Ask Spread (%)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Spread Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def summarize_data_characteristics(df: pd.DataFrame) -> dict:
    """Summarize key characteristics of the option data."""
    summary = {
        'total_options': len(df),
        'calls': len(df[df['cp_flag'] == 'C']),
        'puts': len(df[df['cp_flag'] == 'P']),
        'moneyness_range': (df['moneyness'].min(), df['moneyness'].max()),
        'maturity_range_days': (df['days_to_maturity'].min(), df['days_to_maturity'].max()),
        'maturity_range_years': (df['T'].min(), df['T'].max()),
        'iv_range': (df['computed_iv'].min(), df['computed_iv'].max()),
        'iv_mean': df['computed_iv'].mean(),
        'iv_std': df['computed_iv'].std(),
        'atm_options': len(df[(df['moneyness'] >= 0.98) & (df['moneyness'] <= 1.02)]),
        'otm_calls': len(df[(df['cp_flag'] == 'C') & (df['moneyness'] > 1.02)]),
        'otm_puts': len(df[(df['cp_flag'] == 'P') & (df['moneyness'] < 0.98)]),
        'spot_price': df['S'].iloc[0] if len(df) > 0 else None,
        'unique_maturities': df['days_to_maturity'].nunique(),
        'unique_strikes': df['K'].nunique()
    }
    
    return summary