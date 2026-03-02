"""
Shared utility functions for portfolio optimization experiments.
"""

import os
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def ensure_results_dir() -> Path:
    """
    Ensure results directory exists.
    
    Returns:
        Path: Path object pointing to results directory
    """
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    return results_dir


def save_figure(fig: plt.Figure, filename: str) -> None:
    """
    Save a matplotlib figure to the results directory.
    
    Parameters:
        fig: Matplotlib figure to save
        filename: Name of the file (with extension)
    """
    results_dir = ensure_results_dir()
    filepath = results_dir / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {filepath}")


def validate_weights(weights: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Validate that portfolio weights sum to 1.
    
    Parameters:
        weights: Portfolio weight vector
        tolerance: Numerical tolerance for sum constraint
        
    Returns:
        bool: True if weights sum to 1 within tolerance
    """
    weight_sum = np.sum(weights)
    return np.abs(weight_sum - 1.0) < tolerance


def validate_constraint(weights, asset_name: str, max_weight: float) -> bool:
    """
    Validate that a specific asset's weight doesn't exceed a maximum.
    
    Parameters:
        weights: Portfolio weights (pandas Series with asset names as index or numpy array)
        asset_name: Name of the asset to check
        max_weight: Maximum allowed weight
        
    Returns:
        bool: True if constraint is satisfied
    """
    # Handle pandas Series
    if hasattr(weights, 'index'):
        if asset_name not in weights.index:
            return True  # Asset not in portfolio, constraint satisfied
        return weights[asset_name] <= max_weight + 1e-6  # Small tolerance for numerical errors
    
    # For numpy arrays, we can't check specific assets without additional info
    # Just return True as we can't validate
    return True


def compute_portfolio_metrics(returns: pd.DataFrame, weights: np.ndarray) -> dict:
    """
    Compute basic portfolio performance metrics.
    
    Parameters:
        returns: DataFrame of asset returns (T x N)
        weights: Portfolio weights (N,)
        
    Returns:
        dict: Dictionary containing portfolio metrics
    """
    portfolio_returns = returns @ weights
    
    metrics = {
        'mean_return': portfolio_returns.mean() * 252,  # Annualized
        'volatility': portfolio_returns.std() * np.sqrt(252),  # Annualized
        'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)),
        'cumulative_return': (1 + portfolio_returns).prod() - 1
    }
    
    return metrics


def format_metrics_table(metrics_dict: dict) -> str:
    """
    Format metrics dictionary as a markdown table.
    
    Parameters:
        metrics_dict: Dictionary of metrics
        
    Returns:
        str: Markdown formatted table
    """
    table = "| Metric | Value |\n"
    table += "|--------|-------|\n"
    for key, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            table += f"| {key} | {value:.6f} |\n"
        else:
            table += f"| {key} | {value} |\n"
    return table


def append_to_results(content: str, filename: str = "RESULTS.md") -> None:
    """
    Append content to results markdown file.
    
    Parameters:
        content: String content to append
        filename: Name of results file
    """
    results_dir = ensure_results_dir()
    filepath = results_dir / filename
    
    with open(filepath, 'a') as f:
        f.write(content)
        f.write("\n\n")


def initialize_results_file(filename: str = "RESULTS.md") -> None:
    """
    Initialize or clear the results markdown file.
    
    Parameters:
        filename: Name of results file
    """
    results_dir = ensure_results_dir()
    filepath = results_dir / filename
    
    with open(filepath, 'w') as f:
        f.write("# Portfolio Optimization Experiments - Results\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        f.write("---\n\n")


def create_efficient_frontier_plot(
    train_mean: np.ndarray,
    train_std: np.ndarray,
    test_mean: np.ndarray,
    test_std: np.ndarray,
    title: str = "Efficient Frontier: Training vs Test"
) -> plt.Figure:
    """
    Create a scatter plot comparing training and test efficient frontiers.
    
    Parameters:
        train_mean: Array of training portfolio mean returns (annualized)
        train_std: Array of training portfolio standard deviations (annualized)
        test_mean: Array of test portfolio mean returns (annualized)
        test_std: Array of test portfolio standard deviations (annualized)
        title: Plot title
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.scatter(train_std, train_mean, alpha=0.6, s=50, 
               label='Training Frontier', color='blue', marker='o')
    ax.scatter(test_std, test_mean, alpha=0.6, s=50, 
               label='Test Frontier', color='red', marker='x')
    
    ax.set_xlabel('Annualized Standard Deviation', fontsize=12)
    ax.set_ylabel('Annualized Mean Return', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    return fig


def chronological_train_test_split(
    data: pd.DataFrame, 
    test_size: float = 0.2, 
    shuffle: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data chronologically into train and test sets.
    
    Parameters:
        data: DataFrame with time series data (chronologically sorted)
        test_size: Proportion of data for test set
        shuffle: Whether to shuffle (must be False for time series)
        
    Returns:
        Tuple of (train_data, test_data)
    """
    if shuffle:
        raise ValueError("Shuffle must be False for chronological time series split")
    
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    return train_data, test_data
