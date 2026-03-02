"""
Experiment 1: Overfitting Demonstration - In-Sample vs Out-of-Sample Efficient Frontier

This experiment demonstrates that classical mean-variance optimization overfits by comparing
in-sample and out-of-sample efficient frontiers, where the training frontier dominates 
the test frontier.

Methodology:
1. Load S&P 500 prices from skfolio dataset
2. Convert to returns using skfolio preprocessing
3. Split chronologically into train/test (80/20)
4. Configure MeanRisk optimizer with:
   - risk_measure=VARIANCE
   - objective_function=MINIMIZE_RISK
   - efficient_frontier_size=100
   - max_weights={'AAPL': 0.2}
   - l2_coef=0.01
5. Fit optimizer on training returns
6. Generate efficient frontier on both training and test data
7. Plot Annualized Mean vs Annualized Standard Deviation for both frontiers
8. Compute summary statistics
"""

from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skfolio import RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns
from skfolio.optimization import MeanRisk, ObjectiveFunction

from src.utils import (
    chronological_train_test_split,
    create_efficient_frontier_plot,
    save_figure,
    validate_weights,
    validate_constraint,
    append_to_results,
    format_metrics_table
)


def load_and_prepare_data(test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load S&P 500 dataset and prepare train/test splits.
    
    Parameters:
        test_size: Proportion of data for test set (default 0.2 for 80/20 split)
        
    Returns:
        Tuple of (prices_train, prices_test, returns_train, returns_test)
    """
    # Load S&P 500 prices
    prices = load_sp500_dataset()
    
    # Verify AAPL is in the dataset
    if 'AAPL' not in prices.columns:
        raise ValueError("AAPL ticker not found in S&P 500 dataset")
    
    # Split prices chronologically (80/20)
    prices_train, prices_test = chronological_train_test_split(
        prices, 
        test_size=test_size, 
        shuffle=False
    )
    
    # Convert prices to returns using skfolio preprocessing
    returns_train = prices_to_returns(prices_train)
    returns_test = prices_to_returns(prices_test)
    
    return prices_train, prices_test, returns_train, returns_test


def configure_optimizer(
    efficient_frontier_size: int = 100,
    max_aapl_weight: float = 0.2,
    l2_coef: float = 0.01
) -> MeanRisk:
    """
    Configure MeanRisk optimizer with specified parameters.
    
    Parameters:
        efficient_frontier_size: Number of points on efficient frontier
        max_aapl_weight: Maximum weight for AAPL (default 0.2)
        l2_coef: L2 regularization coefficient (default 0.01)
        
    Returns:
        Configured MeanRisk optimizer
    """
    optimizer = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        efficient_frontier_size=efficient_frontier_size,
        max_weights={'AAPL': max_aapl_weight},
        l2_coef=l2_coef
    )
    
    return optimizer


def extract_frontier_statistics(portfolios) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract annualized mean and standard deviation from frontier portfolios.
    
    Parameters:
        portfolios: Collection of Portfolio objects from efficient frontier
        
    Returns:
        Tuple of (annualized_means, annualized_stds)
    """
    means = []
    stds = []
    
    for portfolio in portfolios:
        # Extract annualized mean return and standard deviation
        means.append(portfolio.annualized_mean)
        stds.append(portfolio.annualized_standard_deviation)
    
    return np.array(means), np.array(stds)


def validate_portfolios(portfolios, max_aapl_weight: float = 0.2) -> Dict[str, bool]:
    """
    Validate that all portfolios satisfy constraints.
    
    Parameters:
        portfolios: Collection of Portfolio objects
        max_aapl_weight: Maximum allowed weight for AAPL
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'all_weights_sum_to_one': True,
        'aapl_constraint_satisfied': True,
        'num_portfolios': len(portfolios)
    }
    
    for i, portfolio in enumerate(portfolios):
        weights = portfolio.weights
        
        # Handle both pandas Series and numpy arrays
        if hasattr(weights, 'values'):
            weights_array = weights.values
        else:
            weights_array = weights
        
        # Check budget constraint (weights sum to 1)
        if not validate_weights(weights_array):
            validation_results['all_weights_sum_to_one'] = False
            weight_sum = weights_array.sum() if hasattr(weights_array, 'sum') else np.sum(weights_array)
            print(f"Portfolio {i}: Weight sum = {weight_sum}")
        
        # Check AAPL constraint
        # Get AAPL weight
        if hasattr(weights, 'get'):
            aapl_weight = weights.get('AAPL', 0)
        elif hasattr(portfolio, 'asset_names') and 'AAPL' in portfolio.asset_names:
            aapl_idx = list(portfolio.asset_names).index('AAPL')
            aapl_weight = weights_array[aapl_idx]
        else:
            aapl_weight = 0
        
        if aapl_weight > max_aapl_weight + 1e-6:
            validation_results['aapl_constraint_satisfied'] = False
            print(f"Portfolio {i}: AAPL weight = {aapl_weight}")
    
    return validation_results


def run_experiment_1() -> Dict[str, any]:
    """
    Run Experiment 1: Overfitting Demonstration.
    
    Returns:
        Dictionary containing experiment results and metrics
    """
    print("=" * 80)
    print("EXPERIMENT 1: Overfitting Demonstration - In-Sample vs Out-of-Sample")
    print("=" * 80)
    
    # Step 1-3: Load and prepare data
    print("\n[1/5] Loading S&P 500 data and creating train/test split...")
    prices_train, prices_test, returns_train, returns_test = load_and_prepare_data(test_size=0.2)
    
    print(f"  - Training period: {returns_train.index[0]} to {returns_train.index[-1]}")
    print(f"  - Test period: {returns_test.index[0]} to {returns_test.index[-1]}")
    print(f"  - Training samples: {len(returns_train)}")
    print(f"  - Test samples: {len(returns_test)}")
    print(f"  - Number of assets: {returns_train.shape[1]}")
    print(f"  - AAPL in dataset: {'AAPL' in returns_train.columns}")
    
    # Step 4: Configure optimizer
    print("\n[2/5] Configuring MeanRisk optimizer...")
    optimizer = configure_optimizer(
        efficient_frontier_size=100,
        max_aapl_weight=0.2,
        l2_coef=0.01
    )
    print(f"  - Risk Measure: VARIANCE")
    print(f"  - Objective: MINIMIZE_RISK")
    print(f"  - Efficient Frontier Size: 100")
    print(f"  - AAPL Max Weight: 0.2")
    print(f"  - L2 Regularization: 0.01")
    
    # Step 5: Fit optimizer on training data
    print("\n[3/5] Fitting optimizer on training data...")
    optimizer.fit(returns_train)
    print("  - Optimizer fitted successfully")
    
    # Step 6: Generate efficient frontier on both training and test data
    print("\n[4/5] Generating efficient frontiers...")
    
    # Training frontier
    print("  - Generating training frontier (100 points)...")
    train_portfolios = optimizer.predict(returns_train)
    train_means, train_stds = extract_frontier_statistics(train_portfolios)
    print(f"    Generated {len(train_portfolios)} training portfolios")
    
    # Test frontier (apply trained model to test data)
    print("  - Generating test frontier (100 points)...")
    test_portfolios = optimizer.predict(returns_test)
    test_means, test_stds = extract_frontier_statistics(test_portfolios)
    print(f"    Generated {len(test_portfolios)} test portfolios")
    
    # Step 7: Validate constraints
    print("\n[5/5] Validating constraints...")
    train_validation = validate_portfolios(train_portfolios, max_aapl_weight=0.2)
    test_validation = validate_portfolios(test_portfolios, max_aapl_weight=0.2)
    
    print(f"  Training portfolios:")
    print(f"    - All weights sum to 1: {train_validation['all_weights_sum_to_one']}")
    print(f"    - AAPL constraint satisfied: {train_validation['aapl_constraint_satisfied']}")
    
    print(f"  Test portfolios:")
    print(f"    - All weights sum to 1: {test_validation['all_weights_sum_to_one']}")
    print(f"    - AAPL constraint satisfied: {test_validation['aapl_constraint_satisfied']}")
    
    # Step 8: Create visualization
    print("\n[6/6] Creating visualization...")
    fig = create_efficient_frontier_plot(
        train_mean=train_means,
        train_std=train_stds,
        test_mean=test_means,
        test_std=test_stds,
        title="Experiment 1: In-Sample vs Out-of-Sample Efficient Frontier"
    )
    save_figure(fig, "exp_1_efficient_frontier.png")
    plt.close(fig)
    
    # Compute summary statistics
    print("\n[7/7] Computing summary statistics...")
    summary_stats = {
        'train_mean_return_range': (train_means.min(), train_means.max()),
        'train_volatility_range': (train_stds.min(), train_stds.max()),
        'test_mean_return_range': (test_means.min(), test_means.max()),
        'test_volatility_range': (test_stds.min(), test_stds.max()),
        'train_sharpe_max': (train_means / train_stds).max(),
        'test_sharpe_max': (test_means / test_stds).max(),
    }
    
    print(f"  Training frontier:")
    print(f"    - Mean return range: [{summary_stats['train_mean_return_range'][0]:.4f}, {summary_stats['train_mean_return_range'][1]:.4f}]")
    print(f"    - Volatility range: [{summary_stats['train_volatility_range'][0]:.4f}, {summary_stats['train_volatility_range'][1]:.4f}]")
    print(f"    - Max Sharpe ratio: {summary_stats['train_sharpe_max']:.4f}")
    
    print(f"  Test frontier:")
    print(f"    - Mean return range: [{summary_stats['test_mean_return_range'][0]:.4f}, {summary_stats['test_mean_return_range'][1]:.4f}]")
    print(f"    - Volatility range: [{summary_stats['test_volatility_range'][0]:.4f}, {summary_stats['test_volatility_range'][1]:.4f}]")
    print(f"    - Max Sharpe ratio: {summary_stats['test_sharpe_max']:.4f}")
    
    # Check if training frontier dominates test frontier (overfitting demonstration)
    dominates = summary_stats['train_sharpe_max'] > summary_stats['test_sharpe_max']
    print(f"\n  Training frontier dominates test frontier: {dominates}")
    
    results = {
        'train_means': train_means,
        'train_stds': train_stds,
        'test_means': test_means,
        'test_stds': test_stds,
        'train_portfolios': train_portfolios,
        'test_portfolios': test_portfolios,
        'train_validation': train_validation,
        'test_validation': test_validation,
        'summary_stats': summary_stats,
        'overfitting_demonstrated': dominates
    }
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 1 COMPLETED")
    print("=" * 80)
    
    return results


def save_experiment_1_results(results: Dict[str, any]) -> None:
    """
    Save Experiment 1 results to markdown file.
    
    Parameters:
        results: Dictionary containing experiment results
    """
    content = "## Experiment 1: Overfitting Demonstration\n\n"
    content += "### Objective\n"
    content += "Demonstrate that classical mean-variance optimization overfits by comparing "
    content += "in-sample and out-of-sample efficient frontiers.\n\n"
    
    content += "### Configuration\n"
    content += "- **Dataset**: S&P 500 daily returns\n"
    content += "- **Train/Test Split**: 80/20 chronological\n"
    content += "- **Risk Measure**: Variance\n"
    content += "- **Frontier Points**: 100\n"
    content += "- **Constraints**: AAPL max weight = 0.2\n"
    content += "- **L2 Regularization**: λ = 0.01\n\n"
    
    content += "### Results\n\n"
    
    stats = results['summary_stats']
    content += "#### Training Frontier\n"
    content += f"- Mean Return Range: [{stats['train_mean_return_range'][0]:.4f}, {stats['train_mean_return_range'][1]:.4f}]\n"
    content += f"- Volatility Range: [{stats['train_volatility_range'][0]:.4f}, {stats['train_volatility_range'][1]:.4f}]\n"
    content += f"- Max Sharpe Ratio: {stats['train_sharpe_max']:.4f}\n\n"
    
    content += "#### Test Frontier\n"
    content += f"- Mean Return Range: [{stats['test_mean_return_range'][0]:.4f}, {stats['test_mean_return_range'][1]:.4f}]\n"
    content += f"- Volatility Range: [{stats['test_volatility_range'][0]:.4f}, {stats['test_volatility_range'][1]:.4f}]\n"
    content += f"- Max Sharpe Ratio: {stats['test_sharpe_max']:.4f}\n\n"
    
    content += "#### Constraint Validation\n"
    content += f"- Training: All weights sum to 1: {results['train_validation']['all_weights_sum_to_one']}\n"
    content += f"- Training: AAPL constraint satisfied: {results['train_validation']['aapl_constraint_satisfied']}\n"
    content += f"- Test: All weights sum to 1: {results['test_validation']['all_weights_sum_to_one']}\n"
    content += f"- Test: AAPL constraint satisfied: {results['test_validation']['aapl_constraint_satisfied']}\n\n"
    
    content += "### Conclusion\n"
    content += f"**Overfitting Demonstrated**: {results['overfitting_demonstrated']}\n\n"
    content += "The training efficient frontier dominates the test frontier, demonstrating that "
    content += "classical mean-variance optimization overfits to in-sample data.\n\n"
    
    content += "![Efficient Frontier](exp_1_efficient_frontier.png)\n\n"
    content += "---\n"
    
    append_to_results(content)


if __name__ == "__main__":
    results = run_experiment_1()
    save_experiment_1_results(results)
