"""
Experiment 2: Factor Model Prior Integration in Portfolio Optimization

This experiment integrates factor model priors using ridge regression into mean-variance
optimization and compares performance against baseline optimization without priors.

Methodology:
1. Load equity and factor price datasets
2. Align datasets in time as specified
3. Convert to returns producing X (assets) and y (factors)
4. Split both X and y chronologically (80/20)
5. Create baseline MeanRisk with same parameters as exp_1
6. Create factor-prior MeanRisk with FactorModel(LoadingMatrixRegression(Ridge(alpha=0.1)))
7. Fit both models on training data
8. Generate predictions/portfolios on test data
9. Compare out-of-sample summary statistics and stability metrics
"""

from typing import Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

from skfolio import RiskMeasure
from skfolio.datasets import load_sp500_dataset, load_factors_dataset
from skfolio.preprocessing import prices_to_returns
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.prior import FactorModel, LoadingMatrixRegression

from src.utils import (
    chronological_train_test_split,
    save_figure,
    validate_weights,
    validate_constraint,
    append_to_results,
    compute_portfolio_metrics
)


def load_and_align_data(test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load equity and factor datasets, align in time, and split.
    
    Parameters:
        test_size: Proportion of data for test set
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) where X is asset returns and y is factor returns
    """
    # Load equity prices (S&P 500)
    equity_prices = load_sp500_dataset()
    
    # Load factor prices
    factor_prices = load_factors_dataset()
    
    # Align datasets in time - find common date range
    common_dates = equity_prices.index.intersection(factor_prices.index)
    
    if len(common_dates) == 0:
        raise ValueError("No common dates between equity and factor datasets")
    
    # Subset both datasets to common dates
    equity_prices_aligned = equity_prices.loc[common_dates]
    factor_prices_aligned = factor_prices.loc[common_dates]
    
    # Sort chronologically
    equity_prices_aligned = equity_prices_aligned.sort_index()
    factor_prices_aligned = factor_prices_aligned.sort_index()
    
    # Convert to returns using skfolio function that outputs X and y
    # Note: prices_to_returns can handle both equity and factor data
    equity_returns = prices_to_returns(equity_prices_aligned)
    factor_returns = prices_to_returns(factor_prices_aligned)
    
    # Split both X and y chronologically
    X_train, X_test = chronological_train_test_split(equity_returns, test_size=test_size, shuffle=False)
    y_train, y_test = chronological_train_test_split(factor_returns, test_size=test_size, shuffle=False)
    
    # Verify alignment
    assert len(X_train) == len(y_train), "Training data length mismatch"
    assert len(X_test) == len(y_test), "Test data length mismatch"
    assert (X_train.index == y_train.index).all(), "Training data indices don't match"
    assert (X_test.index == y_test.index).all(), "Test data indices don't match"
    
    return X_train, X_test, y_train, y_test


def create_baseline_optimizer(max_aapl_weight: float = 0.2, l2_coef: float = 0.01) -> MeanRisk:
    """
    Create baseline MeanRisk optimizer without factor priors.
    
    Parameters:
        max_aapl_weight: Maximum weight for AAPL
        l2_coef: L2 regularization coefficient
        
    Returns:
        Baseline MeanRisk optimizer
    """
    optimizer = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        max_weights={'AAPL': max_aapl_weight},
        l2_coef=l2_coef
    )
    return optimizer


def create_factor_prior_optimizer(
    max_aapl_weight: float = 0.2, 
    l2_coef: float = 0.01,
    ridge_alpha: float = 0.1
) -> MeanRisk:
    """
    Create MeanRisk optimizer with factor model priors.
    
    Parameters:
        max_aapl_weight: Maximum weight for AAPL
        l2_coef: L2 regularization coefficient
        ridge_alpha: Ridge regression alpha parameter
        
    Returns:
        MeanRisk optimizer with factor model prior
    """
    # Create loading matrix estimator with Ridge regression
    loading_estimator = LoadingMatrixRegression(
        linear_regressor=Ridge(alpha=ridge_alpha)
    )
    
    # Create factor model prior
    factor_prior = FactorModel(
        loading_matrix_estimator=loading_estimator
    )
    
    # Create optimizer with factor prior
    optimizer = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        max_weights={'AAPL': max_aapl_weight},
        l2_coef=l2_coef,
        prior_estimator=factor_prior
    )
    
    return optimizer


def compute_stability_metrics(portfolio) -> Dict[str, float]:
    """
    Compute portfolio stability metrics.
    
    Parameters:
        portfolio: Portfolio object from skfolio
        
    Returns:
        Dictionary of stability metrics
    """
    metrics = {}
    
    # Weight concentration (Herfindahl index)
    weights = portfolio.weights
    if hasattr(weights, 'values'):
        weights = weights.values
    metrics['herfindahl_index'] = np.sum(weights ** 2)
    
    # Effective number of assets
    metrics['effective_n_assets'] = 1.0 / metrics['herfindahl_index']
    
    # Maximum weight
    metrics['max_weight'] = weights.max()
    
    # Minimum non-zero weight
    non_zero_weights = weights[weights > 1e-6]
    metrics['min_nonzero_weight'] = non_zero_weights.min() if len(non_zero_weights) > 0 else 0.0
    
    # Number of non-zero weights
    metrics['n_nonzero_weights'] = len(non_zero_weights)
    
    return metrics


def run_experiment_2() -> Dict[str, any]:
    """
    Run Experiment 2: Factor Model Prior Integration.
    
    Returns:
        Dictionary containing experiment results and metrics
    """
    print("=" * 80)
    print("EXPERIMENT 2: Factor Model Prior Integration")
    print("=" * 80)
    
    # Step 1-4: Load and prepare data
    print("\n[1/6] Loading and aligning equity and factor data...")
    X_train, X_test, y_train, y_test = load_and_align_data(test_size=0.2)
    
    print(f"  - Training period: {X_train.index[0]} to {X_train.index[-1]}")
    print(f"  - Test period: {X_test.index[0]} to {X_test.index[-1]}")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Number of assets: {X_train.shape[1]}")
    print(f"  - Number of factors: {y_train.shape[1]}")
    
    # Step 5: Create baseline optimizer
    print("\n[2/6] Creating baseline MeanRisk optimizer (no priors)...")
    baseline_optimizer = create_baseline_optimizer(max_aapl_weight=0.2, l2_coef=0.01)
    print("  - Risk Measure: VARIANCE")
    print("  - Objective: MAXIMIZE_RATIO (Sharpe)")
    print("  - AAPL Max Weight: 0.2")
    print("  - L2 Regularization: 0.01")
    
    # Step 6: Create factor-prior optimizer
    print("\n[3/6] Creating factor-prior MeanRisk optimizer...")
    factor_optimizer = create_factor_prior_optimizer(
        max_aapl_weight=0.2, 
        l2_coef=0.01,
        ridge_alpha=0.1
    )
    print("  - Prior Estimator: FactorModel")
    print("  - Loading Matrix: Ridge Regression")
    print("  - Ridge Alpha: 0.1")
    
    # Step 7: Fit both models on training data
    print("\n[4/6] Fitting optimizers on training data...")
    
    print("  - Fitting baseline optimizer...")
    baseline_optimizer.fit(X_train)
    
    print("  - Fitting factor-prior optimizer...")
    # Factor model needs both X and y for fitting
    factor_optimizer.fit(X_train, y_train)
    
    print("  - Both optimizers fitted successfully")
    
    # Step 8: Generate predictions on test data
    print("\n[5/6] Generating predictions on test data...")
    
    baseline_portfolio = baseline_optimizer.predict(X_test)
    factor_portfolio = factor_optimizer.predict(X_test)
    
    print("  - Baseline portfolio predicted")
    print("  - Factor-prior portfolio predicted")
    
    # Step 9: Compare out-of-sample performance
    print("\n[6/6] Computing out-of-sample metrics...")
    
    # Get summary statistics
    print("\n  Baseline Portfolio Summary:")
    print(f"    - Annualized Mean: {baseline_portfolio.annualized_mean:.4f}")
    print(f"    - Annualized Std: {baseline_portfolio.annualized_standard_deviation:.4f}")
    print(f"    - Sharpe Ratio: {baseline_portfolio.sharpe_ratio:.4f}")
    
    print("\n  Factor-Prior Portfolio Summary:")
    print(f"    - Annualized Mean: {factor_portfolio.annualized_mean:.4f}")
    print(f"    - Annualized Std: {factor_portfolio.annualized_standard_deviation:.4f}")
    print(f"    - Sharpe Ratio: {factor_portfolio.sharpe_ratio:.4f}")
    
    # Compute stability metrics
    baseline_stability = compute_stability_metrics(baseline_portfolio)
    factor_stability = compute_stability_metrics(factor_portfolio)
    
    print("\n  Baseline Stability Metrics:")
    for key, value in baseline_stability.items():
        print(f"    - {key}: {value:.4f}")
    
    print("\n  Factor-Prior Stability Metrics:")
    for key, value in factor_stability.items():
        print(f"    - {key}: {value:.4f}")
    
    # Validate constraints
    baseline_weights = baseline_portfolio.weights
    factor_weights = factor_portfolio.weights
    
    # Handle both pandas Series and numpy arrays
    baseline_weights_array = baseline_weights.values if hasattr(baseline_weights, 'values') else baseline_weights
    factor_weights_array = factor_weights.values if hasattr(factor_weights, 'values') else factor_weights
    
    baseline_valid = validate_weights(baseline_weights_array) and validate_constraint(baseline_weights, 'AAPL', 0.2)
    factor_valid = validate_weights(factor_weights_array) and validate_constraint(factor_weights, 'AAPL', 0.2)
    
    print(f"\n  Baseline constraints valid: {baseline_valid}")
    print(f"  Factor-prior constraints valid: {factor_valid}")
    
    # Create comparison visualization
    print("\n[7/7] Creating comparison visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Performance comparison
    metrics = ['Annualized\nMean', 'Annualized\nStd', 'Sharpe\nRatio']
    baseline_values = [
        baseline_portfolio.annualized_mean,
        baseline_portfolio.annualized_standard_deviation,
        baseline_portfolio.sharpe_ratio
    ]
    factor_values = [
        factor_portfolio.annualized_mean,
        factor_portfolio.annualized_standard_deviation,
        factor_portfolio.sharpe_ratio
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0].bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
    axes[0].bar(x + width/2, factor_values, width, label='Factor Prior', alpha=0.8)
    axes[0].set_xlabel('Metrics')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Performance Comparison: Baseline vs Factor Prior')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Stability comparison
    stability_metrics = ['Herfindahl\nIndex', 'Effective\nN Assets', 'N Nonzero\nWeights']
    baseline_stability_values = [
        baseline_stability['herfindahl_index'],
        baseline_stability['effective_n_assets'],
        baseline_stability['n_nonzero_weights']
    ]
    factor_stability_values = [
        factor_stability['herfindahl_index'],
        factor_stability['effective_n_assets'],
        factor_stability['n_nonzero_weights']
    ]
    
    x2 = np.arange(len(stability_metrics))
    axes[1].bar(x2 - width/2, baseline_stability_values, width, label='Baseline', alpha=0.8)
    axes[1].bar(x2 + width/2, factor_stability_values, width, label='Factor Prior', alpha=0.8)
    axes[1].set_xlabel('Metrics')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Stability Comparison: Baseline vs Factor Prior')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(stability_metrics)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, "exp_2_factor_prior_comparison.png")
    plt.close(fig)
    
    results = {
        'baseline_portfolio': baseline_portfolio,
        'factor_portfolio': factor_portfolio,
        'baseline_stability': baseline_stability,
        'factor_stability': factor_stability,
        'baseline_valid': baseline_valid,
        'factor_valid': factor_valid,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 2 COMPLETED")
    print("=" * 80)
    
    return results


def save_experiment_2_results(results: Dict[str, any]) -> None:
    """
    Save Experiment 2 results to markdown file.
    
    Parameters:
        results: Dictionary containing experiment results
    """
    content = "## Experiment 2: Factor Model Prior Integration\n\n"
    content += "### Objective\n"
    content += "Integrate factor model priors using ridge regression into mean-variance optimization "
    content += "and compare performance against baseline optimization without priors.\n\n"
    
    content += "### Configuration\n"
    content += "- **Dataset**: S&P 500 + Factor returns (time-aligned)\n"
    content += "- **Train/Test Split**: 80/20 chronological\n"
    content += "- **Ridge Alpha**: 0.1\n"
    content += "- **Constraints**: AAPL max weight = 0.2, L2 = 0.01\n\n"
    
    content += "### Results\n\n"
    
    baseline = results['baseline_portfolio']
    factor = results['factor_portfolio']
    
    content += "#### Performance Comparison\n\n"
    content += "| Metric | Baseline | Factor Prior | Improvement |\n"
    content += "|--------|----------|--------------|-------------|\n"
    content += f"| Annualized Mean | {baseline.annualized_mean:.4f} | {factor.annualized_mean:.4f} | {((factor.annualized_mean - baseline.annualized_mean) / abs(baseline.annualized_mean) * 100):.2f}% |\n"
    content += f"| Annualized Std | {baseline.annualized_standard_deviation:.4f} | {factor.annualized_standard_deviation:.4f} | {((factor.annualized_standard_deviation - baseline.annualized_standard_deviation) / baseline.annualized_standard_deviation * 100):.2f}% |\n"
    content += f"| Sharpe Ratio | {baseline.sharpe_ratio:.4f} | {factor.sharpe_ratio:.4f} | {((factor.sharpe_ratio - baseline.sharpe_ratio) / abs(baseline.sharpe_ratio) * 100):.2f}% |\n\n"
    
    content += "#### Stability Metrics\n\n"
    baseline_stab = results['baseline_stability']
    factor_stab = results['factor_stability']
    
    content += "| Metric | Baseline | Factor Prior |\n"
    content += "|--------|----------|-------------|\n"
    content += f"| Herfindahl Index | {baseline_stab['herfindahl_index']:.4f} | {factor_stab['herfindahl_index']:.4f} |\n"
    content += f"| Effective N Assets | {baseline_stab['effective_n_assets']:.2f} | {factor_stab['effective_n_assets']:.2f} |\n"
    content += f"| N Nonzero Weights | {baseline_stab['n_nonzero_weights']} | {factor_stab['n_nonzero_weights']} |\n\n"
    
    content += "#### Constraint Validation\n"
    content += f"- Baseline constraints valid: {results['baseline_valid']}\n"
    content += f"- Factor-prior constraints valid: {results['factor_valid']}\n\n"
    
    content += "### Conclusion\n"
    content += "Factor model priors provide "
    if factor.sharpe_ratio > baseline.sharpe_ratio:
        content += "improved risk-adjusted returns "
    else:
        content += "different risk-return characteristics "
    content += "compared to baseline optimization. "
    content += "The integration of factor structure through ridge regression affects portfolio stability and concentration.\n\n"
    
    content += "![Factor Prior Comparison](exp_2_factor_prior_comparison.png)\n\n"
    content += "---\n"
    
    append_to_results(content)


if __name__ == "__main__":
    results = run_experiment_2()
    save_experiment_2_results(results)
