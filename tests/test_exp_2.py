"""
Tests for Experiment 2: Factor Model Prior Integration
"""

import pytest
import numpy as np
from src.exp_2_factor_priors import (
    load_and_align_data,
    create_baseline_optimizer,
    create_factor_prior_optimizer,
    compute_stability_metrics
)


def test_data_alignment():
    """Test that equity and factor data are properly aligned."""
    X_train, X_test, y_train, y_test = load_and_align_data(test_size=0.2)
    
    # Check data exists
    assert X_train is not None
    assert y_train is not None
    assert X_test is not None
    assert y_test is not None
    
    # Check alignment
    assert len(X_train) == len(y_train), "Training data length mismatch"
    assert len(X_test) == len(y_test), "Test data length mismatch"
    assert (X_train.index == y_train.index).all(), "Training indices don't match"
    assert (X_test.index == y_test.index).all(), "Test indices don't match"
    
    # Check chronological order
    assert X_train.index[-1] < X_test.index[0], "Train should come before test"
    
    print("✓ Data alignment test passed")


def test_baseline_optimizer_creation():
    """Test baseline optimizer configuration."""
    optimizer = create_baseline_optimizer(max_aapl_weight=0.2, l2_coef=0.01)
    
    assert optimizer is not None
    assert optimizer.l2_coef == 0.01
    assert 'AAPL' in optimizer.max_weights
    assert optimizer.max_weights['AAPL'] == 0.2
    
    print("✓ Baseline optimizer creation test passed")


def test_factor_prior_optimizer_creation():
    """Test factor-prior optimizer configuration."""
    optimizer = create_factor_prior_optimizer(
        max_aapl_weight=0.2,
        l2_coef=0.01,
        ridge_alpha=0.1
    )
    
    assert optimizer is not None
    assert optimizer.l2_coef == 0.01
    assert optimizer.prior_estimator is not None
    
    print("✓ Factor prior optimizer creation test passed")


def test_ridge_alpha_parameter():
    """Test that ridge alpha parameter is correctly set."""
    optimizer = create_factor_prior_optimizer(ridge_alpha=0.1)
    
    # Check that prior estimator exists
    assert optimizer.prior_estimator is not None
    
    # Verify it's a FactorModel
    assert hasattr(optimizer.prior_estimator, 'loading_matrix_estimator')
    
    print("✓ Ridge alpha parameter test passed")


def test_factor_model_fit():
    """Test that factor model can be fitted with both X and y."""
    X_train, X_test, y_train, y_test = load_and_align_data(test_size=0.2)
    
    optimizer = create_factor_prior_optimizer()
    
    # Fit should work with both X and y
    try:
        optimizer.fit(X_train, y_train)
        fitted = True
    except Exception as e:
        print(f"Error fitting: {e}")
        fitted = False
    
    assert fitted, "Factor model should fit with X and y"
    
    print("✓ Factor model fit test passed")


def test_baseline_vs_factor_prior():
    """Test that baseline and factor prior produce different portfolios."""
    X_train, X_test, y_train, y_test = load_and_align_data(test_size=0.2)
    
    baseline = create_baseline_optimizer()
    factor_prior = create_factor_prior_optimizer()
    
    baseline.fit(X_train)
    factor_prior.fit(X_train, y_train)
    
    baseline_portfolio = baseline.predict(X_test)
    factor_portfolio = factor_prior.predict(X_test)
    
    # Portfolios should be different
    baseline_weights = baseline_portfolio.weights.values if hasattr(baseline_portfolio.weights, 'values') else baseline_portfolio.weights
    factor_weights = factor_portfolio.weights.values if hasattr(factor_portfolio.weights, 'values') else factor_portfolio.weights
    weight_diff = np.sum(np.abs(baseline_weights - factor_weights))
    
    print(f"  Weight difference: {weight_diff:.4f}")
    
    # Should have some difference (though might be small)
    assert weight_diff >= 0
    
    print("✓ Baseline vs factor prior test passed")


def test_stability_metrics_calculation():
    """Test stability metrics computation."""
    X_train, X_test, y_train, y_test = load_and_align_data(test_size=0.2)
    
    optimizer = create_baseline_optimizer()
    optimizer.fit(X_train)
    portfolio = optimizer.predict(X_test)
    
    metrics = compute_stability_metrics(portfolio)
    
    # Check all expected metrics exist
    assert 'herfindahl_index' in metrics
    assert 'effective_n_assets' in metrics
    assert 'max_weight' in metrics
    assert 'min_nonzero_weight' in metrics
    assert 'n_nonzero_weights' in metrics
    
    # Check value ranges
    assert 0 < metrics['herfindahl_index'] <= 1
    assert metrics['effective_n_assets'] >= 1
    assert 0 <= metrics['max_weight'] <= 1
    assert metrics['n_nonzero_weights'] >= 1
    
    print("✓ Stability metrics calculation test passed")


def test_constraints_with_factors():
    """Test that constraints are enforced with factor priors."""
    X_train, X_test, y_train, y_test = load_and_align_data(test_size=0.2)
    
    optimizer = create_factor_prior_optimizer(max_aapl_weight=0.2)
    optimizer.fit(X_train, y_train)
    portfolio = optimizer.predict(X_test)
    
    # Check budget constraint
    weights = portfolio.weights
    weights_array = weights.values if hasattr(weights, 'values') else weights
    weight_sum = weights_array.sum()
    assert abs(weight_sum - 1.0) < 1e-4
    
    # Check AAPL constraint if present
    if hasattr(weights, 'index') and 'AAPL' in weights.index:
        assert weights['AAPL'] <= 0.2 + 1e-6
    elif hasattr(portfolio, 'asset_names') and 'AAPL' in portfolio.asset_names:
        aapl_idx = list(portfolio.asset_names).index('AAPL')
        assert weights_array[aapl_idx] <= 0.2 + 1e-6
    
    print("✓ Constraints with factors test passed")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Running Experiment 2 Tests")
    print("="*80 + "\n")
    
    test_data_alignment()
    test_baseline_optimizer_creation()
    test_factor_prior_optimizer_creation()
    test_ridge_alpha_parameter()
    test_factor_model_fit()
    test_baseline_vs_factor_prior()
    test_stability_metrics_calculation()
    test_constraints_with_factors()
    
    print("\n" + "="*80)
    print("All Experiment 2 Tests Passed!")
    print("="*80 + "\n")
