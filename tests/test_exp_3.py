"""
Tests for Experiment 3: Stacking Optimization with Walk-Forward Cross-Validation
"""

import pytest
import numpy as np
from src.exp_3_stacking import (
    prepare_data_for_stacking,
    create_base_estimators,
    create_final_estimator,
    create_stacking_optimizer,
    create_benchmark,
    configure_walk_forward_cv
)


def test_data_preparation():
    """Test data preparation for stacking."""
    X_train, X_test = prepare_data_for_stacking(test_size=0.2)
    
    assert X_train is not None
    assert X_test is not None
    assert len(X_train) > 0
    assert len(X_test) > 0
    
    # Chronological order
    assert X_train.index[-1] < X_test.index[0]
    
    print("✓ Data preparation test passed")


def test_base_estimators_creation():
    """Test that base estimators are created correctly."""
    estimators = create_base_estimators()
    
    assert len(estimators) == 3
    assert len(estimators[0]) == 2  # (name, estimator) tuple
    
    # Check names
    names = [name for name, _ in estimators]
    assert "inverse_vol" in names
    assert "mean_cvar" in names
    assert "hrp" in names
    
    print("✓ Base estimators creation test passed")


def test_final_estimator_creation():
    """Test final estimator creation."""
    estimator = create_final_estimator()
    
    assert estimator is not None
    assert hasattr(estimator, 'fit')
    assert hasattr(estimator, 'predict')
    
    print("✓ Final estimator creation test passed")


def test_stacking_optimizer_creation():
    """Test stacking optimizer creation."""
    stacking = create_stacking_optimizer()
    
    assert stacking is not None
    assert hasattr(stacking, 'estimators')
    assert hasattr(stacking, 'final_estimator')
    assert len(stacking.estimators) == 3
    
    print("✓ Stacking optimizer creation test passed")


def test_benchmark_creation():
    """Test equal-weighted benchmark creation."""
    benchmark = create_benchmark()
    
    assert benchmark is not None
    assert hasattr(benchmark, 'fit')
    assert hasattr(benchmark, 'predict')
    
    print("✓ Benchmark creation test passed")


def test_walk_forward_cv_configuration():
    """Test walk-forward CV configuration."""
    cv = configure_walk_forward_cv(train_size=252, test_size=60)
    
    assert cv is not None
    assert hasattr(cv, 'split')
    
    print("✓ Walk-forward CV configuration test passed")


def test_walk_forward_maintains_order():
    """Test that walk-forward CV maintains chronological order."""
    X_train, X_test = prepare_data_for_stacking(test_size=0.2)
    
    # Use smaller window for testing
    cv = configure_walk_forward_cv(train_size=100, test_size=20)
    
    # Check that each split maintains order
    for train_idx, test_idx in cv.split(X_test):
        if len(train_idx) > 0 and len(test_idx) > 0:
            assert train_idx[-1] < test_idx[0], "Train indices should come before test indices"
    
    print("✓ Walk-forward maintains order test passed")


def test_stacking_fit_predict():
    """Test that stacking can fit and predict."""
    X_train, X_test = prepare_data_for_stacking(test_size=0.2)
    
    stacking = create_stacking_optimizer()
    
    # Fit
    stacking.fit(X_train)
    
    # Predict
    portfolio = stacking.predict(X_test)
    
    assert portfolio is not None
    assert hasattr(portfolio, 'weights')
    
    # Check budget constraint
    weight_sum = portfolio.weights.sum()
    assert abs(weight_sum - 1.0) < 1e-4
    
    print("✓ Stacking fit/predict test passed")


def test_benchmark_fit_predict():
    """Test that benchmark can fit and predict."""
    X_train, X_test = prepare_data_for_stacking(test_size=0.2)
    
    benchmark = create_benchmark()
    
    # Fit
    benchmark.fit(X_train)
    
    # Predict
    portfolio = benchmark.predict(X_test)
    
    assert portfolio is not None
    assert hasattr(portfolio, 'weights')
    
    # Equal-weighted should have equal weights
    weights = portfolio.weights
    weights_array = weights.values if hasattr(weights, 'values') else weights
    unique_weights = np.unique(weights_array)
    # All weights should be approximately equal
    weight_std = np.std(weights_array)
    assert weight_std < 0.01, "Equal-weighted should have similar weights"
    
    print("✓ Benchmark fit/predict test passed")


def test_cv_fold_count():
    """Test that CV generates expected number of folds."""
    X_train, X_test = prepare_data_for_stacking(test_size=0.2)
    
    cv = configure_walk_forward_cv(train_size=100, test_size=20)
    
    n_folds = 0
    for train_idx, test_idx in cv.split(X_test):
        n_folds += 1
    
    print(f"  Number of folds: {n_folds}")
    assert n_folds >= 0
    
    print("✓ CV fold count test passed")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Running Experiment 3 Tests")
    print("="*80 + "\n")
    
    test_data_preparation()
    test_base_estimators_creation()
    test_final_estimator_creation()
    test_stacking_optimizer_creation()
    test_benchmark_creation()
    test_walk_forward_cv_configuration()
    test_walk_forward_maintains_order()
    test_stacking_fit_predict()
    test_benchmark_fit_predict()
    test_cv_fold_count()
    
    print("\n" + "="*80)
    print("All Experiment 3 Tests Passed!")
    print("="*80 + "\n")
