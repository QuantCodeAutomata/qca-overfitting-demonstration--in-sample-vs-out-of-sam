"""
Tests for Experiment 4: Scikit-learn API Validation
"""

import pytest
import numpy as np
from src.exp_4_api_validation import (
    load_data_for_api_test,
    check_fit_predict_api,
    check_transform_api,
    check_get_set_params,
    check_sklearn_compatibility,
    check_cross_validation
)
from skfolio.optimization import MeanRisk, InverseVolatility, HierarchicalRiskParity, EqualWeighted
from skfolio import RiskMeasure
from src.utils import chronological_train_test_split


def test_data_loading_for_api():
    """Test data loading for API tests."""
    returns = load_data_for_api_test()
    
    assert returns is not None
    assert len(returns) > 0
    assert returns.shape[1] > 0
    
    print("✓ Data loading for API test passed")


def test_meanrisk_fit_predict():
    """Test MeanRisk fit/predict API."""
    returns = load_data_for_api_test()
    X_train, X_test = chronological_train_test_split(returns, test_size=0.2, shuffle=False)
    
    estimator = MeanRisk(risk_measure=RiskMeasure.VARIANCE)
    result = check_fit_predict_api(estimator, X_train, X_test, "MeanRisk")
    
    assert result['has_fit'] == True
    assert result['has_predict'] == True
    assert result['fit_successful'] == True
    assert result['predict_successful'] == True
    
    print("✓ MeanRisk fit/predict test passed")


def test_inverse_volatility_fit_predict():
    """Test InverseVolatility fit/predict API."""
    returns = load_data_for_api_test()
    X_train, X_test = chronological_train_test_split(returns, test_size=0.2, shuffle=False)
    
    estimator = InverseVolatility()
    result = check_fit_predict_api(estimator, X_train, X_test, "InverseVolatility")
    
    assert result['has_fit'] == True
    assert result['has_predict'] == True
    assert result['fit_successful'] == True
    assert result['predict_successful'] == True
    
    print("✓ InverseVolatility fit/predict test passed")


def test_hrp_fit_predict():
    """Test HierarchicalRiskParity fit/predict API."""
    returns = load_data_for_api_test()
    X_train, X_test = chronological_train_test_split(returns, test_size=0.2, shuffle=False)
    
    estimator = HierarchicalRiskParity()
    result = check_fit_predict_api(estimator, X_train, X_test, "HRP")
    
    assert result['has_fit'] == True
    assert result['has_predict'] == True
    assert result['fit_successful'] == True
    assert result['predict_successful'] == True
    
    print("✓ HRP fit/predict test passed")


def test_equal_weighted_fit_predict():
    """Test EqualWeighted fit/predict API."""
    returns = load_data_for_api_test()
    X_train, X_test = chronological_train_test_split(returns, test_size=0.2, shuffle=False)
    
    estimator = EqualWeighted()
    result = check_fit_predict_api(estimator, X_train, X_test, "EqualWeighted")
    
    assert result['has_fit'] == True
    assert result['has_predict'] == True
    assert result['fit_successful'] == True
    assert result['predict_successful'] == True
    
    print("✓ EqualWeighted fit/predict test passed")


def test_get_set_params_meanrisk():
    """Test get_params/set_params for MeanRisk."""
    estimator = MeanRisk(risk_measure=RiskMeasure.VARIANCE)
    result = check_get_set_params(estimator, "MeanRisk")
    
    assert result['has_get_params'] == True
    assert result['has_set_params'] == True
    assert result['get_params_successful'] == True
    
    print("✓ MeanRisk get/set params test passed")


def test_sklearn_train_test_split_compatibility():
    """Test sklearn train_test_split compatibility."""
    returns = load_data_for_api_test()
    result = check_sklearn_compatibility(returns)
    
    assert result['train_test_split_compatible'] == True
    assert result['maintains_chronological_order'] == True
    
    print("✓ sklearn compatibility test passed")


def test_cross_validation_no_data_leakage():
    """Test that cross-validation maintains chronological order."""
    returns = load_data_for_api_test()
    
    # Use smaller dataset for faster testing
    X_small = returns.iloc[-300:] if len(returns) > 300 else returns
    
    estimator = InverseVolatility()
    result = check_cross_validation(estimator, X_small, "InverseVolatility")
    
    assert result['maintains_order'] == True
    assert result['no_data_leakage'] == True
    
    print("✓ Cross-validation no data leakage test passed")


def test_api_consistency_across_estimators():
    """Test that all estimators have consistent API."""
    returns = load_data_for_api_test()
    X_train, X_test = chronological_train_test_split(returns, test_size=0.2, shuffle=False)
    
    estimators = [
        (MeanRisk(), "MeanRisk"),
        (InverseVolatility(), "InverseVolatility"),
        (HierarchicalRiskParity(), "HRP"),
        (EqualWeighted(), "EqualWeighted")
    ]
    
    for estimator, name in estimators:
        # All should have fit and predict
        assert hasattr(estimator, 'fit')
        assert hasattr(estimator, 'predict')
        assert hasattr(estimator, 'get_params')
        assert hasattr(estimator, 'set_params')
    
    print("✓ API consistency test passed")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Running Experiment 4 Tests")
    print("="*80 + "\n")
    
    test_data_loading_for_api()
    test_meanrisk_fit_predict()
    test_inverse_volatility_fit_predict()
    test_hrp_fit_predict()
    test_equal_weighted_fit_predict()
    test_get_set_params_meanrisk()
    test_sklearn_train_test_split_compatibility()
    test_cross_validation_no_data_leakage()
    test_api_consistency_across_estimators()
    
    print("\n" + "="*80)
    print("All Experiment 4 Tests Passed!")
    print("="*80 + "\n")
