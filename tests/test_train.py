import pytest
from sklearn.linear_model import LinearRegression

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import train

# Test 1: Check if data is loaded properly
def test_load_data():
    X_train, X_test, y_train, y_test = train.load_data()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] > 0
    assert y_test.shape[0] > 0

# Test 2: Model instance should be LinearRegression
def test_model_instance():
    X_train, _, y_train, _ = train.load_data()
    model = train.train_model(X_train, y_train)
    assert isinstance(model, LinearRegression)

# Test 3: Check if model was trained (has coefficients)
def test_model_trained():
    X_train, _, y_train, _ = train.load_data()
    model = train.train_model(X_train, y_train)
    assert hasattr(model, "coef_")
    assert model.coef_ is not None

# Test 4: R² score should exceed threshold (e.g., 0.5)
def test_r2_score_threshold():
    X_train, X_test, y_train, y_test = train.load_data()
    model = train.train_model(X_train, y_train)
    r2, _ = train.evaluate_model(model, X_test, y_test)
    assert r2 > 0.5  # Acceptable performance threshold
