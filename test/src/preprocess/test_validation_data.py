import pytest
import pandas as pd
from src.preprocess.validation_data import validation_data,validate_binary_column_invalid_values,is_float,is_int_or_float,is_int

def test_validation_data_model_ctr_valid():
    data = pd.DataFrame({'y': [0, 1, 0, 1]})
    validation_data(data, "model_ctr", "y")  # Should not raise any exception

def test_validation_data_model_ctr_invalid_y():
    data = pd.DataFrame({'y': [0, 1, 2]})
    with pytest.raises(ValueError, match="y must be 1 or 0. Found 3 unique values"):
        validation_data(data, "model_ctr", "y")

def test_validation_data_model_invalid():
    data = pd.DataFrame({'y': [0, 1]})
    with pytest.raises(ValueError, match="Model type is not defined"):
        validation_data(data, "invalid_model", "y")

def test_validate_binary_column_invalid_values_valid():
    data = pd.DataFrame({'y': [0, 1, 0, 1]})
    validate_binary_column_invalid_values(data, "y")  # Should not raise any exception

def test_validate_binary_column_invalid_values_invalid_y_values():
    data = pd.DataFrame({'y': [0, 1, 2]})
    with pytest.raises(ValueError, match="y must be 1 or 0. Found 3 unique values"):
        validate_binary_column_invalid_values(data, "y")

def test_validate_binary_column_invalid_values_invalid_y_type():
    data = pd.DataFrame({'y': [0, "a"]})
    with pytest.raises(TypeError, match="y must be int or float, found str"):
        validate_binary_column_invalid_values(data, "y")

def test_validate_binary_column_invalid_values_invalid_y_type_bool():
    data = pd.DataFrame({'y': [0, True]})
    with pytest.raises(TypeError, match="y must be int or float, found bool"):
        validate_binary_column_invalid_values(data, "y")

def test_is_int():
    assert is_int(1) == True
    assert is_int(1.0) == False
    assert is_int("1") == False

def test_is_float():
    assert is_float(1.0) == True
    assert is_float(1) == False
    assert is_float("1.0") == False

def test_is_int_or_float():
    assert is_int_or_float(1) == True
    assert is_int_or_float(1.0) == True
    assert is_int_or_float("1") == False
