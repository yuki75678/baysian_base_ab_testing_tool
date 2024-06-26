import pandas as pd
import numpy as np


def validation_data(data: pd.DataFrame, model: str, y_col: str)->None:
    """
    Validate the input data based on the specified model.

    Args:
        data (pd.DataFrame): The input data for validation.
        model (str): The model type to use for validation.
        y_col (str): The column name of the target variable.

    Raises:
        ValueError: If the model type is not defined.
    """
    if model == "model_ctr":
        validate_binary_column_invalid_values(data=data, y_col=y_col)

    else:
        raise ValueError("Model type is not defined")


def validate_binary_column_invalid_values(data: pd.DataFrame, y_col: str)->None:
    """
    Validate that the specified column contains only binary values (0 or 1).

    Args:
        data (pd.DataFrame): The input data containing the column to validate.
        y_col (str): The column name to validate.

    Raises:
        ValueError: If the column contains more than two unique values or values other than 0 and 1.
        TypeError: If the column contains non-numeric values.
    """
    acceptable_value_list = [0, 1]
    y_unique = data[y_col].unique()

    if len(y_unique) > 2:
        raise ValueError(
            f"y must be 1 or 0. Found {len(y_unique)} unique values: {y_unique}"
        )

    for y in y_unique:
        if not (is_int_or_float(y=y)):
            raise TypeError(f"y must be int or float, found {type(y).__name__}")
        else:
            print(y)
        if not (
            is_number_in_acceptable_value_list(
                y=y, acceptable_value_list=acceptable_value_list
            )
        ):
            raise ValueError(f"y must be 1 or 0, found {y}")


def is_int(y) -> bool:
    """Check if the value is an integer."""
    return isinstance(y, (int, np.integer)) and not isinstance(y, bool)


def is_float(y) -> bool:
    """Check if the value is a float."""
    return isinstance(y, (float, np.floating)) and not isinstance(y, bool)


def is_int_or_float(y) -> bool:
    """Check if the value is either an integer or a float."""
    return is_int(y=y) or is_float(y=y)


def is_number_in_acceptable_value_list(y, acceptable_value_list)->bool:
    """Check if the value is in the list of acceptable values."""
    return y in acceptable_value_list


