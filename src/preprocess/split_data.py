import pandas as pd
from typing import Tuple


def split_data(data: pd.DataFrame, group_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into two groups based on the specified group column.

    Args:
        data (pd.DataFrame): The input data to be split.
        group_col (str): The column name used to split the data into two groups.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Two dataframes, one for each group.

    Raises:
        ValueError: If the group column does not contain exactly two unique values.
        ValueError: If Input data size not equals to sum of test and control length
    """
    group_size_validation(data=data, group_col=group_col)
    group = data[group_col].unique()

    test = data[data[group_col].eq(group[0])].copy().reset_index()
    control = data[data[group_col].eq(group[1])].copy().reset_index()

    if len(test) + len(control)==len(data):
        raise ValueError(f"The sum of test and control groups must equal to input, found {len(test)} and {len(control)}. However input size is {len(data)}")

    return test, control


def group_size_validation(data: pd.DataFrame, group_col: str) -> None:
    """
    Validate that the group column contains exactly two unique values.

    Args:
        data (pd.DataFrame): The input data containing the group column.
        group_col (str): The column name to validate.

    Raises:
        ValueError: If the group column does not contain exactly two unique values.
    """
    n_of_group = len(data[group_col].unique())
    if n_of_group != 2:
        raise ValueError(f"total number of group must be 2. Now {n_of_group} is given")
