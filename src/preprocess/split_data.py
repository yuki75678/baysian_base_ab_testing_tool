import pandas as pd
from typing import Tuple


def split_data(data: pd.DataFrame, group_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splite data to 2 group"""
    group_size_validation(data=data, group_col=group_col)
    group = data[group_col].unique()

    test = data[data[group_col].eq(group[0])].copy().reset_index()
    control = data[data[group_col].eq(group[1])].copy().reset_index()

    assert len(test) == len(control)

    return test, control


def group_size_validation(data: pd.DataFrame, group_col: str) -> None:
    """Validate group size equals to 2"""
    n_of_group = len(data[group_col].unique())
    if n_of_group != 2:
        raise ValueError(f"total number of group must be 2. Now {n_of_group} is given")
