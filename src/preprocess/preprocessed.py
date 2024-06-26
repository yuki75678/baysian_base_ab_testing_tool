import pandas as pd
import torch


class Preprocessed:
    def __init__(self, test_data: pd.DataFrame, control_data: pd.DataFrame, y_col: str):
        """
        Initialize the Preprocessed object with control and test data, and the column name for y.

        Args:
            control_data (pd.DataFrame): The control group data.
            test_data (pd.DataFrame): The test group data.
            y_col (str): The column name for the target variable.
        """
        self._test_data = test_data
        self._control_data = control_data
        if y_col not in control_data.columns or y_col not in test_data.columns:
            raise ValueError(
                f"Column {y_col} must be present in both control and test data."
            )
        self._y_col = y_col

    @property
    def test_data(self) -> pd.DataFrame:
        """Returns the test group data."""
        return self._test_data

    @property
    def control_data(self) -> pd.DataFrame:
        """Returns the control group data."""
        return self._control_data
    
    @property
    def test_data_y_as_torch(self) -> torch.Tensor:
        """Returns the test group data."""
        return torch.tensor(self._test_data[self._y_col])

    @property
    def control_data_y_as_torch(self) -> pd.DataFrame:
        """Returns the control group data."""
        return torch.tensor(self._control_data[self._y_col])

    @property
    def y_col(self) -> str:
        """Returns the column name for the target variable."""
        return self._y_col
    

