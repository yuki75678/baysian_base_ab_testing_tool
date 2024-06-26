import pandas as pd


class Preprocessed:
    def __init__(self, control_data: pd.DataFrame, test_data: pd.DataFrame, y_col: str):
        """
        Initialize the Preprocessed object with control and test data, and the column name for y.
        
        Args:
            control_data (pd.DataFrame): The control group data.
            test_data (pd.DataFrame): The test group data.
            y_col (str): The column name for the target variable.
        """
        self._control_data = control_data
        self._test_data = test_data
        if y_col not in control_data.columns or y_col not in test_data.columns:
            raise ValueError(f"Column {y_col} must be present in both control and test data.")

    @property
    def control_data(self) -> pd.DataFrame:
        """Returns the control group data."""
        return self._control_data

    @property
    def test_data(self) -> pd.DataFrame:
        """Returns the test group data."""
        return self._test_data

    @property
    def y_col(self) -> str:
        """Returns the column name for the target variable."""
        return self._y_col
