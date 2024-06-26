import pandas as pd


class Preprocessed:
    def __init__(self, control_data: pd.DataFrame, test_data: pd.DataFrame, y_col: str):
        self.control_data = control_data
        self.test_data = test_data
        self.y_col = y_col

    def get_control_data(self) -> pd.DataFrame:
        return self.control_data

    def get_test_data(self) -> pd.DataFrame:
        return self.test_data

    def get_y_col(self) -> str:
        return self.y_col
