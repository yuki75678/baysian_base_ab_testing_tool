import pandas as pd
from src.preprocess.validation_data import validation_data
from src.preprocess.split_data import split_data
from src.preprocess.preprocessed import Preprocessed


def preprocess_pipeline(
    data: pd.DataFrame, y_col: str, model: str, group_col: str
) -> Preprocessed:
    """ 
    Perform preprocessing using a pipeline
    Step 1: Validate input data type based on the model type
    Step 2: Split data into test and control groups
    Args:
        data: The input data for the AB test
        y_col: The column containing the values to be used in the AB test
        model: The Bayesian model assumption for each data generation process
        group_col: The column used to split the data into control and test groups
    Returns:
        Preprocessed: An instance containing control_data, test_data, and y_col
    """
    try:
        validation_data(data=data, model=model, y_col=y_col)
    except Exception as e:
        raise ValueError(f"Validation failed: {e}")

    try:
        control_data, test_data = split_data(data=data, group_col=group_col)
    except Exception as e:
        raise ValueError(f"Data split failed: {e}")

    return Preprocessed(control_data=control_data, test_data=test_data, y_col=y_col)
