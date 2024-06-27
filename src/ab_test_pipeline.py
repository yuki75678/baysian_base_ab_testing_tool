import pandas as pd
from src.training.train import train
from src.preprocess import preprocess_pipeline
from src.simulation.ctr_model.monte_carlro_ctr import monte_carlo_simularer_ctr


def ab_test_pipline(
        data:pd.DataFrame,
        y_col:str,
        model:str,
        group_col:str,
        num_iterations:int,
        lr:float,
        betas:tuple,
        )->None:
    """
    Run A/B testing pipeline with preprocessing, model training, and Monte Carlo simulation.

    Parameters:
    data (pd.DataFrame): The input data frame.
    y_col (str): The column name of the target variable.
    model (str): The model type to be used for training.
    group_col (str): The column name for group assignment (A[test]/B[contorol]).
    num_iterations (int): Number of iterations for training the model.
    lr (float): Learning rate for the optimizer.
    betas (tuple): Betas parameters for the optimizer.
    """
    # split data to group A or B
    preprocessed = preprocess_pipeline(data=data,y_col=y_col,model=model,group_col=group_col)
    
    
    # train posterior surrogate model
    adam_params = {"lr": lr, "betas": betas}
    test_model_record = train(
        data=preprocessed.test_data_y_as_torch,
        num_iterations=num_iterations,
        adam_params=adam_params)
    control_model_record = train(
        data=preprocessed.control_data_y_as_torch,
        num_iterations=num_iterations,
        adam_params=adam_params)
    

    # conduct monte carlo simulation based on trained model
    monte_carlo_simularer_ctr(
        test_model_record=test_model_record,
        control_model_record=control_model_record
                              )
