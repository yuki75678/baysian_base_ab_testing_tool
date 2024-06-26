import torch
import pyro
from copy import deepcopy
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from src.pyro_model.ctr.model_ctr import model_ctr, guide_ctr
from src.training.train_record import TrainRecord
from src.training.eval_model import evaluate_model_theoretical


def train(
    data: torch.Tensor,
    num_iterations: int = 100,
    adam_params=None,
) -> TrainRecord:
    """
    Train postrior distribution by SVI method
    Note that training is conducted on each data separately (e.g. control data)
    Args:
        data: tensor data of test group or train group
        num_iterations: training iteration
        adam_params: svi's adam param settings
    """
    train_data_for_record = deepcopy(data)

    pyro.clear_param_store()
    # Setting optimizer
    optimizer = Adam(adam_params)
    # Setting ELBO
    svi = SVI(model_ctr, guide_ctr, optimizer, loss=Trace_ELBO())

    # model training
    for i in range(num_iterations):
        loss = svi.step(data.float())
        if i % 100 == 0:
            print(f"Iteration {i} : loss = {loss}")

    theoretical_mean_of_model, _ = evaluate_model_theoretical()
    print("theoretical_mean_of_model", theoretical_mean_of_model)

    return TrainRecord(
        model_param=pyro.get_param_store(),
        theoretical_mean_of_model=theoretical_mean_of_model,
        train_data=train_data_for_record,
    )
