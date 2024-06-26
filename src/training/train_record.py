import pandas as pd
from copy import deepcopy
from pyro.params.param_store import ParamStoreDict


class TrainRecord:
    def __init__(
        self,
        model_param: ParamStoreDict,
        theorical_mean_of_model: float,
        train_data: pd.DataFrame,
    ) -> None:
        print(type(model_param))
        self.model_param = model_param
        self.theorical_mean_of_model = deepcopy(theorical_mean_of_model)

        # save train data on training point
        self.train_data = deepcopy(train_data)

    def get_model_param(self) -> ParamStoreDict:
        return self.model_param

    def get_theorical_mean_of_model(self) -> float:
        return self.theorical_mean_of_model

    def get_train_data(self) -> pd.DataFrame:
        return self.train_data
