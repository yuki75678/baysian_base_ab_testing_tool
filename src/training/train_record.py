import pandas as pd
from copy import deepcopy
from pyro.params.param_store import ParamStoreDict


class TrainRecord:
    def __init__(
        self,
        model_param: ParamStoreDict,
        theoretical_mean_of_model: float,
        train_data: pd.DataFrame,
    ) -> None:
        self._model_param = model_param
        self._theoretical_mean_of_model = deepcopy(theoretical_mean_of_model)

        # save train data on training point
        self._train_data = deepcopy(train_data)

    @property
    def model_param(self) -> ParamStoreDict:
        return self._model_param

    @property
    def theoretical_mean_of_model(self) -> float:
        return self._theoretical_mean_of_model

    @property
    def train_data(self) -> pd.DataFrame:
        return self._train_data
