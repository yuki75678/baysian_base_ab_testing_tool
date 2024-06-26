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
        if not isinstance(model_param, ParamStoreDict):
            raise TypeError("model_param must be of type ParamStoreDict")
        
        self._model_param = deepcopy(model_param)
        self._theoretical_mean_of_model = theoretical_mean_of_model

        # save train data on training point
        self._train_data = train_data.copy()

    @property
    def model_param(self) -> ParamStoreDict:
        return self._model_param

    @property
    def theoretical_mean_of_model(self) -> float:
        return self._theoretical_mean_of_model

    @property
    def train_data(self) -> pd.DataFrame:
        return self._train_data
