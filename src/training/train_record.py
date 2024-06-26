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
        """
        Initializes the TrainRecord with model parameters, theoretical mean of the model, and training data.

        Args:
            model_param (ParamStoreDict): The parameters of the trained model.
            theoretical_mean_of_model (float): The theoretical mean of the model.
            train_data (pd.DataFrame): The training data used for training.

        Raises:
            TypeError: If model_param is not an instance of ParamStoreDict.
        """
        if not isinstance(model_param, ParamStoreDict):
            raise TypeError("model_param must be of type ParamStoreDict")

        self._model_param = deepcopy(model_param)
        self._theoretical_mean_of_model = theoretical_mean_of_model

        # save train data on training point
        self._train_data = train_data.copy()

    @property
    def model_param(self) -> ParamStoreDict:
        """
        Gets the model parameters.

        Returns:
            ParamStoreDict: The parameters of the trained model.
        """
        return self._model_param

    @property
    def theoretical_mean_of_model(self) -> float:
        """
        Gets the theoretical mean of the model.

        Returns:
            float: The theoretical mean of the model.
        """
        return self._theoretical_mean_of_model

    @property
    def train_data(self) -> pd.DataFrame:
        """
        Gets the training data.

        Returns:
            pd.DataFrame: The training data used for training.
        """
        return self._train_data
