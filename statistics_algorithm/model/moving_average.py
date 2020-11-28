from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np

from moving_average_base import MovingAverage


class _BaseTsModel(ABC):
    """
    Base class for all time series models
    """
    def __init__(self, name: str):
        self._model_name = name
        self._ts_model = None
        self._fit_len = None
        self._dt = None

    @abstractmethod
    def fit(self, y, dt=None, **kwargs):
        return 0

    @abstractmethod
    def predict(self, dt, **kwargs):
        return 0

    @abstractmethod
    def reconstruct(self, y, **kwargs):
        # TODO: This method need to be further re-factored, to seperate the data and parameter of a model,
        #       by potentially re=fractorize the base class.
        return 0


class MovingAverage(_BaseTsModel):
    """

    """

    def __init__(self):
        super(tsflowMovingAverage, self).__init__(name="ma")

    def fit(self, y, dt=None, **kwargs):
        self._ts_model = MovingAverage(data=y)
        return deepcopy(self)

    def predict(self, dt, **kwargs):
        if isinstance(dt, int):
            _predict_len = dt
        else:
            _predict_len = dt.shape[0]
      
        _predict_res = self._ts_model.get_last_window(_predict_len)
        return np.array(_predict_res[-_predict_len:])

    def reconstruct(self, y, **kwargs):
        self.fit(y)
        return deepcopy(self)