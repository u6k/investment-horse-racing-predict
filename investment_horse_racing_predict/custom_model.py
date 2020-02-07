from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


class RandomRegressor(BaseEstimator, RegressorMixin):
    def fit(self, x, y):
        self._min_y = np.min(y)
        self._max_y = np.max(y)

        return self

    def predict(self, x):
        return np.random.rand(len(x)) * (self._max_y - self._min_y) + self._min_y
