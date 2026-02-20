import numpy as np
from typing import Dict, Optional


class MinMaxScaler:
    def __init__(self, feature_range: tuple = (0, 1)):
        self.feature_range = feature_range
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        X = np.array(X, dtype=np.float64)

        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)

        range_diff = self.data_max_ - self.data_min_
        range_diff[range_diff == 0] = 1.0

        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / range_diff
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("MinMaxScaler must be fitted before transform")

        X = np.array(X, dtype=np.float64)
        return X * self.scale_ + self.min_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("MinMaxScaler must be fitted before inverse_transform")

        X = np.array(X, dtype=np.float64)
        return (X - self.min_) / self.scale_

    def get_state(self) -> Dict:
        return {
            "feature_range": self.feature_range,
            "min": self.min_,
            "max": self.max_,
            "data_min": self.data_min_,
            "data_max": self.data_max_,
            "scale": self.scale_,
            "fitted": self._fitted,
        }

    def set_state(self, state: Dict) -> "MinMaxScaler":
        self.feature_range = state["feature_range"]
        self.min_ = state["min"]
        self.max_ = state["max"]
        self.data_min_ = state["data_min"]
        self.data_max_ = state["data_max"]
        self.scale_ = state["scale"]
        self._fitted = state["fitted"]
        return self


class StandardScaler:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.array(X, dtype=np.float64)

        self.mean_ = np.mean(X, axis=0)
        self.var_ = np.var(X, axis=0)
        self.std_ = np.sqrt(self.var_)

        self.std_[self.std_ == 0] = 1.0

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("StandardScaler must be fitted before transform")

        X = np.array(X, dtype=np.float64)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("StandardScaler must be fitted before inverse_transform")

        X = np.array(X, dtype=np.float64)
        return X * self.std_ + self.mean_

    def get_state(self) -> Dict:
        return {
            "mean": self.mean_,
            "std": self.std_,
            "var": self.var_,
            "fitted": self._fitted,
        }

    def set_state(self, state: Dict) -> "StandardScaler":
        self.mean_ = state["mean"]
        self.std_ = state["std"]
        self.var_ = state["var"]
        self._fitted = state["fitted"]
        return self
