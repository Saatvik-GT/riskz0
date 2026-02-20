import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from .encoder import LabelEncoder, OneHotEncoder
from .scaler import StandardScaler
from .feature_engineer import FeatureEngineer
import sys

sys.path.append(".")
from config import TARGET_COLUMN, CLASS_LABELS


class PreprocessingPipeline:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(handle_unknown="ignore")
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()

        self.categorical_feature_names_: List[str] = []
        self.numerical_feature_names_: List[str] = []
        self.onehot_feature_names_: List[str] = []
        self.n_features_in_: int = 0
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "PreprocessingPipeline":
        y = df[TARGET_COLUMN].values
        self.label_encoder.fit(y)

        self.feature_engineer.fit(df, df.columns.tolist())

        X_categorical, X_numerical, cat_names, num_names = (
            self.feature_engineer.transform(df)
        )

        self.categorical_feature_names_ = cat_names
        self.numerical_feature_names_ = num_names

        if X_categorical.shape[1] > 0:
            self.onehot_encoder.fit(X_categorical)
            self.onehot_feature_names_ = self.onehot_encoder.get_feature_names(
                cat_names
            )

        if X_numerical.shape[1] > 0:
            self.scaler.fit(X_numerical)

        self.n_features_in_ = len(self.onehot_feature_names_) + len(
            self.numerical_feature_names_
        )
        self._fitted = True

        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise ValueError("Pipeline must be fitted before transform")

        X_categorical, X_numerical, _, _ = self.feature_engineer.transform(df)

        if X_categorical.shape[1] > 0:
            X_cat_encoded = self.onehot_encoder.transform(X_categorical)
        else:
            X_cat_encoded = np.array([]).reshape(len(df), 0)

        if X_numerical.shape[1] > 0:
            X_num_scaled = self.scaler.transform(X_numerical)
        else:
            X_num_scaled = np.array([]).reshape(len(df), 0)

        if X_cat_encoded.shape[1] > 0 and X_num_scaled.shape[1] > 0:
            X = np.hstack([X_cat_encoded, X_num_scaled])
        elif X_cat_encoded.shape[1] > 0:
            X = X_cat_encoded
        else:
            X = X_num_scaled

        if TARGET_COLUMN in df.columns:
            y = self.label_encoder.transform(df[TARGET_COLUMN].values)
        else:
            y = None

        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        return self.fit(df).transform(df)

    def inverse_transform_labels(self, y: np.ndarray) -> np.ndarray:
        return self.label_encoder.inverse_transform(y)

    def get_feature_names(self) -> List[str]:
        return self.onehot_feature_names_ + self.numerical_feature_names_

    def get_class_names(self) -> List[str]:
        return self.label_encoder.classes_

    def save(self, filepath: str):
        state = {
            "label_encoder": self.label_encoder.get_state(),
            "onehot_encoder": self.onehot_encoder.get_state(),
            "scaler": self.scaler.get_state(),
            "feature_engineer": self.feature_engineer.get_state(),
            "categorical_feature_names": self.categorical_feature_names_,
            "numerical_feature_names": self.numerical_feature_names_,
            "onehot_feature_names": self.onehot_feature_names_,
            "n_features_in": self.n_features_in_,
            "fitted": self._fitted,
        }
        np.save(filepath, state, allow_pickle=True)

    def load(self, filepath: str) -> "PreprocessingPipeline":
        state = np.load(filepath, allow_pickle=True).item()

        self.label_encoder.set_state(state["label_encoder"])
        self.onehot_encoder.set_state(state["onehot_encoder"])
        self.scaler.set_state(state["scaler"])
        self.feature_engineer.set_state(state["feature_engineer"])
        self.categorical_feature_names_ = state["categorical_feature_names"]
        self.numerical_feature_names_ = state["numerical_feature_names"]
        self.onehot_feature_names_ = state["onehot_feature_names"]
        self.n_features_in_ = state["n_features_in"]
        self._fitted = state["fitted"]

        return self
