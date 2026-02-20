import numpy as np
from typing import Dict, List, Optional


class LabelEncoder:
    def __init__(self):
        self.classes_: List[str] = []
        self.class_to_index: Dict[str, int] = {}
        self.index_to_class: Dict[int, str] = {}
        self._fitted = False

    def fit(self, y: np.ndarray) -> "LabelEncoder":
        unique_classes = np.unique(y)
        self.classes_ = list(unique_classes)
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}
        self.index_to_class = {idx: cls for cls, idx in self.class_to_index.items()}
        self._fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("LabelEncoder must be fitted before transform")

        y = np.array(y)
        encoded = np.zeros(len(y), dtype=np.int64)

        for i, val in enumerate(y):
            if val in self.class_to_index:
                encoded[i] = self.class_to_index[val]
            else:
                raise ValueError(f"Unknown class: {val}")

        return encoded

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("LabelEncoder must be fitted before inverse_transform")

        y = np.array(y)
        decoded = []

        for idx in y:
            if idx in self.index_to_class:
                decoded.append(self.index_to_class[idx])
            else:
                raise ValueError(f"Unknown index: {idx}")

        return np.array(decoded)

    def get_state(self) -> Dict:
        return {
            "classes": self.classes_,
            "class_to_index": self.class_to_index,
            "index_to_class": self.index_to_class,
            "fitted": self._fitted,
        }

    def set_state(self, state: Dict) -> "LabelEncoder":
        self.classes_ = state["classes"]
        self.class_to_index = state["class_to_index"]
        self.index_to_class = state["index_to_class"]
        self._fitted = state["fitted"]
        return self


class OneHotEncoder:
    def __init__(self, handle_unknown: str = "ignore"):
        self.handle_unknown = handle_unknown
        self.categories_: Dict[int, List[str]] = {}
        self.category_to_index_: Dict[int, Dict[str, int]] = {}
        self.n_features_in_ = 0
        self.n_categories_per_feature_: List[int] = []
        self._fitted = False

    def fit(self, X: np.ndarray) -> "OneHotEncoder":
        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_in_ = X.shape[1]
        self.categories_ = {}
        self.category_to_index_ = {}
        self.n_categories_per_feature_ = []

        for col_idx in range(self.n_features_in_):
            unique_cats = np.unique(X[:, col_idx])
            unique_cats = [str(cat) for cat in unique_cats]

            self.categories_[col_idx] = unique_cats
            self.category_to_index_[col_idx] = {
                cat: idx for idx, cat in enumerate(unique_cats)
            }
            self.n_categories_per_feature_.append(len(unique_cats))

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("OneHotEncoder must be fitted before transform")

        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        total_categories = sum(self.n_categories_per_feature_)

        one_hot = np.zeros((n_samples, total_categories), dtype=np.float64)

        current_col = 0
        for col_idx in range(self.n_features_in_):
            n_cats = self.n_categories_per_feature_[col_idx]
            cat_mapping = self.category_to_index_[col_idx]

            for sample_idx in range(n_samples):
                val = str(X[sample_idx, col_idx])

                if val in cat_mapping:
                    cat_idx = cat_mapping[val]
                    one_hot[sample_idx, current_col + cat_idx] = 1.0
                elif self.handle_unknown == "error":
                    raise ValueError(f"Unknown category '{val}' in feature {col_idx}")

            current_col += n_cats

        return one_hot

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def get_feature_names(
        self, original_names: Optional[List[str]] = None
    ) -> List[str]:
        if not self._fitted:
            raise ValueError("OneHotEncoder must be fitted before get_feature_names")

        feature_names = []

        for col_idx in range(self.n_features_in_):
            prefix = (
                f"feature_{col_idx}"
                if original_names is None
                else original_names[col_idx]
            )

            for cat in self.categories_[col_idx]:
                feature_names.append(f"{prefix}_{cat}")

        return feature_names

    def get_state(self) -> Dict:
        return {
            "handle_unknown": self.handle_unknown,
            "categories": self.categories_,
            "category_to_index": self.category_to_index_,
            "n_features_in": self.n_features_in_,
            "n_categories_per_feature": self.n_categories_per_feature_,
            "fitted": self._fitted,
        }

    def set_state(self, state: Dict) -> "OneHotEncoder":
        self.handle_unknown = state["handle_unknown"]
        self.categories_ = state["categories"]
        self.category_to_index_ = state["category_to_index"]
        self.n_features_in_ = state["n_features_in"]
        self.n_categories_per_feature_ = state["n_categories_per_feature"]
        self._fitted = state["fitted"]
        return self
