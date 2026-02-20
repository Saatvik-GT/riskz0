import numpy as np
from typing import Optional, List, Dict, Literal
import sys

sys.path.append(".")
from config import NUM_CLASSES


class KNNClassifier:
    def __init__(
        self,
        k: int = 5,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
        weighted: bool = False,
    ):
        self.k = k
        self.metric = metric
        self.weighted = weighted

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.n_classes: int = NUM_CLASSES

        self._fitted = False

    def _euclidean_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)

        cross_term = X1 @ X2.T

        distances_sq = X1_sq + X2_sq - 2 * cross_term

        distances_sq = np.maximum(distances_sq, 0)

        return np.sqrt(distances_sq)

    def _manhattan_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        n1 = X1.shape[0]
        n2 = X2.shape[0]

        distances = np.zeros((n1, n2))

        for i in range(n1):
            distances[i, :] = np.sum(np.abs(X1[i] - X2), axis=1)

        return distances

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        if self.metric == "euclidean":
            return self._euclidean_distance(X, self.X_train)
        elif self.metric == "manhattan":
            return self._manhattan_distance(X, self.X_train)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _get_k_neighbors(self, distances: np.ndarray) -> np.ndarray:
        k = min(self.k, len(self.y_train))

        neighbor_indices = np.argsort(distances, axis=1)[:, :k]

        return neighbor_indices

    def _majority_vote(
        self, neighbor_indices: np.ndarray, distances: np.ndarray
    ) -> np.ndarray:
        n_samples = neighbor_indices.shape[0]
        k = neighbor_indices.shape[1]

        neighbor_labels = self.y_train[neighbor_indices]

        if self.weighted:
            neighbor_distances = np.zeros((n_samples, k))
            for i in range(n_samples):
                neighbor_distances[i] = distances[i, neighbor_indices[i]]

            neighbor_distances = np.maximum(neighbor_distances, 1e-10)
            weights = 1.0 / neighbor_distances

            weighted_votes = np.zeros((n_samples, self.n_classes))
            for i in range(n_samples):
                for j in range(k):
                    label = neighbor_labels[i, j]
                    weighted_votes[i, label] += weights[i, j]

            predictions = np.argmax(weighted_votes, axis=1)
        else:
            predictions = np.zeros(n_samples, dtype=np.int64)
            for i in range(n_samples):
                labels = neighbor_labels[i]
                counts = np.bincount(labels, minlength=self.n_classes)
                predictions[i] = np.argmax(counts)

        return predictions

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y, dtype=np.int64)

        self.n_classes = max(NUM_CLASSES, int(np.max(y)) + 1)

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X, dtype=np.float64)

        distances = self._compute_distances(X)
        neighbor_indices = self._get_k_neighbors(distances)
        predictions = self._majority_vote(neighbor_indices, distances)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X, dtype=np.float64)
        n_samples = X.shape[0]

        distances = self._compute_distances(X)
        neighbor_indices = self._get_k_neighbors(distances)
        k = neighbor_indices.shape[1]

        neighbor_labels = self.y_train[neighbor_indices]

        if self.weighted:
            neighbor_distances = np.zeros((n_samples, k))
            for i in range(n_samples):
                neighbor_distances[i] = distances[i, neighbor_indices[i]]

            neighbor_distances = np.maximum(neighbor_distances, 1e-10)
            weights = 1.0 / neighbor_distances

            proba = np.zeros((n_samples, self.n_classes))
            for i in range(n_samples):
                for j in range(k):
                    label = neighbor_labels[i, j]
                    proba[i, label] += weights[i, j]

                proba[i] /= np.sum(proba[i])
        else:
            proba = np.zeros((n_samples, self.n_classes))
            for i in range(n_samples):
                labels = neighbor_labels[i]
                counts = np.bincount(labels, minlength=self.n_classes)
                proba[i] = counts / k

        return proba

    def get_state(self) -> Dict:
        return {
            "X_train": self.X_train,
            "y_train": self.y_train,
            "k": self.k,
            "metric": self.metric,
            "weighted": self.weighted,
            "n_classes": self.n_classes,
            "fitted": self._fitted,
        }

    def set_state(self, state: Dict) -> "KNNClassifier":
        self.X_train = state["X_train"]
        self.y_train = state["y_train"]
        self.k = state["k"]
        self.metric = state["metric"]
        self.weighted = state["weighted"]
        self.n_classes = state["n_classes"]
        self._fitted = state["fitted"]
        return self
