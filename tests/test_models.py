import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    softmax,
    one_hot_encode,
)
from src.models.logistic_regression import LogisticRegression
from src.models.knn import KNNClassifier


class TestMetrics:
    def test_accuracy_perfect(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])

        acc = accuracy_score(y_true, y_pred)

        assert acc == 1.0

    def test_accuracy_partial(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 1, 3])

        acc = accuracy_score(y_true, y_pred)

        assert acc == 0.75

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])

        cm = confusion_matrix(y_true, y_pred)

        assert cm.shape == (3, 3)
        assert cm[0, 0] == 2

    def test_precision_score(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        prec = precision_score(y_true, y_pred, average="macro")

        assert 0 <= prec <= 1

    def test_recall_score(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        rec = recall_score(y_true, y_pred, average="macro")

        assert 0 <= rec <= 1

    def test_f1_score(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        f1 = f1_score(y_true, y_pred, average="macro")

        assert 0 <= f1 <= 1

    def test_softmax_sums_to_one(self):
        logits = np.array([[1, 2, 3], [4, 5, 6]])

        probs = softmax(logits)

        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_softmax_positive(self):
        logits = np.array([[1, 2, 3]])

        probs = softmax(logits)

        assert np.all(probs > 0)

    def test_one_hot_encode(self):
        y = np.array([0, 1, 2])

        one_hot = one_hot_encode(y, 3)

        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(one_hot, expected)


class TestLogisticRegression:
    @pytest.fixture
    def simple_data(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)
        return X, y

    def test_initialization(self):
        model = LogisticRegression(n_features=10, n_classes=4)

        assert model.n_features == 10
        assert model.n_classes == 4
        assert not model._fitted

    def test_fit_initializes_weights(self, simple_data):
        X, y = simple_data
        model = LogisticRegression(n_features=5, n_classes=3, epochs=10, verbose=False)

        model.fit(X, y)

        assert model._fitted
        assert model.weights is not None
        assert model.bias is not None
        assert model.weights.shape == (5, 3)

    def test_predict_returns_valid_classes(self, simple_data):
        X, y = simple_data
        model = LogisticRegression(n_features=5, n_classes=3, epochs=10, verbose=False)
        model.fit(X, y)

        predictions = model.predict(X)

        assert len(predictions) == len(y)
        assert np.all((predictions >= 0) & (predictions < 3))

    def test_predict_proba_sums_to_one(self, simple_data):
        X, y = simple_data
        model = LogisticRegression(n_features=5, n_classes=3, epochs=10, verbose=False)
        model.fit(X, y)

        probs = model.predict_proba(X)

        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_get_state_set_state(self, simple_data):
        X, y = simple_data
        model = LogisticRegression(n_features=5, n_classes=3, epochs=10, verbose=False)
        model.fit(X, y)

        state = model.get_state()
        new_model = LogisticRegression(n_features=5, n_classes=3).set_state(state)

        np.testing.assert_array_equal(new_model.weights, model.weights)
        assert new_model._fitted


class TestKNNClassifier:
    @pytest.fixture
    def simple_data(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)
        return X, y

    def test_initialization(self):
        model = KNNClassifier(k=5)

        assert model.k == 5
        assert not model._fitted

    def test_fit_stores_data(self, simple_data):
        X, y = simple_data
        model = KNNClassifier(k=3)

        model.fit(X, y)

        assert model._fitted
        assert model.X_train is not None
        assert model.y_train is not None

    def test_predict_returns_valid_classes(self, simple_data):
        X, y = simple_data
        model = KNNClassifier(k=3)
        model.fit(X, y)

        predictions = model.predict(X[:10])

        assert len(predictions) == 10
        assert np.all((predictions >= 0) & (predictions < 3))

    def test_predict_proba_sums_to_one(self, simple_data):
        X, y = simple_data
        model = KNNClassifier(k=3)
        model.fit(X, y)

        probs = model.predict_proba(X[:10])

        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_euclidean_distance(self, simple_data):
        X, y = simple_data
        model = KNNClassifier(k=3, metric="euclidean")
        model.fit(X, y)

        distances = model._euclidean_distance(X[:5], X[:5])

        np.testing.assert_array_almost_equal(np.diag(distances), 0)

    def test_manhattan_distance(self, simple_data):
        X, y = simple_data
        model = KNNClassifier(k=3, metric="manhattan")
        model.fit(X, y)

        distances = model._manhattan_distance(X[:5], X[:5])

        np.testing.assert_array_almost_equal(np.diag(distances), 0)

    def test_weighted_voting(self, simple_data):
        X, y = simple_data
        model = KNNClassifier(k=3, weighted=True)
        model.fit(X, y)

        predictions = model.predict(X[:10])

        assert len(predictions) == 10
