import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.encoder import LabelEncoder, OneHotEncoder
from src.preprocessing.scaler import MinMaxScaler, StandardScaler


class TestLabelEncoder:
    def test_fit_creates_mapping(self):
        encoder = LabelEncoder()
        y = np.array(["Low", "High", "Medium", "High", "Low"])
        encoder.fit(y)

        assert len(encoder.classes_) == 3
        assert "Low" in encoder.class_to_index
        assert "High" in encoder.class_to_index
        assert "Medium" in encoder.class_to_index

    def test_transform_returns_integers(self):
        encoder = LabelEncoder()
        y = np.array(["Low", "High", "Medium", "High", "Low"])
        encoder.fit(y)

        encoded = encoder.transform(y)

        assert encoded.dtype == np.int64
        assert len(encoded) == 5
        assert np.all(encoded >= 0)

    def test_inverse_transform_reverses(self):
        encoder = LabelEncoder()
        y = np.array(["Low", "High", "Medium", "High", "Low"])
        encoder.fit(y)

        encoded = encoder.transform(y)
        decoded = encoder.inverse_transform(encoded)

        np.testing.assert_array_equal(decoded, y)

    def test_get_state_set_state(self):
        encoder = LabelEncoder()
        y = np.array(["Low", "High", "Medium"])
        encoder.fit(y)

        state = encoder.get_state()
        new_encoder = LabelEncoder().set_state(state)

        np.testing.assert_array_equal(new_encoder.classes_, encoder.classes_)


class TestOneHotEncoder:
    def test_fit_creates_categories(self):
        encoder = OneHotEncoder()
        X = np.array([["A", "X"], ["B", "Y"], ["A", "X"]])
        encoder.fit(X)

        assert encoder.n_features_in_ == 2
        assert 0 in encoder.categories_
        assert 1 in encoder.categories_

    def test_transform_creates_binary_matrix(self):
        encoder = OneHotEncoder()
        X = np.array([["A", "X"], ["B", "Y"], ["A", "X"]])
        encoder.fit(X)

        one_hot = encoder.transform(X)

        assert one_hot.shape[0] == 3
        assert one_hot.shape[1] == 4
        assert np.all((one_hot == 0) | (one_hot == 1))

    def test_handle_unknown_ignore(self):
        encoder = OneHotEncoder(handle_unknown="ignore")
        X_train = np.array([["A"], ["B"]])
        X_test = np.array([["A"], ["C"]])

        encoder.fit(X_train)
        one_hot = encoder.transform(X_test)

        assert one_hot.shape[0] == 2
        assert one_hot[1, 0] == 0
        assert one_hot[1, 1] == 0

    def test_get_feature_names(self):
        encoder = OneHotEncoder()
        X = np.array([["A", "X"], ["B", "Y"]])
        encoder.fit(X)

        names = encoder.get_feature_names(["col1", "col2"])

        assert len(names) == 4
        assert "col1_A" in names
        assert "col1_B" in names


class TestMinMaxScaler:
    def test_fit_computes_min_max(self):
        scaler = MinMaxScaler()
        X = np.array([[1, 10], [2, 20], [3, 30]])
        scaler.fit(X)

        np.testing.assert_array_equal(scaler.data_min_, [1, 10])
        np.testing.assert_array_equal(scaler.data_max_, [3, 30])

    def test_transform_scales_to_range(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        X = np.array([[1, 10], [2, 20], [3, 30]])
        scaler.fit(X)

        X_scaled = scaler.transform(X)

        assert np.all(X_scaled >= 0)
        assert np.all(X_scaled <= 1)
        np.testing.assert_array_almost_equal(X_scaled[0], [0, 0])
        np.testing.assert_array_almost_equal(X_scaled[2], [1, 1])

    def test_inverse_transform_reverses(self):
        scaler = MinMaxScaler()
        X = np.array([[1, 10], [2, 20], [3, 30]])
        scaler.fit(X)

        X_scaled = scaler.transform(X)
        X_recovered = scaler.inverse_transform(X_scaled)

        np.testing.assert_array_almost_equal(X_recovered, X)

    def test_handles_constant_feature(self):
        scaler = MinMaxScaler()
        X = np.array([[1, 10], [1, 20], [1, 30]])
        scaler.fit(X)

        X_scaled = scaler.transform(X)

        assert not np.any(np.isnan(X_scaled))


class TestStandardScaler:
    def test_fit_computes_mean_std(self):
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler.fit(X)

        np.testing.assert_array_almost_equal(scaler.mean_, [3, 4])

    def test_transform_centers_data(self):
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler.fit(X)

        X_scaled = scaler.transform(X)

        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)

    def test_inverse_transform_reverses(self):
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler.fit(X)

        X_scaled = scaler.transform(X)
        X_recovered = scaler.inverse_transform(X_scaled)

        np.testing.assert_array_almost_equal(X_recovered, X)
