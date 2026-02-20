import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_split import stratified_train_test_split, train_val_test_split


class TestDataSplit:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.array([0] * 50 + [1] * 50 + [2] * 50 + [3] * 50)
        return X, y

    def test_stratified_split_maintains_ratio(self, sample_data):
        X, y = sample_data
        original_ratios = [np.mean(y == i) for i in range(4)]

        X_train, X_test, y_train, y_test = stratified_train_test_split(
            X, y, test_ratio=0.2, random_seed=42
        )

        train_ratios = [np.mean(y_train == i) for i in range(4)]
        test_ratios = [np.mean(y_test == i) for i in range(4)]

        for orig, train, test in zip(original_ratios, train_ratios, test_ratios):
            assert abs(orig - train) < 0.05
            assert abs(orig - test) < 0.05

    def test_stratified_split_sizes(self, sample_data):
        X, y = sample_data

        X_train, X_test, y_train, y_test = stratified_train_test_split(
            X, y, test_ratio=0.2, random_seed=42
        )

        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert abs(len(X_train) / len(X) - 0.8) < 0.05

    def test_train_val_test_split_sizes(self, sample_data):
        X, y = sample_data

        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
        )

        total = len(X)
        assert abs(len(X_train) / total - 0.7) < 0.05
        assert abs(len(X_val) / total - 0.15) < 0.05
        assert abs(len(X_test) / total - 0.15) < 0.05

    def test_train_val_test_split_stratification(self, sample_data):
        X, y = sample_data
        original_ratios = [np.mean(y == i) for i in range(4)]

        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
        )

        for split_y in [y_train, y_val, y_test]:
            split_ratios = [np.mean(split_y == i) for i in range(4)]
            for orig, split in zip(original_ratios, split_ratios):
                assert abs(orig - split) < 0.1

    def test_reproducibility(self, sample_data):
        X, y = sample_data

        X_train1, X_test1, y_train1, y_test1 = stratified_train_test_split(
            X, y, test_ratio=0.2, random_seed=42
        )
        X_train2, X_test2, y_train2, y_test2 = stratified_train_test_split(
            X, y, test_ratio=0.2, random_seed=42
        )

        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(y_train1, y_train2)
