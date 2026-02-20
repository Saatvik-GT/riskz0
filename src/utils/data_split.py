import numpy as np
from typing import Tuple
import sys

sys.path.append(".")
from config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED


def stratified_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2,
    random_seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(random_seed)

    unique_classes = np.unique(y)

    train_indices = []
    test_indices = []

    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)

        n_test = max(1, int(len(cls_indices) * test_ratio))

        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1"
    )

    np.random.seed(random_seed)

    unique_classes = np.unique(y)

    train_indices = []
    val_indices = []
    test_indices = []

    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)

        n_total = len(cls_indices)
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))
        n_test = n_total - n_train - n_val

        if n_test < 1:
            n_test = 1
            n_val = n_train - 1

        train_indices.extend(cls_indices[:n_train])
        val_indices.extend(cls_indices[n_train : n_train + n_val])
        test_indices.extend(cls_indices[n_train + n_val :])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    return (
        X[train_indices],
        X[val_indices],
        X[test_indices],
        y[train_indices],
        y[val_indices],
        y[test_indices],
    )
