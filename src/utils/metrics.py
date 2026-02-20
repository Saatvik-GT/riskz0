import numpy as np
from typing import Tuple, Dict, List, Union


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.array(logits, dtype=np.float64)

    if logits.ndim == 1:
        logits = logits.reshape(1, -1)

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def one_hot_encode(y: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.array(y, dtype=np.int64)
    n_samples = len(y)

    one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
    one_hot[np.arange(n_samples), y] = 1.0

    return one_hot


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(y_true == y_pred)


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = None
) -> np.ndarray:
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)

    if n_classes is None:
        n_classes = max(np.max(y_true), np.max(y_pred)) + 1

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1

    return cm


def precision_score(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
) -> Union[float, np.ndarray]:
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)

    n_classes = max(np.max(y_true), np.max(y_pred)) + 1
    cm = confusion_matrix(y_true, y_pred, n_classes)

    precisions = []
    for cls in range(n_classes):
        tp = cm[cls, cls]
        fp = np.sum(cm[:, cls]) - tp

        if tp + fp == 0:
            precisions.append(0.0)
        else:
            precisions.append(tp / (tp + fp))

    precisions = np.array(precisions)

    if average == "macro":
        return np.mean(precisions)
    elif average == "weighted":
        class_counts = np.sum(cm, axis=1)
        return np.sum(precisions * class_counts) / np.sum(class_counts)
    elif average is None or average == "none":
        return precisions
    else:
        raise ValueError(f"Unknown average type: {average}")


def recall_score(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
) -> Union[float, np.ndarray]:
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)

    n_classes = max(np.max(y_true), np.max(y_pred)) + 1
    cm = confusion_matrix(y_true, y_pred, n_classes)

    recalls = []
    for cls in range(n_classes):
        tp = cm[cls, cls]
        fn = np.sum(cm[cls, :]) - tp

        if tp + fn == 0:
            recalls.append(0.0)
        else:
            recalls.append(tp / (tp + fn))

    recalls = np.array(recalls)

    if average == "macro":
        return np.mean(recalls)
    elif average == "weighted":
        class_counts = np.sum(cm, axis=1)
        return np.sum(recalls * class_counts) / np.sum(class_counts)
    elif average is None or average == "none":
        return recalls
    else:
        raise ValueError(f"Unknown average type: {average}")


def f1_score(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
) -> Union[float, np.ndarray]:
    precisions = precision_score(y_true, y_pred, average=None)
    recalls = recall_score(y_true, y_pred, average=None)

    f1s = []
    for p, r in zip(precisions, recalls):
        if p + r == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * p * r / (p + r))

    f1s = np.array(f1s)

    if average == "macro":
        return np.mean(f1s)
    elif average == "weighted":
        y_true = np.array(y_true, dtype=np.int64)
        y_pred = np.array(y_pred, dtype=np.int64)
        n_classes = max(np.max(y_true), np.max(y_pred)) + 1
        cm = confusion_matrix(y_true, y_pred, n_classes)
        class_counts = np.sum(cm, axis=1)
        return np.sum(f1s * class_counts) / np.sum(class_counts)
    elif average is None or average == "none":
        return f1s
    else:
        raise ValueError(f"Unknown average type: {average}")


def classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None
) -> str:
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)

    n_classes = max(np.max(y_true), np.max(y_pred)) + 1

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    cm = confusion_matrix(y_true, y_pred, n_classes)

    precisions = precision_score(y_true, y_pred, average=None)
    recalls = recall_score(y_true, y_pred, average=None)
    f1s = f1_score(y_true, y_pred, average=None)

    support = np.sum(cm, axis=1)

    lines = []
    lines.append("=" * 70)
    lines.append("Classification Report")
    lines.append("=" * 70)
    lines.append(
        f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}"
    )
    lines.append("-" * 70)

    for i in range(n_classes):
        lines.append(
            f"{class_names[i]:<15} {precisions[i]:>10.4f} {recalls[i]:>10.4f} "
            f"{f1s[i]:>10.4f} {support[i]:>10d}"
        )

    lines.append("-" * 70)

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    total_support = np.sum(support)

    lines.append(
        f"{'Macro Avg':<15} {macro_precision:>10.4f} {macro_recall:>10.4f} "
        f"{macro_f1:>10.4f} {total_support:>10d}"
    )

    weighted_precision = precision_score(y_true, y_pred, average="weighted")
    weighted_recall = recall_score(y_true, y_pred, average="weighted")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    lines.append(
        f"{'Weighted Avg':<15} {weighted_precision:>10.4f} {weighted_recall:>10.4f} "
        f"{weighted_f1:>10.4f} {total_support:>10d}"
    )

    lines.append("=" * 70)

    acc = accuracy_score(y_true, y_pred)
    lines.append(f"Accuracy: {acc:.4f}")
    lines.append("=" * 70)

    return "\n".join(lines)
