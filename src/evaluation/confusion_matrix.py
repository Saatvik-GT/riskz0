import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict
import sys

sys.path.append(".")
from config import CLASS_LABELS, FIGURE_SIZE, CONFUSION_MATRIX_CMAP


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    if class_names is None:
        class_names = CLASS_LABELS

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=CONFUSION_MATRIX_CMAP,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Count"},
        linewidths=0.5,
        linecolor="gray",
    )

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    total = np.sum(cm)
    accuracy = np.trace(cm) / total

    stats_text = f"Accuracy: {accuracy:.4f}\nTotal Samples: {int(total)}"
    ax.text(
        1.02,
        0.5,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Confusion matrix saved to: {save_path}")

    return fig


def plot_normalized_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Normalized Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    if class_names is None:
        class_names = CLASS_LABELS

    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Proportion"},
        linewidths=0.5,
        linecolor="gray",
        vmin=0,
        vmax=1,
    )

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Normalized confusion matrix saved to: {save_path}")

    return fig


def error_analysis(
    cm: np.ndarray, class_names: Optional[List[str]] = None, top_n: int = 5
) -> Dict:
    if class_names is None:
        class_names = CLASS_LABELS

    n_classes = cm.shape[0]
    errors = []

    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                errors.append(
                    {
                        "true_class": class_names[i],
                        "predicted_class": class_names[j],
                        "count": int(cm[i, j]),
                        "true_class_idx": i,
                        "predicted_class_idx": j,
                    }
                )

    errors.sort(key=lambda x: x["count"], reverse=True)

    return {
        "top_errors": errors[:top_n],
        "total_errors": sum(e["count"] for e in errors),
        "error_rate": sum(e["count"] for e in errors) / np.sum(cm),
    }
