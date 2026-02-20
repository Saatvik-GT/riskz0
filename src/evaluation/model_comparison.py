import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import sys

sys.path.append(".")
from src.utils.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from config import CLASS_LABELS, COLOR_PALETTE


def compare_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    if class_names is None:
        class_names = CLASS_LABELS

    results = {}

    for model_name, model in models.items():
        y_pred = model.predict(X_test)

        results[model_name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "predictions": y_pred,
        }

    return results


def plot_model_comparison(
    comparison_results: Dict,
    metrics: List[str] = None,
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
) -> plt.Figure:
    if metrics is None:
        metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    model_names = list(comparison_results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)

    x = np.arange(n_models)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette("husl", n_metrics)

    for i, metric in enumerate(metrics):
        values = [comparison_results[model][metric] for model in model_names]

        bars = ax.bar(
            x + i * width,
            values,
            width,
            label=metric.replace("_", " ").title(),
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
        )

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(model_names, fontsize=11)

    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim([0, 1.15])
    ax.grid(axis="y", alpha=0.3)

    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random Baseline")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Model comparison saved to: {save_path}")

    return fig


def create_comparison_table(comparison_results: Dict) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append("MODEL COMPARISON RESULTS")
    lines.append("=" * 100)

    header = f"{'Model':<20} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}"
    lines.append(header)
    lines.append("-" * 100)

    for model_name, metrics in comparison_results.items():
        line = (
            f"{model_name:<20} "
            f"{metrics['accuracy']:>12.4f} "
            f"{metrics['precision_macro']:>12.4f} "
            f"{metrics['recall_macro']:>12.4f} "
            f"{metrics['f1_macro']:>12.4f}"
        )
        lines.append(line)

    lines.append("=" * 100)

    best_model = max(comparison_results.items(), key=lambda x: x[1]["accuracy"])
    lines.append(
        f"\nBest Model (by Accuracy): {best_model[0]} with {best_model[1]['accuracy']:.4f}"
    )
    lines.append("=" * 100)

    return "\n".join(lines)


def plot_class_distribution(
    y: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Class Distribution",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    if class_names is None:
        class_names = CLASS_LABELS

    unique, counts = np.unique(y, return_counts=True)

    fig, ax = plt.subplots(figsize=figsize)

    colors = COLOR_PALETTE[: len(unique)]

    bars = ax.bar(
        [class_names[i] for i in unique],
        counts,
        color=colors,
        edgecolor="black",
        linewidth=1,
    )

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{count}\n({count / len(y) * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Risk Level", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Class distribution saved to: {save_path}")

    return fig


def plot_per_class_metrics(
    comparison_results: Dict,
    class_names: Optional[List[str]] = None,
    title: str = "Per-Class Performance",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
) -> plt.Figure:
    if class_names is None:
        class_names = CLASS_LABELS

    n_models = len(comparison_results)
    n_classes = len(class_names)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    metrics_funcs = [
        (
            "Precision",
            lambda y_true, y_pred: precision_score(y_true, y_pred, average=None),
        ),
        ("Recall", lambda y_true, y_pred: recall_score(y_true, y_pred, average=None)),
        ("F1-Score", lambda y_true, y_pred: f1_score(y_true, y_pred, average=None)),
    ]

    for idx, (metric_name, metric_func) in enumerate(metrics_funcs):
        ax = axes[idx]

        x = np.arange(n_classes)
        width = 0.8 / n_models

        for i, (model_name, results) in enumerate(comparison_results.items()):
            y_test = results.get("y_test")
            y_pred = results["predictions"]

            if y_test is not None:
                metric_values = metric_func(y_test, y_pred)

                ax.bar(x + i * width, metric_values, width, label=model_name, alpha=0.8)

        ax.set_xlabel("Class", fontsize=10, fontweight="bold")
        ax.set_ylabel(metric_name, fontsize=10, fontweight="bold")
        ax.set_title(f"Per-Class {metric_name}", fontsize=12, fontweight="bold")
        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.legend(loc="best", fontsize=8)
        ax.set_ylim([0, 1.1])
        ax.grid(axis="y", alpha=0.3)

    ax_legend = axes[3]
    ax_legend.axis("off")
    ax_legend.text(
        0.5,
        0.5,
        "Per-Class Metrics\n\nHigher is better for all metrics.\n\n"
        "Precision: How many predicted positives are actually positive\n"
        "Recall: How many actual positives were correctly predicted\n"
        "F1-Score: Harmonic mean of Precision and Recall",
        ha="center",
        va="center",
        fontsize=11,
        transform=ax_legend.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Per-class metrics saved to: {save_path}")

    return fig
