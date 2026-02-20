import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict


def plot_learning_curves(
    train_history: List[float],
    val_history: List[float],
    metric_name: str = "Accuracy",
    title: str = "Learning Curves",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_history) + 1)

    ax.plot(
        epochs,
        train_history,
        "b-",
        linewidth=2,
        label=f"Training {metric_name}",
        marker="o",
        markersize=3,
    )
    ax.plot(
        epochs,
        val_history,
        "r-",
        linewidth=2,
        label=f"Validation {metric_name}",
        marker="s",
        markersize=3,
    )

    best_val_idx = (
        np.argmax(val_history)
        if metric_name.lower() in ["accuracy", "f1"]
        else np.argmin(val_history)
    )
    best_val = val_history[best_val_idx]

    ax.scatter(
        [best_val_idx + 1],
        [best_val],
        color="green",
        s=100,
        zorder=5,
        label=f"Best Val: {best_val:.4f}",
        edgecolors="black",
        linewidths=2,
    )
    ax.axvline(x=best_val_idx + 1, color="green", linestyle="--", alpha=0.5)

    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_name, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.set_xlim([1, len(train_history)])

    if metric_name.lower() in ["accuracy", "f1"]:
        ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Learning curve saved to: {save_path}")

    return fig


def plot_loss_history(
    train_loss: List[float],
    val_loss: List[float],
    title: str = "Training and Validation Loss",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_loss) + 1)

    ax.plot(
        epochs,
        train_loss,
        "b-",
        linewidth=2,
        label="Training Loss",
        marker="o",
        markersize=3,
    )
    ax.plot(
        epochs,
        val_loss,
        "r-",
        linewidth=2,
        label="Validation Loss",
        marker="s",
        markersize=3,
    )

    best_val_idx = np.argmin(val_loss)
    best_val = val_loss[best_val_idx]

    ax.scatter(
        [best_val_idx + 1],
        [best_val],
        color="green",
        s=100,
        zorder=5,
        label=f"Best Val Loss: {best_val:.4f}",
        edgecolors="black",
        linewidths=2,
    )
    ax.axvline(x=best_val_idx + 1, color="green", linestyle="--", alpha=0.5)

    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.set_xlim([1, len(train_loss)])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Loss history saved to: {save_path}")

    return fig


def plot_combined_history(
    history: Dict,
    title: str = "Training History",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    if "train_loss_history" in history and len(history["train_loss_history"]) > 0:
        epochs = range(1, len(history["train_loss_history"]) + 1)

        axes[0].plot(
            epochs,
            history["train_loss_history"],
            "b-",
            linewidth=2,
            label="Training Loss",
            marker="o",
            markersize=3,
        )
        if "val_loss_history" in history and len(history["val_loss_history"]) > 0:
            axes[0].plot(
                epochs,
                history["val_loss_history"],
                "r-",
                linewidth=2,
                label="Validation Loss",
                marker="s",
                markersize=3,
            )

        axes[0].set_xlabel("Epoch", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Loss", fontsize=11, fontweight="bold")
        axes[0].set_title("Loss Over Training", fontsize=13, fontweight="bold")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)

    if "train_acc_history" in history and len(history["train_acc_history"]) > 0:
        epochs = range(1, len(history["train_acc_history"]) + 1)

        axes[1].plot(
            epochs,
            history["train_acc_history"],
            "b-",
            linewidth=2,
            label="Training Accuracy",
            marker="o",
            markersize=3,
        )
        if "val_acc_history" in history and len(history["val_acc_history"]) > 0:
            axes[1].plot(
                epochs,
                history["val_acc_history"],
                "r-",
                linewidth=2,
                label="Validation Accuracy",
                marker="s",
                markersize=3,
            )

        axes[1].set_xlabel("Epoch", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Accuracy", fontsize=11, fontweight="bold")
        axes[1].set_title("Accuracy Over Training", fontsize=13, fontweight="bold")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Combined history saved to: {save_path}")

    return fig
