import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
import sys

sys.path.append(".")
from src.utils.metrics import accuracy_score


def permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    random_seed: int = 42,
) -> Dict:
    np.random.seed(random_seed)

    baseline_score = accuracy_score(y, model.predict(X))

    n_features = X.shape[1]
    importances = np.zeros(n_features)
    importances_std = np.zeros(n_features)

    for feature_idx in range(n_features):
        scores = []

        for _ in range(n_repeats):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feature_idx])

            score = accuracy_score(y, model.predict(X_permuted))
            scores.append(baseline_score - score)

        importances[feature_idx] = np.mean(scores)
        importances_std[feature_idx] = np.std(scores)

    result = {
        "importances_mean": importances,
        "importances_std": importances_std,
        "feature_names": feature_names,
        "baseline_score": baseline_score,
    }

    return result


def plot_feature_importance(
    importance_result: Dict,
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10),
) -> plt.Figure:
    importances = importance_result["importances_mean"]
    stds = importance_result["importances_std"]
    feature_names = importance_result["feature_names"]

    indices = np.argsort(importances)[::-1][:top_n]

    top_importances = importances[indices]
    top_stds = stds[indices]
    top_names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_importances)))

    bars = ax.barh(
        range(len(top_importances)),
        top_importances,
        xerr=top_stds,
        align="center",
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        capsize=3,
    )

    ax.set_yticks(range(len(top_importances)))
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()

    ax.set_xlabel("Importance (Decrease in Accuracy)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1)

    for i, (imp, std) in enumerate(zip(top_importances, top_stds)):
        if imp > 0:
            ax.text(imp + std + 0.002, i, f"{imp:.4f}", va="center", fontsize=8)

    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Feature importance plot saved to: {save_path}")

    return fig


def coefficients_importance(weights: np.ndarray, feature_names: List[str]) -> Dict:
    importance = np.mean(np.abs(weights), axis=1)

    return {
        "importances_mean": importance,
        "importances_std": np.zeros(len(importance)),
        "feature_names": feature_names,
    }


def get_top_features(importance_result: Dict, top_n: int = 10) -> List[Dict]:
    importances = importance_result["importances_mean"]
    feature_names = importance_result["feature_names"]

    indices = np.argsort(importances)[::-1][:top_n]

    top_features = []
    for i, idx in enumerate(indices):
        top_features.append(
            {
                "rank": i + 1,
                "feature": feature_names[idx],
                "importance": importances[idx],
            }
        )

    return top_features
