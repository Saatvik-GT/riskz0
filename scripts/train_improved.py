"""
Improved Logistic Regression Training Script

Trains the original baseline (with original feature set) vs the improved model
(with new features + tuned hyperparameters) for a fair comparison.

Usage:
    python scripts/train_improved.py

Outputs:
    - Console: side-by-side accuracy/precision/recall/F1 comparison
    - reports/figures/improved_vs_baseline_comparison.png
    - reports/figures/improved_training_history.png
    - reports/figures/improved_confusion_matrix.png
    - reports/figures/improved_confusion_matrix_normalized.png
    - reports/figures/improved_feature_importance.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_PATH,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    LOGREG_LEARNING_RATE,
    LOGREG_EPOCHS,
    LOGREG_L2_LAMBDA,
    LOGREG_BATCH_SIZE,
    LOGREG_EARLY_STOPPING_PATIENCE,
    IMPROVED_LOGREG_LEARNING_RATE,
    IMPROVED_LOGREG_EPOCHS,
    IMPROVED_LOGREG_L2_LAMBDA,
    IMPROVED_LOGREG_BATCH_SIZE,
    IMPROVED_LOGREG_EARLY_STOPPING_PATIENCE,
    IMPROVED_LOGREG_MOMENTUM,
    IMPROVED_LOGREG_LR_SCHEDULE,
    IMPROVED_LOGREG_GRAD_CLIP,
    IMPROVED_LOGREG_USE_CLASS_WEIGHTS,
    CLASS_LABELS,
    RANDOM_SEED,
    TARGET_COLUMN,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.encoder import LabelEncoder, OneHotEncoder
from src.preprocessing.scaler import StandardScaler
from src.preprocessing.feature_engineer import FeatureEngineer
from src.utils.data_split import train_val_test_split
from src.utils.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from src.utils.serialization import save_model
from src.models.logistic_regression import LogisticRegression
from src.evaluation.confusion_matrix import (
    plot_confusion_matrix,
    plot_normalized_confusion_matrix,
    error_analysis,
)
from src.evaluation.feature_importance import (
    permutation_importance,
    plot_feature_importance,
)
from src.evaluation.learning_curves import plot_combined_history


# ── Original baseline config (before our changes) ───────────────────────
ORIGINAL_DROP_FEATURES = [
    "Project_ID",
    "Project_Budget_USD",
    "Team_Size",
    "Stakeholder_Count",
    "Tech_Environment_Stability",
]
ORIGINAL_NUMERICAL_FEATURES = [
    "Complexity_Score", "Estimated_Timeline_Months", "External_Dependencies_Count",
    "Change_Request_Frequency", "Team_Turnover_Rate", "Vendor_Reliability_Score",
    "Historical_Risk_Incidents", "Communication_Frequency", "Geographical_Distribution",
    "Schedule_Pressure", "Budget_Utilization_Rate", "Market_Volatility",
    "Integration_Complexity", "Resource_Availability", "Organizational_Change_Frequency",
    "Cross_Functional_Dependencies", "Previous_Delivery_Success_Rate",
    "Technical_Debt_Level", "Project_Start_Month", "Current_Phase_Duration_Months",
    "Seasonal_Risk_Factor", "Past_Similar_Projects",
]


class OriginalFeatureEngineer:
    """Feature engineer that mimics the ORIGINAL behavior (4 engineered features only)."""
    def __init__(self):
        self.drop_features = ORIGINAL_DROP_FEATURES.copy()
        self.categorical_features = CATEGORICAL_FEATURES.copy()
        self.numerical_features = ORIGINAL_NUMERICAL_FEATURES.copy()
        self.created_features_ = []
        self._fitted = False

    def fit(self, df, feature_names):
        remaining = [f for f in feature_names if f not in self.drop_features]
        self.categorical_features = [f for f in remaining if f in CATEGORICAL_FEATURES]
        self.numerical_features = [f for f in remaining if f in ORIGINAL_NUMERICAL_FEATURES]
        self.created_features_ = [
            "Risk_Pressure_Index", "Complexity_Experience_Ratio",
            "Resource_Risk_Score", "Timeline_Risk_Factor",
        ]
        self._fitted = True
        return self

    def transform(self, df):
        if not self._fitted:
            raise ValueError("Must be fitted first")
        cat_data = []
        for col in self.categorical_features:
            if col in df.columns:
                cat_data.append(df[col].values.reshape(-1, 1))
        X_cat = np.hstack(cat_data) if cat_data else np.array([]).reshape(len(df), 0)

        num_data = []
        for col in self.numerical_features:
            if col in df.columns:
                num_data.append(df[col].values.reshape(-1, 1))
        X_num = np.hstack(num_data) if num_data else np.array([]).reshape(len(df), 0)

        engineered = self._create_engineered_features(df)
        if X_num.shape[1] > 0:
            X_num = np.hstack([X_num, engineered])
        else:
            X_num = engineered

        return X_cat, X_num, self.categorical_features, self.numerical_features + self.created_features_

    def _create_engineered_features(self, df):
        n = len(df)
        eng = np.zeros((n, 4))
        def _col(name, default):
            return df[name].values.astype(np.float64) if name in df.columns else np.full(n, default)

        complexity = _col("Complexity_Score", 5.0)
        turnover = _col("Team_Turnover_Rate", 0.0)
        ext_deps = _col("External_Dependencies_Count", 0.0)
        budget_util = _col("Budget_Utilization_Rate", 1.0)
        timeline = _col("Estimated_Timeline_Months", 12.0)
        prev_success = _col("Previous_Delivery_Success_Rate", 0.7)
        resource = _col("Resource_Availability", 0.5)

        eng[:, 0] = complexity * (1 + turnover) * (1 + ext_deps / 10)
        eng[:, 1] = complexity / (prev_success + 0.01)
        eng[:, 2] = (1 - resource) * budget_util * complexity
        eng[:, 3] = timeline * (1 + ext_deps / 5) * (1 - prev_success)
        return eng

    def get_feature_names(self):
        return self.categorical_features, self.numerical_features + self.created_features_

    def get_state(self):
        return {"drop_features": self.drop_features, "categorical_features": self.categorical_features,
                "numerical_features": self.numerical_features, "created_features": self.created_features_,
                "fitted": self._fitted}

    def set_state(self, state):
        self.drop_features = state["drop_features"]
        self.categorical_features = state["categorical_features"]
        self.numerical_features = state["numerical_features"]
        self.created_features_ = state["created_features"]
        self._fitted = state["fitted"]
        return self


class OriginalPipeline:
    """Pipeline using original feature engineering."""
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(handle_unknown="ignore")
        self.scaler = StandardScaler()
        self.feature_engineer = OriginalFeatureEngineer()
        self.categorical_feature_names_ = []
        self.numerical_feature_names_ = []
        self.onehot_feature_names_ = []
        self.n_features_in_ = 0
        self._fitted = False

    def fit_transform(self, df):
        y = df[TARGET_COLUMN].values
        self.label_encoder.fit(y)
        self.feature_engineer.fit(df, df.columns.tolist())
        X_cat, X_num, cat_names, num_names = self.feature_engineer.transform(df)
        self.categorical_feature_names_ = cat_names
        self.numerical_feature_names_ = num_names
        if X_cat.shape[1] > 0:
            self.onehot_encoder.fit(X_cat)
            self.onehot_feature_names_ = self.onehot_encoder.get_feature_names(cat_names)
        if X_num.shape[1] > 0:
            self.scaler.fit(X_num)
        self.n_features_in_ = len(self.onehot_feature_names_) + len(self.numerical_feature_names_)
        self._fitted = True

        # Transform
        if X_cat.shape[1] > 0:
            X_cat_enc = self.onehot_encoder.transform(X_cat)
        else:
            X_cat_enc = np.array([]).reshape(len(df), 0)
        if X_num.shape[1] > 0:
            X_num_sc = self.scaler.transform(X_num)
        else:
            X_num_sc = np.array([]).reshape(len(df), 0)
        if X_cat_enc.shape[1] > 0 and X_num_sc.shape[1] > 0:
            X = np.hstack([X_cat_enc, X_num_sc])
        elif X_cat_enc.shape[1] > 0:
            X = X_cat_enc
        else:
            X = X_num_sc
        y_enc = self.label_encoder.transform(df[TARGET_COLUMN].values)
        return X, y_enc

    def get_feature_names(self):
        return self.onehot_feature_names_ + self.numerical_feature_names_


def train_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train a model and return test metrics."""
    print(f"\n{'='*70}")
    print(f"  TRAINING: {name}")
    print(f"{'='*70}")

    model.fit(X_train, y_train, X_val, y_val)
    y_pred_test = model.predict(X_test)
    y_pred_val = model.predict(X_val)

    metrics = {
        "val_acc": accuracy_score(y_val, y_pred_val),
        "test_acc": accuracy_score(y_test, y_pred_test),
        "test_precision": precision_score(y_test, y_pred_test, average="macro"),
        "test_recall": recall_score(y_test, y_pred_test, average="macro"),
        "test_f1": f1_score(y_test, y_pred_test, average="macro"),
        "test_precision_w": precision_score(y_test, y_pred_test, average="weighted"),
        "test_recall_w": recall_score(y_test, y_pred_test, average="weighted"),
        "test_f1_w": f1_score(y_test, y_pred_test, average="weighted"),
        "predictions": y_pred_test,
        "cm": confusion_matrix(y_test, y_pred_test),
    }

    print(f"\n  Validation Accuracy: {metrics['val_acc']:.4f}")
    print(f"  Test Accuracy:      {metrics['test_acc']:.4f}")
    print(f"\n{classification_report(y_test, y_pred_test, CLASS_LABELS)}")
    return metrics


def plot_comparison(baseline_metrics, improved_metrics, save_path):
    """Generate side-by-side bar chart comparing baseline vs improved."""
    metrics_names = ["Accuracy", "Precision\n(Macro)", "Recall\n(Macro)", "F1-Score\n(Macro)",
                     "Precision\n(Weighted)", "Recall\n(Weighted)", "F1-Score\n(Weighted)"]
    baseline_vals = [
        baseline_metrics["test_acc"], baseline_metrics["test_precision"],
        baseline_metrics["test_recall"], baseline_metrics["test_f1"],
        baseline_metrics["test_precision_w"], baseline_metrics["test_recall_w"],
        baseline_metrics["test_f1_w"],
    ]
    improved_vals = [
        improved_metrics["test_acc"], improved_metrics["test_precision"],
        improved_metrics["test_recall"], improved_metrics["test_f1"],
        improved_metrics["test_precision_w"], improved_metrics["test_recall_w"],
        improved_metrics["test_f1_w"],
    ]

    x = np.arange(len(metrics_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(16, 8))

    bars1 = ax.bar(x - width/2, baseline_vals, width, label="Baseline LogReg (Original Features)",
                   color="#e74c3c", edgecolor="black", linewidth=0.5, alpha=0.85)
    bars2 = ax.bar(x + width/2, improved_vals, width, label="Improved LogReg (Enhanced Features + Tuned)",
                   color="#2ecc71", edgecolor="black", linewidth=0.5, alpha=0.85)

    for bar, val in zip(bars1, baseline_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#c0392b")
    for bar, val in zip(bars2, improved_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#27ae60")

    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title("Baseline vs Improved Logistic Regression — Test Set Metrics", fontsize=15, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim([0, 1.18])
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.4)

    for i in range(len(metrics_names)):
        delta = improved_vals[i] - baseline_vals[i]
        sign = "+" if delta >= 0 else ""
        color = "#27ae60" if delta >= 0 else "#c0392b"
        max_val = max(baseline_vals[i], improved_vals[i])
        ax.text(x[i], max_val + 0.04, f"{sign}{delta:.4f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color=color, bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                       edgecolor=color, alpha=0.8))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\n  Comparison chart saved to: {save_path}")
    plt.close(fig)


def main():
    print("=" * 70)
    print("  IMPROVED LOGISTIC REGRESSION — FAIR COMPARISON")
    print("  Baseline: original features (4 engineered) + default hyperparams")
    print("  Improved: enhanced features (12 engineered) + tuned hyperparams")
    print("=" * 70)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"      Dataset shape: {df.shape}")

    # ── 2. Preprocess with ORIGINAL pipeline (for baseline) ─────────────
    print("\n[2/6] Preprocessing with original pipeline...")
    orig_pipeline = OriginalPipeline()
    X_orig, y = orig_pipeline.fit_transform(df)
    print(f"      Original feature matrix: {X_orig.shape} ({X_orig.shape[1]} features)")

    # ── 3. Preprocess with IMPROVED pipeline (for improved model) ───────
    print("\n      Preprocessing with improved pipeline...")
    imp_pipeline = PreprocessingPipeline()
    X_imp, y_imp = imp_pipeline.fit_transform(df)
    print(f"      Improved feature matrix: {X_imp.shape} ({X_imp.shape[1]} features)")

    imp_pipeline.save(os.path.join(PROCESSED_DATA_DIR, "preprocessing_pipeline.npy"))

    # ── 4. Split BOTH datasets (same random seed → same samples) ────────
    print("\n[3/6] Splitting data (70/15/15)...")
    X_train_o, X_val_o, X_test_o, y_train, y_val, y_test = train_val_test_split(
        X_orig, y, random_seed=RANDOM_SEED
    )
    X_train_i, X_val_i, X_test_i, _, _, _ = train_val_test_split(
        X_imp, y_imp, random_seed=RANDOM_SEED
    )
    print(f"      Train: {X_train_o.shape[0]}, Val: {X_val_o.shape[0]}, Test: {X_test_o.shape[0]}")

    # ── 5. Train BASELINE (original features + original hyperparams) ────
    print("\n[4/6] Training BASELINE Logistic Regression...")
    baseline = LogisticRegression(
        n_features=X_train_o.shape[1],
        learning_rate=LOGREG_LEARNING_RATE,
        epochs=LOGREG_EPOCHS,
        l2_lambda=LOGREG_L2_LAMBDA,
        batch_size=LOGREG_BATCH_SIZE,
        early_stopping_patience=LOGREG_EARLY_STOPPING_PATIENCE,
        verbose=True,
    )
    baseline_metrics = train_model(
        "Baseline LogReg (Original)", baseline,
        X_train_o, y_train, X_val_o, y_val, X_test_o, y_test
    )

    # ── 6. Train IMPROVED (new features + tuned hyperparams) ────────────
    print("\n[5/6] Training IMPROVED Logistic Regression...")
    improved = LogisticRegression(
        n_features=X_train_i.shape[1],
        learning_rate=IMPROVED_LOGREG_LEARNING_RATE,
        epochs=IMPROVED_LOGREG_EPOCHS,
        l2_lambda=IMPROVED_LOGREG_L2_LAMBDA,
        batch_size=IMPROVED_LOGREG_BATCH_SIZE,
        early_stopping_patience=IMPROVED_LOGREG_EARLY_STOPPING_PATIENCE,
        momentum=IMPROVED_LOGREG_MOMENTUM,
        lr_schedule=IMPROVED_LOGREG_LR_SCHEDULE,
        use_class_weights=IMPROVED_LOGREG_USE_CLASS_WEIGHTS,
        grad_clip=IMPROVED_LOGREG_GRAD_CLIP,
        verbose=True,
    )
    improved_metrics = train_model(
        "Improved LogReg (Enhanced)", improved,
        X_train_i, y_train, X_val_i, y_val, X_test_i, y_test
    )

    save_model(improved, os.path.join(MODELS_DIR, "improved_logistic_regression.npy"))

    # ── 7. Visualizations ───────────────────────────────────────────────
    print("\n[6/6] Generating visualizations...")

    plot_comparison(
        baseline_metrics, improved_metrics,
        os.path.join(FIGURES_DIR, "improved_vs_baseline_comparison.png"),
    )

    history = {
        "train_loss_history": improved.train_loss_history,
        "val_loss_history": improved.val_loss_history,
        "train_acc_history": improved.train_acc_history,
        "val_acc_history": improved.val_acc_history,
    }
    plot_combined_history(
        history,
        title="Improved Logistic Regression — Training History",
        save_path=os.path.join(FIGURES_DIR, "improved_training_history.png"),
    )

    plot_confusion_matrix(
        improved_metrics["cm"], CLASS_LABELS,
        title="Improved LogReg — Confusion Matrix",
        save_path=os.path.join(FIGURES_DIR, "improved_confusion_matrix.png"),
    )
    plot_normalized_confusion_matrix(
        improved_metrics["cm"], CLASS_LABELS,
        title="Improved LogReg — Normalized Confusion Matrix",
        save_path=os.path.join(FIGURES_DIR, "improved_confusion_matrix_normalized.png"),
    )

    feature_names = imp_pipeline.get_feature_names()
    print("      Computing feature importance (permutation)...")
    importance = permutation_importance(
        improved, X_test_i, y_test, feature_names, n_repeats=5, random_seed=RANDOM_SEED
    )
    plot_feature_importance(
        importance, top_n=25,
        title="Improved LogReg — Feature Importance (Permutation)",
        save_path=os.path.join(FIGURES_DIR, "improved_feature_importance.png"),
    )

    imp_error = error_analysis(improved_metrics["cm"], CLASS_LABELS)

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  Baseline:  {X_train_o.shape[1]} features, default hyperparams")
    print(f"  Improved:  {X_train_i.shape[1]} features, tuned hyperparams + momentum + LR schedule")
    print()

    header = f"{'Metric':<25} {'Baseline':>12} {'Improved':>12} {'Delta':>10}"
    print(header)
    print("-" * 70)

    comparisons = [
        ("Accuracy", baseline_metrics["test_acc"], improved_metrics["test_acc"]),
        ("Precision (Macro)", baseline_metrics["test_precision"], improved_metrics["test_precision"]),
        ("Recall (Macro)", baseline_metrics["test_recall"], improved_metrics["test_recall"]),
        ("F1-Score (Macro)", baseline_metrics["test_f1"], improved_metrics["test_f1"]),
        ("Precision (Weighted)", baseline_metrics["test_precision_w"], improved_metrics["test_precision_w"]),
        ("Recall (Weighted)", baseline_metrics["test_recall_w"], improved_metrics["test_recall_w"]),
        ("F1-Score (Weighted)", baseline_metrics["test_f1_w"], improved_metrics["test_f1_w"]),
    ]

    for name, base, imp in comparisons:
        delta = imp - base
        sign = "+" if delta >= 0 else ""
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"  {name:<23} {base:>12.4f} {imp:>12.4f} {sign}{delta:>9.4f} {arrow}")

    print("=" * 70)

    print(f"\n  Error Analysis (Improved):")
    print(f"    Total errors: {imp_error['total_errors']}")
    print(f"    Error rate:   {imp_error['error_rate']:.4f}")
    print("    Top confusion pairs:")
    for err in imp_error["top_errors"]:
        print(f"      {err['true_class']} → {err['predicted_class']}: {err['count']} times")

    print("\n" + "=" * 70)
    print("  HOW TO VIEW VISUALIZATIONS:")
    print("=" * 70)
    print("  Run these commands to open the generated figures:\n")
    print("    open reports/figures/improved_vs_baseline_comparison.png")
    print("    open reports/figures/improved_training_history.png")
    print("    open reports/figures/improved_confusion_matrix.png")
    print("    open reports/figures/improved_confusion_matrix_normalized.png")
    print("    open reports/figures/improved_feature_importance.png")
    print("\n  Or open all at once:")
    print("    open reports/figures/improved_*.png")
    print("=" * 70)

    return {
        "baseline_acc": baseline_metrics["test_acc"],
        "improved_acc": improved_metrics["test_acc"],
        "improvement": improved_metrics["test_acc"] - baseline_metrics["test_acc"],
    }


if __name__ == "__main__":
    results = main()
