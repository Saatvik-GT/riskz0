import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_PATH,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    LOGREG_LEARNING_RATE,
    LOGREG_EPOCHS,
    LOGREG_L2_LAMBDA,
    LOGREG_BATCH_SIZE,
    LOGREG_EARLY_STOPPING_PATIENCE,
    KNN_K_VALUES,
    KNN_DISTANCE_METRICS,
    CLASS_LABELS,
    RANDOM_SEED,
)
from src.preprocessing.pipeline import PreprocessingPipeline
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
from src.models.knn import KNNClassifier
from src.evaluation.confusion_matrix import (
    plot_confusion_matrix,
    plot_normalized_confusion_matrix,
    error_analysis,
)
from src.evaluation.feature_importance import (
    permutation_importance,
    plot_feature_importance,
    coefficients_importance,
)
from src.evaluation.learning_curves import plot_combined_history, plot_learning_curves
from src.evaluation.model_comparison import (
    compare_models,
    plot_model_comparison,
    create_comparison_table,
)


def main():
    print("=" * 80)
    print("PROJECT RISK PREDICTION - BASELINE TRAINING")
    print("=" * 80)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("\n[1/7] Loading and preprocessing data...")
    df = pd.read_csv(DATA_PATH)
    print(f"      Dataset shape: {df.shape}")

    print("\n[2/7] Creating preprocessing pipeline...")
    pipeline = PreprocessingPipeline()
    X, y = pipeline.fit_transform(df)

    print(f"      Feature matrix shape: {X.shape}")
    print(f"      Target shape: {y.shape}")
    print(f"      Number of features: {X.shape[1]}")

    pipeline_path = os.path.join(PROCESSED_DATA_DIR, "preprocessing_pipeline.npy")
    pipeline.save(pipeline_path)
    print(f"      Pipeline saved to: {pipeline_path}")

    print("\n[3/7] Splitting data (70% train, 15% val, 15% test)...")
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, random_seed=RANDOM_SEED
    )

    print(f"      Train: {X_train.shape[0]} samples")
    print(f"      Val:   {X_val.shape[0]} samples")
    print(f"      Test:  {X_test.shape[0]} samples")

    print("\n[4/7] Training Logistic Regression...")
    logreg = LogisticRegression(
        n_features=X_train.shape[1],
        learning_rate=LOGREG_LEARNING_RATE,
        epochs=LOGREG_EPOCHS,
        l2_lambda=LOGREG_L2_LAMBDA,
        batch_size=LOGREG_BATCH_SIZE,
        early_stopping_patience=LOGREG_EARLY_STOPPING_PATIENCE,
        verbose=True,
    )

    logreg.fit(X_train, y_train, X_val, y_val)

    logreg_pred_val = logreg.predict(X_val)
    logreg_val_acc = accuracy_score(y_val, logreg_pred_val)
    print(f"\n      Logistic Regression Validation Accuracy: {logreg_val_acc:.4f}")

    logreg_path = os.path.join(MODELS_DIR, "logistic_regression.npy")
    save_model(logreg, logreg_path)
    print(f"      Model saved to: {logreg_path}")

    print("\n[5/7] Training k-NN (Grid Search for k)...")
    best_knn = None
    best_knn_acc = 0
    best_k = 0
    knn_results = {}

    for k in KNN_K_VALUES:
        for metric in KNN_DISTANCE_METRICS:
            knn = KNNClassifier(k=k, metric=metric, weighted=False)
            knn.fit(X_train, y_train)

            y_pred = knn.predict(X_val)
            acc = accuracy_score(y_val, y_pred)

            knn_results[f"k={k}_{metric}"] = acc

            if acc > best_knn_acc:
                best_knn_acc = acc
                best_knn = knn
                best_k = k

            print(f"      k={k:2d}, metric={metric:10s} -> Val Acc: {acc:.4f}")

    print(f"\n      Best k-NN: k={best_k}, Validation Accuracy: {best_knn_acc:.4f}")

    knn_path = os.path.join(MODELS_DIR, "knn_classifier.npy")
    save_model(best_knn, knn_path)
    print(f"      Model saved to: {knn_path}")

    print("\n[6/7] Evaluating on Test Set...")

    logreg_pred_test = logreg.predict(X_test)
    logreg_test_acc = accuracy_score(y_test, logreg_pred_test)

    knn_pred_test = best_knn.predict(X_test)
    knn_test_acc = accuracy_score(y_test, knn_pred_test)

    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION - TEST SET RESULTS")
    print("=" * 80)
    print(classification_report(y_test, logreg_pred_test, CLASS_LABELS))

    print("\n" + "=" * 80)
    print("k-NEAREST NEIGHBORS - TEST SET RESULTS")
    print("=" * 80)
    print(classification_report(y_test, knn_pred_test, CLASS_LABELS))

    models = {"Logistic Regression": logreg, f"k-NN (k={best_k})": best_knn}

    comparison_results = compare_models(models, X_test, y_test, CLASS_LABELS)

    for model_name in models:
        comparison_results[model_name]["y_test"] = y_test

    print("\n" + create_comparison_table(comparison_results))

    print("\n[7/7] Generating visualizations...")

    logreg_cm = confusion_matrix(y_test, logreg_pred_test)
    plot_confusion_matrix(
        logreg_cm,
        CLASS_LABELS,
        title="Logistic Regression - Confusion Matrix",
        save_path=os.path.join(FIGURES_DIR, "logreg_confusion_matrix.png"),
    )

    knn_cm = confusion_matrix(y_test, knn_pred_test)
    plot_confusion_matrix(
        knn_cm,
        CLASS_LABELS,
        title=f"k-NN (k={best_k}) - Confusion Matrix",
        save_path=os.path.join(FIGURES_DIR, "knn_confusion_matrix.png"),
    )

    plot_normalized_confusion_matrix(
        logreg_cm,
        CLASS_LABELS,
        title="Logistic Regression - Normalized Confusion Matrix",
        save_path=os.path.join(FIGURES_DIR, "logreg_confusion_matrix_normalized.png"),
    )

    plot_normalized_confusion_matrix(
        knn_cm,
        CLASS_LABELS,
        title=f"k-NN (k={best_k}) - Normalized Confusion Matrix",
        save_path=os.path.join(FIGURES_DIR, "knn_confusion_matrix_normalized.png"),
    )

    history = {
        "train_loss_history": logreg.train_loss_history,
        "val_loss_history": logreg.val_loss_history,
        "train_acc_history": logreg.train_acc_history,
        "val_acc_history": logreg.val_acc_history,
    }
    plot_combined_history(
        history,
        title="Logistic Regression - Training History",
        save_path=os.path.join(FIGURES_DIR, "logreg_training_history.png"),
    )

    feature_names = pipeline.get_feature_names()

    print("      Computing feature importance (permutation)...")
    importance_result = permutation_importance(
        logreg, X_test, y_test, feature_names, n_repeats=5, random_seed=RANDOM_SEED
    )

    plot_feature_importance(
        importance_result,
        top_n=20,
        title="Logistic Regression - Feature Importance (Permutation)",
        save_path=os.path.join(FIGURES_DIR, "logreg_feature_importance.png"),
    )

    plot_model_comparison(
        comparison_results,
        title="Baseline Models Comparison",
        save_path=os.path.join(FIGURES_DIR, "model_comparison.png"),
    )

    logreg_error = error_analysis(logreg_cm, CLASS_LABELS)
    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION - ERROR ANALYSIS")
    print("=" * 80)
    print(f"Total errors: {logreg_error['total_errors']}")
    print(f"Error rate: {logreg_error['error_rate']:.4f}")
    print("\nTop 5 confusion pairs:")
    for err in logreg_error["top_errors"]:
        print(
            f"  {err['true_class']} -> {err['predicted_class']}: {err['count']} times"
        )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nFinal Test Accuracies:")
    print(f"  Logistic Regression: {logreg_test_acc:.4f}")
    print(f"  k-NN (k={best_k}):        {knn_test_acc:.4f}")

    decision_threshold = 0.55
    proceed_to_nn = (
        logreg_test_acc < decision_threshold and knn_test_acc < decision_threshold
    )

    print(f"\nDecision Point: Threshold = {decision_threshold:.2f}")
    print(
        f"  Logistic Regression: {'PASS' if logreg_test_acc >= decision_threshold else 'FAIL'} ({logreg_test_acc:.4f})"
    )
    print(
        f"  k-NN:                {'PASS' if knn_test_acc >= decision_threshold else 'FAIL'} ({knn_test_acc:.4f})"
    )

    if proceed_to_nn:
        print("\n>>> RECOMMENDATION: Proceed to Neural Network implementation")
        print("    Baselines below threshold - NN may provide significant improvement")
    else:
        print("\n>>> RECOMMENDATION: Consider sticking with baselines")
        print(
            "    Strong baseline performance - NN may only provide marginal improvement"
        )

    return {
        "logreg_test_acc": logreg_test_acc,
        "knn_test_acc": knn_test_acc,
        "best_k": best_k,
        "proceed_to_nn": proceed_to_nn,
    }


if __name__ == "__main__":
    results = main()
