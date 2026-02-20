import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_DIR, MODELS_DIR, CLASS_LABELS
from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.metrics import classification_report
from src.utils.serialization import load_model
from src.models.logistic_regression import LogisticRegression
from src.models.knn import KNNClassifier


def predict_single(project_data: dict, model_type: str = "logreg"):
    pipeline_path = os.path.join(PROCESSED_DATA_DIR, "preprocessing_pipeline.npy")
    pipeline = PreprocessingPipeline().load(pipeline_path)

    df = pd.DataFrame([project_data])
    X, _ = pipeline.transform(df)

    if model_type == "logreg":
        model_path = os.path.join(MODELS_DIR, "logistic_regression.npy")
        model = LogisticRegression(n_features=X.shape[1])
        model = load_model(model, model_path)
    elif model_type == "knn":
        model_path = os.path.join(MODELS_DIR, "knn_classifier.npy")
        model = KNNClassifier()
        model = load_model(model, model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    prediction_idx = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    prediction_label = pipeline.inverse_transform_labels(np.array([prediction_idx]))[0]
    class_names = pipeline.get_class_names()

    return {
        "predicted_class": prediction_label,
        "predicted_class_idx": int(prediction_idx),
        "probabilities": {
            class_names[i]: float(prob) for i, prob in enumerate(probabilities)
        },
        "confidence": float(probabilities[prediction_idx]),
    }


def predict_batch(df: pd.DataFrame, model_type: str = "logreg"):
    pipeline_path = os.path.join(PROCESSED_DATA_DIR, "preprocessing_pipeline.npy")
    pipeline = PreprocessingPipeline().load(pipeline_path)

    X, _ = pipeline.transform(df)

    if model_type == "logreg":
        model_path = os.path.join(MODELS_DIR, "logistic_regression.npy")
        model = LogisticRegression(n_features=X.shape[1])
        model = load_model(model, model_path)
    elif model_type == "knn":
        model_path = os.path.join(MODELS_DIR, "knn_classifier.npy")
        model = KNNClassifier()
        model = load_model(model, model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    predictions_idx = model.predict(X)
    probabilities = model.predict_proba(X)

    predictions_labels = pipeline.inverse_transform_labels(predictions_idx)
    class_names = pipeline.get_class_names()

    results = []
    for i in range(len(df)):
        results.append(
            {
                "predicted_class": predictions_labels[i],
                "predicted_class_idx": int(predictions_idx[i]),
                "probabilities": {
                    class_names[j]: float(probabilities[i, j])
                    for j in range(len(class_names))
                },
                "confidence": float(probabilities[i, predictions_idx[i]]),
            }
        )

    return results


def print_prediction_report(prediction_result: dict):
    print("\n" + "=" * 60)
    print("PREDICTION REPORT")
    print("=" * 60)
    print(f"\nPredicted Risk Level: {prediction_result['predicted_class']}")
    print(f"Confidence: {prediction_result['confidence']:.2%}")
    print("\nClass Probabilities:")
    for cls, prob in prediction_result["probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"  {cls:10s}: {prob:.4f} {bar}")
    print("=" * 60)


if __name__ == "__main__":
    sample_project = {
        "Project_Type": "IT",
        "Team_Size": 10,
        "Project_Budget_USD": 500000,
        "Estimated_Timeline_Months": 12,
        "Complexity_Score": 7.5,
        "Stakeholder_Count": 8,
        "Methodology_Used": "Scrum",
        "Team_Experience_Level": "Senior",
        "Past_Similar_Projects": 2,
        "External_Dependencies_Count": 3,
        "Change_Request_Frequency": 1.5,
        "Project_Phase": "Execution",
        "Requirement_Stability": "Moderate",
        "Team_Turnover_Rate": 0.15,
        "Vendor_Reliability_Score": 0.8,
        "Historical_Risk_Incidents": 1,
        "Communication_Frequency": 3.5,
        "Regulatory_Compliance_Level": "Medium",
        "Technology_Familiarity": "Familiar",
        "Geographical_Distribution": 3,
        "Stakeholder_Engagement_Level": "High",
        "Schedule_Pressure": 0.1,
        "Budget_Utilization_Rate": 0.85,
        "Executive_Sponsorship": "Strong",
        "Funding_Source": "Internal",
        "Market_Volatility": 0.3,
        "Integration_Complexity": 4.5,
        "Resource_Availability": 0.7,
        "Priority_Level": "High",
        "Organizational_Change_Frequency": 1.0,
        "Cross_Functional_Dependencies": 4,
        "Previous_Delivery_Success_Rate": 0.75,
        "Technical_Debt_Level": 0.2,
        "Project_Manager_Experience": "Certified PM",
        "Org_Process_Maturity": "Managed",
        "Data_Security_Requirements": "Medium",
        "Key_Stakeholder_Availability": "Good",
        "Tech_Environment_Stability": "Modern/Stable",
        "Contract_Type": "Time & Materials",
        "Resource_Contention_Level": "Medium",
        "Industry_Volatility": "Moderate",
        "Client_Experience_Level": "Regular",
        "Change_Control_Maturity": "Formal",
        "Risk_Management_Maturity": "Advanced",
        "Team_Colocation": "Hybrid",
        "Documentation_Quality": "Good",
        "Project_Start_Month": 3,
        "Current_Phase_Duration_Months": 4,
        "Seasonal_Risk_Factor": 1.0,
    }

    print("Making prediction for sample project...")

    result_logreg = predict_single(sample_project, model_type="logreg")
    print("\n--- Logistic Regression Prediction ---")
    print_prediction_report(result_logreg)

    result_knn = predict_single(sample_project, model_type="knn")
    print("\n--- k-NN Prediction ---")
    print_prediction_report(result_knn)
