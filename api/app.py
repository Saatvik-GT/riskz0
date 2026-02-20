from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    CLASS_LABELS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.serialization import load_model
from src.models.logistic_regression import LogisticRegression

app = Flask(__name__)
CORS(app)

pipeline = None
model = None


def load_models():
    global pipeline, model

    pipeline_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        PROCESSED_DATA_DIR,
        "preprocessing_pipeline.npy",
    )
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        MODELS_DIR,
        "improved_logistic_regression.npy",
    )

    pipeline = PreprocessingPipeline().load(pipeline_path)

    n_features = len(pipeline.get_feature_names())
    model = LogisticRegression(n_features=n_features)
    model = load_model(model, model_path)

    print("Models loaded successfully!")


load_models()

PROJECTS_DB = []

DEFAULT_VALUES = {
    "Project_Type": "IT",
    "Methodology_Used": "Agile",
    "Team_Experience_Level": "Senior",
    "Project_Phase": "Planning",
    "Requirement_Stability": "Moderate",
    "Regulatory_Compliance_Level": "Medium",
    "Technology_Familiarity": "Familiar",
    "Stakeholder_Engagement_Level": "Medium",
    "Executive_Sponsorship": "Moderate",
    "Funding_Source": "Internal",
    "Priority_Level": "Medium",
    "Project_Manager_Experience": "Mid-level PM",
    "Org_Process_Maturity": "Managed",
    "Data_Security_Requirements": "Medium",
    "Key_Stakeholder_Availability": "Good",
    "Contract_Type": "Time & Materials",
    "Resource_Contention_Level": "Medium",
    "Industry_Volatility": "Moderate",
    "Client_Experience_Level": "Regular",
    "Change_Control_Maturity": "Formal",
    "Risk_Management_Maturity": "Advanced",
    "Team_Colocation": "Hybrid",
    "Documentation_Quality": "Good",
    "Complexity_Score": 5.0,
    "Estimated_Timeline_Months": 12,
    "External_Dependencies_Count": 2,
    "Change_Request_Frequency": 1.0,
    "Team_Turnover_Rate": 0.15,
    "Vendor_Reliability_Score": 0.7,
    "Historical_Risk_Incidents": 1,
    "Communication_Frequency": 3.0,
    "Geographical_Distribution": 2,
    "Schedule_Pressure": 0.1,
    "Budget_Utilization_Rate": 0.9,
    "Market_Volatility": 0.3,
    "Integration_Complexity": 3.0,
    "Resource_Availability": 0.7,
    "Organizational_Change_Frequency": 1.0,
    "Cross_Functional_Dependencies": 3,
    "Previous_Delivery_Success_Rate": 0.75,
    "Technical_Debt_Level": 0.2,
    "Project_Start_Month": 1,
    "Current_Phase_Duration_Months": 3,
    "Seasonal_Risk_Factor": 1.0,
    "Past_Similar_Projects": 2,
    "Team_Size": 10,
    "Project_Budget_USD": 500000,
    "Stakeholder_Count": 5,
}


def fill_defaults(project_data):
    filled = DEFAULT_VALUES.copy()
    filled.update({k: v for k, v in project_data.items() if v is not None})
    return filled


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": model is not None,
            "pipeline_loaded": pipeline is not None,
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        project_data = data.get("project_data", {})

        if not project_data:
            return jsonify({"error": "No project data provided"}), 400

        filled_data = fill_defaults(project_data)

        df = pd.DataFrame([filled_data])

        X, _ = pipeline.transform(df)

        prediction_idx = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        class_names = pipeline.get_class_names()

        prediction_label = class_names[prediction_idx]

        risk_score = _calculate_risk_score(probabilities, prediction_idx)

        project_id = f"PROJ_{len(PROJECTS_DB) + 1:04d}"
        project_entry = {
            "id": project_id,
            "name": project_data.get("name", f"Project {len(PROJECTS_DB) + 1}"),
            "data": project_data,
            "prediction": {
                "risk_level": prediction_label,
                "risk_level_idx": int(prediction_idx),
                "confidence": float(probabilities[prediction_idx]),
                "probabilities": {
                    class_names[i]: float(prob) for i, prob in enumerate(probabilities)
                },
                "risk_score": risk_score,
            },
        }
        PROJECTS_DB.append(project_entry)

        return jsonify(
            {
                "success": True,
                "prediction": project_entry["prediction"],
                "project_id": project_id,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _calculate_risk_score(probabilities, prediction_idx):
    class_weights = [10, 7, 4, 1]

    score = sum(prob * weight for prob, weight in zip(probabilities, class_weights))

    return round(score, 2)


@app.route("/api/projects", methods=["GET"])
def get_projects():
    projects = []

    for proj in PROJECTS_DB:
        pred = proj["prediction"]
        status = _get_status_from_risk_level(pred["risk_level"])
        trend = _calculate_trend(pred["probabilities"])

        projects.append(
            {
                "project": proj["name"],
                "score": pred["risk_score"],
                "status": status,
                "trend": trend,
                "confidence": pred["confidence"],
                "risk_level": pred["risk_level"],
                "id": proj["id"],
            }
        )

    return jsonify({"projects": projects, "total": len(projects)})


def _get_status_from_risk_level(risk_level):
    mapping = {
        "Critical": "critical",
        "High": "warning",
        "Medium": "warning",
        "Low": "healthy",
    }
    return mapping.get(risk_level, "warning")


def _calculate_trend(probabilities):
    classes = list(probabilities.keys())
    probs = list(probabilities.values())

    if probs[0] > probs[-1] + 0.1:
        return "up"
    elif probs[-1] > probs[0] + 0.1:
        return "down"
    else:
        return "stable"


@app.route("/api/kpi", methods=["GET"])
def get_kpi():
    total = len(PROJECTS_DB)

    if total == 0:
        return jsonify(
            {
                "totalProjects": 0,
                "atRiskProjects": 0,
                "healthyProjects": 0,
                "avgRiskScore": 0,
                "avgConfidence": 0,
                "criticalCount": 0,
            }
        )

    at_risk = sum(
        1 for p in PROJECTS_DB if p["prediction"]["risk_level"] in ["Critical", "High"]
    )
    healthy = sum(1 for p in PROJECTS_DB if p["prediction"]["risk_level"] == "Low")
    critical = sum(
        1 for p in PROJECTS_DB if p["prediction"]["risk_level"] == "Critical"
    )

    avg_score = np.mean([p["prediction"]["risk_score"] for p in PROJECTS_DB])
    avg_confidence = np.mean([p["prediction"]["confidence"] for p in PROJECTS_DB])

    return jsonify(
        {
            "totalProjects": total,
            "atRiskProjects": at_risk,
            "healthyProjects": healthy,
            "avgRiskScore": round(avg_score, 2),
            "avgConfidence": round(avg_confidence * 100, 1),
            "criticalCount": critical,
            "completionRate": round((healthy / total) * 100, 1) if total > 0 else 0,
            "activeAlerts": critical + at_risk,
        }
    )


@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    alerts = []

    for proj in PROJECTS_DB:
        pred = proj["prediction"]

        if pred["risk_level"] == "Critical":
            alerts.append(
                {
                    "id": f"alert_{proj['id']}",
                    "type": "critical",
                    "message": f"{proj['name']} has CRITICAL risk level with {pred['confidence'] * 100:.1f}% confidence",
                    "time": "Just now",
                    "project": proj["name"],
                }
            )
        elif pred["risk_level"] == "High":
            alerts.append(
                {
                    "id": f"alert_{proj['id']}",
                    "type": "warning",
                    "message": f"{proj['name']} has HIGH risk level - attention needed",
                    "time": "Just now",
                    "project": proj["name"],
                }
            )

    return jsonify({"alerts": alerts, "total": len(alerts)})


@app.route("/api/insights", methods=["GET"])
def get_insights():
    insights = []

    if len(PROJECTS_DB) == 0:
        return jsonify({"insights": [], "total": 0})

    critical_projects = [
        p for p in PROJECTS_DB if p["prediction"]["risk_level"] == "Critical"
    ]
    high_risk_projects = [
        p for p in PROJECTS_DB if p["prediction"]["risk_level"] == "High"
    ]
    avg_score = np.mean([p["prediction"]["risk_score"] for p in PROJECTS_DB])

    if critical_projects:
        insights.append(
            {
                "id": 1,
                "title": f"{len(critical_projects)} Critical Projects Detected",
                "description": f"Projects {', '.join([p['name'] for p in critical_projects[:3]])} require immediate attention. Risk scores exceed threshold.",
                "priority": "high",
                "category": "risk",
            }
        )

    if avg_score > 6:
        insights.append(
            {
                "id": 2,
                "title": "High Average Risk Score",
                "description": f"Average risk score across all projects is {avg_score:.2f}, which is above the acceptable threshold of 6.0. Consider resource reallocation.",
                "priority": "high",
                "category": "performance",
            }
        )

    if high_risk_projects:
        insights.append(
            {
                "id": 3,
                "title": f"{len(high_risk_projects)} Projects Approaching Critical",
                "description": f"Monitor these projects closely to prevent escalation to critical status.",
                "priority": "medium",
                "category": "schedule",
            }
        )

    healthy_projects = [
        p for p in PROJECTS_DB if p["prediction"]["risk_level"] == "Low"
    ]
    if healthy_projects:
        insights.append(
            {
                "id": 4,
                "title": f"{len(healthy_projects)} Projects Performing Well",
                "description": f"Projects {', '.join([p['name'] for p in healthy_projects[:3]])} are on track with low risk scores.",
                "priority": "low",
                "category": "quality",
            }
        )

    return jsonify({"insights": insights, "total": len(insights)})


@app.route("/api/trends", methods=["GET"])
def get_trends():
    if len(PROJECTS_DB) < 2:
        return jsonify(
            {"trendData": [], "completionData": _get_completion_distribution()}
        )

    sorted_projects = sorted(PROJECTS_DB, key=lambda x: x["id"])

    trend_data = []
    window_size = min(5, len(sorted_projects))

    for i in range(max(1, len(sorted_projects) - 7), len(sorted_projects) + 1):
        window = sorted_projects[max(0, i - window_size) : i]

        if window:
            avg_score = np.mean([p["prediction"]["risk_score"] for p in window])
            avg_confidence = np.mean([p["prediction"]["confidence"] for p in window])

            trend_data.append(
                {
                    "week": f"W{i}",
                    "riskScore": round(avg_score, 2),
                    "confidence": round(avg_confidence * 100, 1),
                    "velocity": round(100 - avg_score * 10, 1),
                }
            )

    return jsonify(
        {"trendData": trend_data, "completionData": _get_completion_distribution()}
    )


def _get_completion_distribution():
    if len(PROJECTS_DB) == 0:
        return [
            {"name": "On Track", "value": 0, "color": "#10b981"},
            {"name": "At Risk", "value": 0, "color": "#f59e0b"},
            {"name": "Delayed", "value": 0, "color": "#ef4444"},
            {"name": "Critical", "value": 0, "color": "#dc2626"},
        ]

    distribution = {"healthy": 0, "warning": 0, "high": 0, "critical": 0}

    for proj in PROJECTS_DB:
        level = proj["prediction"]["risk_level"]
        if level == "Low":
            distribution["healthy"] += 1
        elif level == "Medium":
            distribution["warning"] += 1
        elif level == "High":
            distribution["high"] += 1
        else:
            distribution["critical"] += 1

    return [
        {"name": "On Track", "value": distribution["healthy"], "color": "#10b981"},
        {"name": "At Risk", "value": distribution["warning"], "color": "#f59e0b"},
        {"name": "High Risk", "value": distribution["high"], "color": "#f97316"},
        {"name": "Critical", "value": distribution["critical"], "color": "#ef4444"},
    ]


@app.route("/api/project/<project_id>", methods=["GET"])
def get_project(project_id):
    for proj in PROJECTS_DB:
        if proj["id"] == project_id:
            return jsonify(proj)

    return jsonify({"error": "Project not found"}), 404


@app.route("/api/clear", methods=["POST"])
def clear_projects():
    global PROJECTS_DB
    PROJECTS_DB = []
    return jsonify({"success": True, "message": "All projects cleared"})


@app.route("/api/form-fields", methods=["GET"])
def get_form_fields():
    return jsonify(
        {
            "categorical": {
                "Project_Type": [
                    "IT",
                    "Construction",
                    "R&D",
                    "Healthcare",
                    "Manufacturing",
                    "Marketing",
                ],
                "Methodology_Used": ["Agile", "Scrum", "Kanban", "Waterfall", "Hybrid"],
                "Team_Experience_Level": ["Junior", "Mixed", "Senior", "Expert"],
                "Project_Phase": [
                    "Initiation",
                    "Planning",
                    "Execution",
                    "Monitoring",
                    "Closure",
                ],
                "Requirement_Stability": ["Stable", "Moderate", "Volatile"],
                "Regulatory_Compliance_Level": ["Low", "Medium", "High", "Critical"],
                "Technology_Familiarity": ["New", "Familiar", "Expert"],
                "Stakeholder_Engagement_Level": [
                    "Poor",
                    "Low",
                    "Medium",
                    "High",
                    "Excellent",
                ],
                "Executive_Sponsorship": ["Weak", "Moderate", "Strong"],
                "Funding_Source": ["Internal", "External", "Government", "Mixed"],
                "Priority_Level": ["Low", "Medium", "High", "Critical"],
                "Project_Manager_Experience": [
                    "Junior PM",
                    "Mid-level PM",
                    "Senior PM",
                    "Certified PM",
                ],
                "Org_Process_Maturity": ["Ad-hoc", "Defined", "Managed", "Optimizing"],
                "Data_Security_Requirements": ["Low", "Medium", "High", "Strict"],
                "Key_Stakeholder_Availability": [
                    "Poor",
                    "Limited",
                    "Moderate",
                    "Good",
                    "Excellent",
                ],
                "Contract_Type": [
                    "Fixed-Price",
                    "Time & Materials",
                    "Cost-Plus",
                    "Hybrid",
                ],
                "Resource_Contention_Level": ["Low", "Medium", "High"],
                "Industry_Volatility": ["Stable", "Moderate", "High", "Extreme"],
                "Client_Experience_Level": [
                    "First-time",
                    "Occasional",
                    "Regular",
                    "Strategic",
                ],
                "Change_Control_Maturity": ["None", "Basic", "Advanced", "Formal"],
                "Risk_Management_Maturity": ["None", "Basic", "Advanced", "Formal"],
                "Team_Colocation": [
                    "Fully Remote",
                    "Hybrid",
                    "Partially Colocated",
                    "Fully Colocated",
                ],
                "Documentation_Quality": ["Poor", "Basic", "Good", "Excellent"],
            },
            "numerical": {
                "Complexity_Score": {
                    "min": 1.0,
                    "max": 10.0,
                    "default": 5.0,
                    "description": "Project complexity (1=Simple, 10=Very Complex)",
                },
                "Estimated_Timeline_Months": {"min": 1, "max": 48, "default": 12},
                "External_Dependencies_Count": {"min": 0, "max": 10, "default": 2},
                "Change_Request_Frequency": {"min": 0, "max": 10, "default": 1.0},
                "Team_Turnover_Rate": {
                    "min": 0,
                    "max": 1,
                    "default": 0.15,
                    "description": "0 to 1 (0=No turnover, 1=100% turnover)",
                },
                "Vendor_Reliability_Score": {"min": 0, "max": 1, "default": 0.7},
                "Historical_Risk_Incidents": {"min": 0, "max": 10, "default": 1},
                "Communication_Frequency": {"min": 0, "max": 20, "default": 3.0},
                "Geographical_Distribution": {
                    "min": 1,
                    "max": 5,
                    "default": 2,
                    "description": "1=Single location, 5=Global",
                },
                "Schedule_Pressure": {"min": 0, "max": 1, "default": 0.1},
                "Budget_Utilization_Rate": {"min": 0.5, "max": 1.5, "default": 0.9},
                "Market_Volatility": {"min": 0, "max": 1, "default": 0.3},
                "Integration_Complexity": {"min": 1, "max": 10, "default": 3.0},
                "Resource_Availability": {"min": 0, "max": 1, "default": 0.7},
                "Organizational_Change_Frequency": {
                    "min": 0,
                    "max": 10,
                    "default": 1.0,
                },
                "Cross_Functional_Dependencies": {"min": 0, "max": 10, "default": 3},
                "Previous_Delivery_Success_Rate": {"min": 0, "max": 1, "default": 0.75},
                "Technical_Debt_Level": {"min": 0, "max": 1, "default": 0.2},
                "Project_Start_Month": {"min": 1, "max": 12, "default": 1},
                "Current_Phase_Duration_Months": {"min": 1, "max": 24, "default": 3},
                "Seasonal_Risk_Factor": {"min": 0.9, "max": 1.2, "default": 1.0},
                "Past_Similar_Projects": {"min": 0, "max": 10, "default": 2},
                "Team_Size": {
                    "min": 1,
                    "max": 100,
                    "default": 10,
                    "description": "Number of team members",
                },
                "Project_Budget_USD": {
                    "min": 10000,
                    "max": 10000000,
                    "default": 500000,
                    "description": "Total project budget in USD",
                },
                "Stakeholder_Count": {
                    "min": 1,
                    "max": 50,
                    "default": 5,
                    "description": "Number of key stakeholders",
                },
            },
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
