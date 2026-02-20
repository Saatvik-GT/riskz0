import numpy as np
from typing import Dict, List, Optional, Tuple
import sys

sys.path.append(".")
from config import DROP_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES


class FeatureEngineer:
    def __init__(self):
        self.drop_features = DROP_FEATURES.copy()
        self.categorical_features = CATEGORICAL_FEATURES.copy()
        self.numerical_features = NUMERICAL_FEATURES.copy()
        self.created_features_: List[str] = []
        self.final_feature_names_: List[str] = []
        self._fitted = False

    def fit(self, df, feature_names: List[str]) -> "FeatureEngineer":
        remaining_features = [f for f in feature_names if f not in self.drop_features]

        self.categorical_features = [
            f for f in remaining_features if f in CATEGORICAL_FEATURES
        ]
        self.numerical_features = [
            f for f in remaining_features if f in NUMERICAL_FEATURES
        ]

        self.created_features_ = [
            "Risk_Pressure_Index",
            "Complexity_Experience_Ratio",
            "Resource_Risk_Score",
            "Timeline_Risk_Factor",
            "Budget_Per_Member",
            "Dependency_Pressure",
            "Stakeholder_Complexity",
            "Team_Stability_Index",
            "Schedule_Risk_Score",
            "Vendor_Dependency_Risk",
            "Technical_Risk_Composite",
            "Communication_Adequacy",
        ]

        self._fitted = True
        return self

    def _safe_col(self, df, col_name, default_val, n_samples):
        """Safely extract a column from a DataFrame."""
        if col_name in df.columns:
            return df[col_name].values.astype(np.float64)
        return np.full(n_samples, default_val, dtype=np.float64)

    def transform(self, df) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        if not self._fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")

        cat_data = []
        for col in self.categorical_features:
            if col in df.columns:
                cat_data.append(df[col].values.reshape(-1, 1))

        if cat_data:
            X_categorical = np.hstack(cat_data)
        else:
            X_categorical = np.array([]).reshape(len(df), 0)

        num_data = []
        for col in self.numerical_features:
            if col in df.columns:
                num_data.append(df[col].values.reshape(-1, 1))

        if num_data:
            X_numerical = np.hstack(num_data)
        else:
            X_numerical = np.array([]).reshape(len(df), 0)

        engineered = self._create_engineered_features(df)

        if X_numerical.shape[1] > 0:
            X_numerical = np.hstack([X_numerical, engineered])
        else:
            X_numerical = engineered

        return (
            X_categorical,
            X_numerical,
            self.categorical_features,
            self.numerical_features + self.created_features_,
        )

    def _create_engineered_features(self, df) -> np.ndarray:
        n_samples = len(df)
        n_features = 12
        engineered = np.zeros((n_samples, n_features))

        complexity = self._safe_col(df, "Complexity_Score", 5.0, n_samples)
        turnover = self._safe_col(df, "Team_Turnover_Rate", 0.0, n_samples)
        external_deps = self._safe_col(df, "External_Dependencies_Count", 0.0, n_samples)
        budget_util = self._safe_col(df, "Budget_Utilization_Rate", 1.0, n_samples)
        timeline = self._safe_col(df, "Estimated_Timeline_Months", 12.0, n_samples)
        prev_success = self._safe_col(df, "Previous_Delivery_Success_Rate", 0.7, n_samples)
        resource_avail = self._safe_col(df, "Resource_Availability", 0.5, n_samples)
        team_size = self._safe_col(df, "Team_Size", 10.0, n_samples)
        budget = self._safe_col(df, "Project_Budget_USD", 500000.0, n_samples)
        change_freq = self._safe_col(df, "Change_Request_Frequency", 1.0, n_samples)
        stakeholder_count = self._safe_col(df, "Stakeholder_Count", 5.0, n_samples)
        geo_dist = self._safe_col(df, "Geographical_Distribution", 3.0, n_samples)
        schedule_pressure = self._safe_col(df, "Schedule_Pressure", 0.05, n_samples)
        vendor_reliability = self._safe_col(df, "Vendor_Reliability_Score", 0.7, n_samples)
        tech_debt = self._safe_col(df, "Technical_Debt_Level", 0.2, n_samples)
        integration_complex = self._safe_col(df, "Integration_Complexity", 5.0, n_samples)
        comm_freq = self._safe_col(df, "Communication_Frequency", 3.0, n_samples)

        # Original 4 features
        engineered[:, 0] = complexity * (1 + turnover) * (1 + external_deps / 10)
        engineered[:, 1] = complexity / (prev_success + 0.01)
        engineered[:, 2] = (1 - resource_avail) * budget_util * complexity
        engineered[:, 3] = timeline * (1 + external_deps / 5) * (1 - prev_success)

        # New 8 features
        engineered[:, 4] = budget / (team_size + 1)  # Budget per member
        engineered[:, 5] = external_deps * change_freq  # Dependency pressure
        engineered[:, 6] = stakeholder_count * geo_dist  # Stakeholder complexity
        engineered[:, 7] = (1 - turnover) * prev_success  # Team stability
        engineered[:, 8] = schedule_pressure * timeline  # Schedule risk
        engineered[:, 9] = (1 - vendor_reliability) * external_deps  # Vendor dep risk
        engineered[:, 10] = tech_debt * integration_complex  # Technical risk composite
        engineered[:, 11] = comm_freq / (team_size + 1)  # Communication adequacy

        return engineered

    def get_feature_names(self) -> Tuple[List[str], List[str]]:
        return (
            self.categorical_features,
            self.numerical_features + self.created_features_,
        )

    def get_state(self) -> Dict:
        return {
            "drop_features": self.drop_features,
            "categorical_features": self.categorical_features,
            "numerical_features": self.numerical_features,
            "created_features": self.created_features_,
            "fitted": self._fitted,
        }

    def set_state(self, state: Dict) -> "FeatureEngineer":
        self.drop_features = state["drop_features"]
        self.categorical_features = state["categorical_features"]
        self.numerical_features = state["numerical_features"]
        self.created_features_ = state["created_features"]
        self._fitted = state["fitted"]
        return self
