# Paths
DATA_PATH = "data/raw/project_risk_raw_dataset.csv"
PROCESSED_DATA_DIR = "data/processed/"
MODELS_DIR = "models/"
REPORTS_DIR = "reports/"
FIGURES_DIR = "reports/figures/"

# Features to DROP (multicollinear or identifier)
DROP_FEATURES = [
    "Project_ID",
    "Tech_Environment_Stability",
]

# Categorical features (all object-type columns except Project_ID and Risk_Level)
CATEGORICAL_FEATURES = [
    "Project_Type",
    "Methodology_Used",
    "Team_Experience_Level",
    "Project_Phase",
    "Requirement_Stability",
    "Regulatory_Compliance_Level",
    "Technology_Familiarity",
    "Stakeholder_Engagement_Level",
    "Executive_Sponsorship",
    "Funding_Source",
    "Priority_Level",
    "Project_Manager_Experience",
    "Org_Process_Maturity",
    "Data_Security_Requirements",
    "Key_Stakeholder_Availability",
    "Contract_Type",
    "Resource_Contention_Level",
    "Industry_Volatility",
    "Client_Experience_Level",
    "Change_Control_Maturity",
    "Risk_Management_Maturity",
    "Team_Colocation",
    "Documentation_Quality",
]

# Numerical features (remaining after dropping multicollinear)
NUMERICAL_FEATURES = [
    "Complexity_Score",
    "Estimated_Timeline_Months",
    "External_Dependencies_Count",
    "Change_Request_Frequency",
    "Team_Turnover_Rate",
    "Vendor_Reliability_Score",
    "Historical_Risk_Incidents",
    "Communication_Frequency",
    "Geographical_Distribution",
    "Schedule_Pressure",
    "Budget_Utilization_Rate",
    "Market_Volatility",
    "Integration_Complexity",
    "Resource_Availability",
    "Organizational_Change_Frequency",
    "Cross_Functional_Dependencies",
    "Previous_Delivery_Success_Rate",
    "Technical_Debt_Level",
    "Project_Start_Month",
    "Current_Phase_Duration_Months",
    "Seasonal_Risk_Factor",
    "Past_Similar_Projects",
    "Team_Size",
    "Project_Budget_USD",
    "Stakeholder_Count",
]

# Target
TARGET_COLUMN = "Risk_Level"
CLASS_LABELS = ["Critical", "High", "Medium", "Low"]
NUM_CLASSES = 4

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Logistic Regression hyperparameters (baseline)
LOGREG_LEARNING_RATE = 0.1
LOGREG_EPOCHS = 1000
LOGREG_L2_LAMBDA = 0.001
LOGREG_BATCH_SIZE = 64
LOGREG_EARLY_STOPPING_PATIENCE = 50

# Improved Logistic Regression hyperparameters
IMPROVED_LOGREG_LEARNING_RATE = 0.1
IMPROVED_LOGREG_EPOCHS = 3000
IMPROVED_LOGREG_L2_LAMBDA = 0.0005
IMPROVED_LOGREG_BATCH_SIZE = 32
IMPROVED_LOGREG_EARLY_STOPPING_PATIENCE = 200
IMPROVED_LOGREG_MOMENTUM = 0.3
IMPROVED_LOGREG_LR_SCHEDULE = "step"
IMPROVED_LOGREG_GRAD_CLIP = 10.0
IMPROVED_LOGREG_USE_CLASS_WEIGHTS = False

# k-NN hyperparameters
KNN_K_VALUES = [3, 5, 7, 9, 11, 15, 21]
KNN_DISTANCE_METRICS = ["euclidean", "manhattan"]

# Neural Network hyperparameters (conditional)
NN_HIDDEN_LAYERS = [64, 32, 16]
NN_LEARNING_RATE = 0.001
NN_EPOCHS = 500
NN_BATCH_SIZE = 32
NN_DROPOUT_RATE = 0.3
NN_L2_LAMBDA = 0.0001
NN_EARLY_STOPPING_PATIENCE = 30

# Cross-validation
CV_FOLDS = 5

# Random seed for reproducibility
RANDOM_SEED = 42

# Visualization settings
FIGURE_SIZE = (10, 8)
COLOR_PALETTE = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
CONFUSION_MATRIX_CMAP = "Blues"
