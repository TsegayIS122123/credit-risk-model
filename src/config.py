"""
Configuration parameters for Task 5 - Model Training and Tracking
"""
import os
from pathlib import Path
from datetime import datetime

class Config:
    """Central configuration for model training"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    
    # Data files (priority order - customer-level first)
    DATA_PATHS = [
        PROCESSED_DATA_DIR / "task4_customer_risk_mapping.csv",
        PROCESSED_DATA_DIR / "task4_manual_result.parquet",
        PROCESSED_DATA_DIR / "task4_target_engineered.parquet",
        DATA_DIR / "raw" / "data.csv"
    ]
    
    # Training configuration
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    RANDOM_STATE = 42
    CV_FOLDS = 5
    N_ITER_RANDOM_SEARCH = 10
    
    # Target column
    TARGET_COL = "is_high_risk"
    ID_COLS = ["CustomerId", "TransactionId", "TransactionStartTime"]
    
    # MLflow configuration
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME = f"credit_risk_modeling_{datetime.now().strftime('%Y%m%d')}"
    MLFLOW_ARTIFACT_PATH = "models"
    
    # Model configurations
    MODELS = {
        "logistic_regression": {
            "class": "sklearn.linear_model.LogisticRegression",
            "default_params": {
                "C": 1.0,
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
                "solver": "lbfgs"
            }
        },
        "random_forest": {
            "class": "sklearn.ensemble.RandomForestClassifier",
            "default_params": {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
                "n_jobs": -1
            }
        },
        "xgboost": {
            "class": "xgboost.XGBClassifier",
            "default_params": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": 10,
                "random_state": RANDOM_STATE,
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "n_jobs": -1
            }
        }
    }
    
    # Hyperparameter grids for tuning
    HYPERPARAM_GRIDS = {
        "logistic_regression": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "liblinear", "saga"],
            "class_weight": ["balanced", None, {0: 1, 1: 3}, {0: 1, 1: 5}],
            "max_iter": [500, 1000, 2000]
        },
        "random_forest": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": ["balanced", "balanced_subsample", None, {0: 1, 1: 3}]
        },
        "xgboost": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [3, 6, 9, 12],
            "learning_rate": [0.001, 0.01, 0.1, 0.3],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "scale_pos_weight": [1, 5, 10, 20, 50]
        }
    }
    
    # Evaluation metrics
    EVALUATION_METRICS = [
        "accuracy",
        "precision",
        "recall", 
        "f1",
        "roc_auc"
    ]
    
    # Feature selection
    MAX_CATEGORICAL_UNIQUE = 50
    MIN_FEATURE_IMPORTANCE = 0.001
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.MODELS_DIR,
            cls.REPORTS_DIR,
            cls.REPORTS_DIR / "plots",
            cls.REPORTS_DIR / "metrics"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
    
    @classmethod
    def find_data_file(cls):
        """Find the best available data file"""
        for path in cls.DATA_PATHS:
            if path.exists():
                print(f"✅ Found data file: {path}")
                return path
        
        raise FileNotFoundError(
            f"No data file found. Checked: {[str(p) for p in cls.DATA_PATHS]}"
        )