import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

from src.data_loader import load_raw_data, save_processed_data, PROJECT_ROOT
from src.preprocessing import TitanicPreprocessor
from src.feature_engineering import TitanicFeatureEngineer
from src.evaluate import calculate_metrics

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "titanic_model.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title", "Deck", "AgeGroup", "FareBin"]
NUMERICAL_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "FarePerPerson"]


def build_pipeline(classifier: Any) -> Pipeline:
    """
    Build complete preprocessing + feature engineering + classifier pipeline.
    """
    column_transformer = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", TitanicPreprocessor()),
            ("feature_engineer", TitanicFeatureEngineer(drop_raw_cols=False)),
            ("col_transform", column_transformer),
            ("classifier", classifier),
        ]
    )
    return pipeline


def evaluate_candidate_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple standard classification algorithms using 5-fold Stratified CV.
    """
    candidates = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, clf in candidates.items():
        pipe = build_pipeline(clf)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        roc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        results[name] = {
            "cv_accuracy_mean": float(scores.mean()),
            "cv_accuracy_std": float(scores.std()),
            "cv_roc_auc_mean": float(roc_scores.mean()),
            "cv_roc_auc_std": float(roc_scores.std()),
        }

    return results


def train_and_tune_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train and tune the best performing model via GridSearchCV.
    """
    base_pipeline = build_pipeline(RandomForestClassifier(random_state=42))

    param_grid = {
        "classifier__n_estimators": [100, 150],
        "classifier__max_depth": [4, 6, 8],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        base_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = float(grid_search.best_score_)

    tuning_info = {
        "best_params": best_params,
        "best_cv_accuracy": best_cv_score,
    }
    return best_pipeline, tuning_info


def run_training_pipeline() -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Execute full training pipeline from raw data to saved model artifact.
    """
    print("Loading raw training dataset...")
    raw_df = load_raw_data()

    # Create and save clean processed dataset
    print("Generating processed dataset...")
    preprocessor = TitanicPreprocessor()
    fe = TitanicFeatureEngineer(drop_raw_cols=False)
    processed_df = fe.transform(preprocessor.fit_transform(raw_df))
    save_processed_data(processed_df)
    print(f"Processed dataset saved ({len(processed_df)} rows).")

    X = raw_df.drop(columns=["Survived"])
    y = raw_df["Survived"]

    print("\nBenchmarking candidate models with 5-fold cross validation:")
    benchmark_results = evaluate_candidate_models(X, y)
    for model_name, res in benchmark_results.items():
        print(f" - {model_name:25s}: Acc = {res['cv_accuracy_mean']:.4f} (+/- {res['cv_accuracy_std']:.4f}), ROC-AUC = {res['cv_roc_auc_mean']:.4f}")

    print("\nFine-tuning best model (Random Forest Classifier)...")
    best_pipeline, tuning_info = train_and_tune_model(X, y)
    print(f"Best CV Accuracy: {tuning_info['best_cv_accuracy']:.4f}")
    print(f"Best Hyperparameters: {tuning_info['best_params']}")

    # Calculate train metrics
    y_pred = best_pipeline.predict(X)
    y_prob = best_pipeline.predict_proba(X)[:, 1]
    final_metrics = calculate_metrics(y.to_numpy(), y_pred, y_prob)

    # Extract feature importances
    classifier = best_pipeline.named_steps["classifier"]
    col_transformer = best_pipeline.named_steps["col_transform"]
    cat_feature_names = list(col_transformer.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERICAL_FEATURES + cat_feature_names
    feature_importances = classifier.feature_importances_.tolist()

    metadata = {
        "model_type": "RandomForestClassifier",
        "benchmark_results": benchmark_results,
        "tuning_info": tuning_info,
        "metrics": final_metrics,
        "feature_names": all_feature_names,
        "feature_importances": feature_importances,
    }

    # Persist model and metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTrained model successfully saved to: {MODEL_PATH}")
    print(f"Model metadata and metrics saved to: {METRICS_PATH}")

    return best_pipeline, metadata


if __name__ == "__main__":
    run_training_pipeline()
