"""
Evaluation module for Titanic ML project.
Calculates performance metrics and generates diagnostic evaluation plots.
"""

from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute key classification metrics.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Optional predicted probabilities for positive class (1).

    Returns:
        Dict containing accuracy, precision, recall, f1, roc_auc, and confusion matrix.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """
    Plot formatted confusion matrix heatmap.
    """
    if labels is None:
        labels = ["Did not survive (0)", "Survived (1)"]

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Label", fontsize=11, labelpad=8)
    plt.tight_layout()
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Titanic Model",
) -> plt.Figure:
    """
    Plot ROC Curve with AUC score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2563EB", lw=2, label=f"{model_name} (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Random Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 15,
    title: str = "Top Feature Importances",
) -> plt.Figure:
    """
    Plot bar chart of top feature importances.
    """
    indices = np.argsort(importances)[::-1][:top_n]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(range(len(sorted_features)), sorted_importances[::-1], color="#3B82F6", align="center")
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features[::-1], fontsize=10)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def evaluate_model(
    model: Any, X_val: pd.DataFrame, y_val: pd.Series, model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Perform full evaluation on a validation dataset.
    """
    y_pred = model.predict(X_val)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val)[:, 1]

    metrics = calculate_metrics(y_val.to_numpy(), y_pred, y_prob)
    metrics["model_name"] = model_name
    return metrics
