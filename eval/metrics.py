"""Reusable metrics for Phase 8 evaluation."""

from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)

from src.intents import get_all_intents


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute metrics while retaining every requested taxonomy label."""
    labels = labels or get_all_intents()
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_intent = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    return {
        "labels": list(labels),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_intent": per_intent,
        "confusion_matrix": generate_confusion_matrix(y_true, y_pred, labels).values.tolist(),
    }


def generate_confusion_matrix(
    y_true: List[str], y_pred: List[str], labels: List[str]
) -> pd.DataFrame:
    """Return a labelled confusion matrix with rows=true and columns=predicted."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def format_confusion_matrix_markdown(cm_df: pd.DataFrame) -> str:
    """Render a confusion matrix as a Markdown table."""
    columns = [str(column) for column in cm_df.columns]
    lines = ["| true \\ predicted | " + " | ".join(columns) + " |"]
    lines.append("|---|" + "|".join("---:" for _ in columns) + "|")
    for index, row in cm_df.iterrows():
        values = " | ".join(str(int(value)) for value in row.tolist())
        lines.append(f"| {index} | {values} |")
    return "\n".join(lines)


def compute_escalation_metrics(
    y_true_escalate: List[bool], y_pred_escalate: List[bool]
) -> Dict[str, float]:
    """Compute escalation precision, recall, F1, and error rates."""
    if len(y_true_escalate) != len(y_pred_escalate):
        raise ValueError("Escalation truth and predictions must have equal lengths")
    true_values = [bool(value) for value in y_true_escalate]
    pred_values = [bool(value) for value in y_pred_escalate]
    tp = sum(actual and predicted for actual, predicted in zip(true_values, pred_values))
    fp = sum(not actual and predicted for actual, predicted in zip(true_values, pred_values))
    fn = sum(actual and not predicted for actual, predicted in zip(true_values, pred_values))
    tn = sum(not actual and not predicted for actual, predicted in zip(true_values, pred_values))
    return {
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        "false_escalation_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "missed_escalation_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "true_positive": float(tp),
        "false_positive": float(fp),
        "true_negative": float(tn),
        "false_negative": float(fn),
    }
