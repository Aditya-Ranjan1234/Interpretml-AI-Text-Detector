"""Metrics and reporting helpers for the frozen detector."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score


def evaluate_model(model, X, y: Sequence[int]) -> dict:
    y_true = np.asarray(y, dtype=int)
    y_pred = np.asarray(model.predict(X), dtype=int)
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    class_values = list(model.classes_)
    if 1 not in class_values:
        raise ValueError("Positive class 1 (AI) is missing from model.classes_.")
    ai_col = class_values.index(1)
    metrics = {
        "positive_class": 1,
        "positive_label": "AI",
        "class_order": [int(value) for value in class_values],
        "class_names": {"0": "Human", "1": "AI"},
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_ai": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_ai": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_ai": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc_ai": float(roc_auc_score(y_true, probabilities[:, ai_col])),
        "confusion_matrix_labels_human_ai": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=["Human", "AI"],
            output_dict=True, zero_division=0,
        ),
        "n_records": int(len(y_true)),
    }
    return metrics
