"""One-feature-at-a-time feature-space sensitivity experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_extraction import FEATURES


def one_feature_what_if(model, X: pd.DataFrame, records: list[dict], *, quantiles=(0.1, 0.5, 0.9)) -> pd.DataFrame:
    """Perturb a real record to observed-range quantiles for each feature."""
    if list(X.columns) != FEATURES:
        raise ValueError("What-if inputs must use exact detector feature order.")
    rows = []
    count_features = {
        "char_count", "word_count", "punct_count", "sentence_count",
        "upper_case_count", "title_case_count",
    }
    observed = {}
    for feature in FEATURES:
        method = "nearest" if feature in count_features else "linear"
        observed[feature] = np.quantile(
            X[feature].to_numpy(dtype=float), quantiles, method=method
        )
    for record in records:
        original = np.asarray([[record[name] for name in FEATURES]], dtype=float)
        original_prediction = int(model.predict(original)[0])
        original_probabilities = model.predict_proba(original)[0]
        for feature_index, feature in enumerate(FEATURES):
            for quantile, modified_value in zip(quantiles, observed[feature]):
                modified = original.copy()
                modified[0, feature_index] = modified_value
                predicted = int(model.predict(modified)[0])
                probabilities = model.predict_proba(modified)[0]
                rows.append({
                    "record_id": record.get("id"),
                    "feature": feature,
                    "quantile": float(quantile),
                    "original_value": float(original[0, feature_index]),
                    "modified_value": float(modified_value),
                    "original_predicted_label": original_prediction,
                    "modified_predicted_label": predicted,
                    "original_p_human": float(original_probabilities[list(model.classes_).index(0)]),
                    "original_p_ai": float(original_probabilities[list(model.classes_).index(1)]),
                    "modified_p_human": float(probabilities[list(model.classes_).index(0)]),
                    "modified_p_ai": float(probabilities[list(model.classes_).index(1)]),
                })
    return pd.DataFrame(rows)
