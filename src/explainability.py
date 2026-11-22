"""InterpretML black-box explanations for the original frozen detector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from interpret.blackbox import LimeTabular, MorrisSensitivity, PartialDependence, ShapKernel
from interpret.blackbox._sensitivity import MorrisSampler

from .feature_extraction import FEATURES


class ProbabilityAdapter:
    """Input-format adapter that delegates every output to the original model."""

    def __init__(self, model):
        self.model = model
        self.classes_ = model.classes_

    def predict_proba(self, X):
        array = np.asarray(X, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[1] != len(FEATURES):
            raise ValueError(f"Expected {len(FEATURES)} features, got {array.shape[1]}")
        return self.model.predict_proba(array)

    def predict(self, X):
        array = np.asarray(X, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[1] != len(FEATURES):
            raise ValueError(f"Expected {len(FEATURES)} features, got {array.shape[1]}")
        return self.model.predict(array)


def compatibility_gate(model, X: pd.DataFrame, *, n_records: int = 12, seed: int = 42) -> dict:
    """Prove original inference parity and a small InterpretML local explanation."""
    if X.shape[1] != len(FEATURES) or list(X.columns) != FEATURES:
        raise ValueError("Compatibility matrix must have the exact eight columns in contract order.")
    sample = X.sample(min(n_records, len(X)), random_state=seed).astype(float)
    adapter = ProbabilityAdapter(model)
    direct_proba = model.predict_proba(sample.to_numpy())
    adapted_proba = adapter.predict_proba(sample)
    direct_labels = model.predict(sample.to_numpy())
    if not np.array_equal(direct_labels, adapter.predict(sample)):
        raise AssertionError("Adapter changed predictions.")
    if not np.allclose(direct_proba, adapted_proba, rtol=0, atol=0):
        raise AssertionError("Adapter changed class probabilities.")

    lime = LimeTabular(adapter.predict_proba, sample, feature_names=FEATURES, random_state=seed)
    lime_result = lime.explain_local(sample.iloc[:1], name="Compatibility gate LIME")
    shap = ShapKernel(adapter.predict_proba, sample.iloc[: min(5, len(sample))], feature_names=FEATURES)
    shap_result = shap.explain_local(sample.iloc[:1], name="Compatibility gate Kernel SHAP", nsamples=64)
    return {
        "records_checked": int(len(sample)),
        "predict_shape": list(direct_labels.shape),
        "predict_proba_shape": list(direct_proba.shape),
        "class_order": [int(v) for v in model.classes_],
        "adapter_exact_probability_parity": True,
        "lime_local_generated": lime_result is not None,
        "shap_kernel_local_generated": shap_result is not None,
        "adapter": adapter,
    }


def build_explainers(
    model,
    X: pd.DataFrame,
    *,
    seed: int = 42,
    shap_background: int = 50,
    morris_trajectories: int = 200,
    morris_num_levels: int = 4,
    pdp_num_points: int = 10,
):
    adapter = ProbabilityAdapter(model)
    background = X.sample(min(shap_background, len(X)), random_state=seed).astype(float)
    return {
        "adapter": adapter,
        "morris": MorrisSensitivity(
            adapter.predict_proba,
            X,
            feature_names=FEATURES,
            feature_types=["continuous"] * len(FEATURES),
            sampler=MorrisSampler(N=morris_trajectories, num_levels=morris_num_levels, seed=seed),
        ),
        "pdp": PartialDependence(
            adapter.predict_proba,
            X,
            feature_names=FEATURES,
            feature_types=["continuous"] * len(FEATURES),
            num_points=pdp_num_points,
        ),
        "lime": LimeTabular(adapter.predict_proba, background, feature_names=FEATURES, random_state=seed),
        "shap": ShapKernel(adapter.predict_proba, background, feature_names=FEATURES),
    }
