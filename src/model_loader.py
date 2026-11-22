"""Fetch, integrity-check, and validate the fixed published model."""

from __future__ import annotations

import hashlib
import os
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn
from huggingface_hub import hf_hub_download
from sklearn.ensemble import RandomForestClassifier

from .feature_extraction import FEATURES

MODEL_REPO = "dinisds/stylometric-ai-detector"
MODEL_REVISION = "de4e8a8099533b6af8a46f004d5966a7594b14a6"
MODEL_FILENAME = "random_forest_stylometric_model.joblib"
MODEL_SHA256 = "386cfbd171eeae2f532f86f34b170644f32d59cdd9daa18b7144179a5347b57c"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_model(*, cache_dir: str | Path = "models/cache") -> tuple[RandomForestClassifier, dict]:
    """Download exact pinned bytes then load only after explicit pickle opt-in."""
    if os.environ.get("ALLOW_PICKLE_MODEL") != "1":
        raise RuntimeError(
            "The published model is Joblib/pickle. Review models/README.md and set "
            "ALLOW_PICKLE_MODEL=1 to trust and load this pinned artifact."
        )

    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        revision=MODEL_REVISION,
        cache_dir=str(cache_dir),
    )
    actual_digest = sha256_file(path)
    if actual_digest != MODEL_SHA256:
        raise ValueError(f"Model SHA-256 mismatch: expected {MODEL_SHA256}, got {actual_digest}")

    model = joblib.load(path)
    if not isinstance(model, RandomForestClassifier):
        raise TypeError(f"Expected the published RandomForestClassifier, got {type(model)!r}")
    if int(model.n_features_in_) != len(FEATURES):
        raise ValueError(f"Expected {len(FEATURES)} features, artifact expects {model.n_features_in_}")
    if list(model.classes_) != [0, 1]:
        raise ValueError(f"Expected classes [0, 1] (Human, AI), got {list(model.classes_)}")
    if len(model.estimators_) != 100:
        raise ValueError(f"Expected 100 frozen estimators, got {len(model.estimators_)}")

    probe = np.asarray([[10, 2, 4.0, 1, 1, 2.0, 0, 1]], dtype=np.float64)
    predicted = model.predict(probe)
    probabilities = model.predict_proba(probe)
    if predicted.shape != (1,) or predicted[0] not in (0, 1):
        raise ValueError(f"Unexpected prediction contract: {predicted!r}")
    if probabilities.shape != (1, 2) or not np.isfinite(probabilities).all():
        raise ValueError(f"Unexpected probability contract: {probabilities!r}")
    if (probabilities < 0).any() or (probabilities > 1).any() or not np.isclose(probabilities.sum(), 1):
        raise ValueError(f"Invalid class probabilities: {probabilities!r}")

    metadata = {
        "repo_id": MODEL_REPO,
        "revision": MODEL_REVISION,
        "filename": MODEL_FILENAME,
        "sha256": actual_digest,
        "downloaded_file": str(path),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "classes": [int(value) for value in model.classes_],
        "class_labels": {"0": "Human", "1": "AI"},
        "n_features_in": int(model.n_features_in_),
        "n_estimators": len(model.estimators_),
    }
    return model, metadata
