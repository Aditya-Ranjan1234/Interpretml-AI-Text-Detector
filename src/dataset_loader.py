"""Pinned streaming RAID loader and reproducible class-balanced reservoir sampler."""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm.auto import tqdm

from .feature_extraction import FEATURES, extract_stylometric_features

RAID_REPO = "liamdugan/raid"
RAID_REVISION = "865cac74188466cb0c3b7574a10204007b57a459"
ENGLISH_DOMAINS = {"abstracts", "books", "news", "poetry", "recipes", "reddit", "reviews", "wiki"}
PROVENANCE_FIELDS = (
    "id", "source_id", "adv_source_id", "model", "domain", "attack",
    "decoding", "repetition_penalty", "title",
)


def _label(row: dict[str, Any]) -> int | None:
    model = row.get("model")
    if not isinstance(model, str):
        return None
    return 0 if model.lower() == "human" else 1


def _public_record(row: dict[str, Any], label: int, features: dict) -> dict[str, Any]:
    record = {key: row.get(key) for key in PROVENANCE_FIELDS}
    record.update({"text": row["generation"], "y_true": label})
    record.update(features)
    return record


def sample_raid(
    *,
    per_class: int = 5000,
    seed: int = 42,
    config: str = "raid",
    split: str = "train",
    exclusions_path: str | Path | None = "data/excluded_record_ids.csv",
) -> tuple[list[dict], dict]:
    """Scan the pinned labeled English split and reservoir-sample both classes.

    Uses independent deterministic Algorithm R reservoirs. Counts malformed or
    filtered records by reason and fails if the requested balanced sample cannot
    be assembled; it never silently reduces the target.
    """
    dataset = load_dataset(
        RAID_REPO, name=config, split=split, streaming=True, revision=RAID_REVISION
    )
    reservoirs: dict[int, list[dict]] = {0: [], 1: []}
    seen = Counter()
    excluded = Counter()
    rngs = {0: random.Random(seed), 1: random.Random(seed + 1)}
    if exclusions_path is not None:
        exclusions_path = Path(exclusions_path)
        exclusions_path.parent.mkdir(parents=True, exist_ok=True)
        exclusions_file = exclusions_path.open("w", newline="", encoding="utf-8")
        exclusions_writer = csv.DictWriter(
            exclusions_file, fieldnames=["row_index", "id", "source_id", "reason"]
        )
        exclusions_writer.writeheader()
    else:
        exclusions_file = None
        exclusions_writer = None

    def record_exclusion(row: dict[str, Any], row_index: int, reason: str) -> None:
        excluded[reason] += 1
        if exclusions_writer is not None:
            exclusions_writer.writerow({
                "row_index": row_index,
                "id": row.get("id"),
                "source_id": row.get("source_id"),
                "reason": reason,
            })

    try:
        for row_index, row in enumerate(tqdm(dataset, desc="Scanning RAID", unit="records")):
            seen["records_scanned"] += 1
            if row.get("domain") not in ENGLISH_DOMAINS:
                record_exclusion(row, row_index, "non_english_or_unknown_domain")
                continue
            label = _label(row)
            if label is None:
                record_exclusion(row, row_index, "missing_or_invalid_model_label")
                continue
            text = row.get("generation")
            if not isinstance(text, str):
                record_exclusion(row, row_index, "missing_or_non_string_text")
                continue
            if not text.strip():
                record_exclusion(row, row_index, "empty_text")
                continue

            features = extract_stylometric_features(text)
            if len(features) != len(FEATURES):
                record_exclusion(row, row_index, "feature_extraction_contract_error")
                continue
            seen[f"eligible_class_{label}"] += 1
            bucket = reservoirs[label]
            ordinal = seen[f"eligible_seen_class_{label}"] + 1
            seen[f"eligible_seen_class_{label}"] = ordinal
            candidate = _public_record(row, label, features)
            if len(bucket) < per_class:
                bucket.append(candidate)
            else:
                replace_at = rngs[label].randrange(ordinal)
                if replace_at < per_class:
                    bucket[replace_at] = candidate
    finally:
        if exclusions_file is not None:
            exclusions_file.close()

    counts = {"human": seen["eligible_class_0"], "ai": seen["eligible_class_1"]}
    if any(len(reservoirs[label]) != per_class for label in (0, 1)):
        raise ValueError(
            f"Could not assemble {per_class:,} per class from pinned RAID {config}/{split}. "
            f"Eligible counts: {counts}. Exclusions: {dict(excluded)}"
        )

    records = reservoirs[0] + reservoirs[1]
    # Stable row order independent of reservoir's internal replacement order.
    records.sort(key=lambda row: (row.get("id") or "", row["y_true"]))
    metadata = {
        "dataset_repo": RAID_REPO,
        "dataset_revision": RAID_REVISION,
        "config": config,
        "split": split,
        "seed": seed,
        "sample_per_class": per_class,
        "eligible_counts_seen": counts,
        "sample_counts": {"human": per_class, "ai": per_class},
        "records_scanned": seen["records_scanned"],
        "excluded_counts": dict(excluded),
        "excluded_records_manifest": str(exclusions_path) if exclusions_path is not None else None,
        "source_fields_retained": list(PROVENANCE_FIELDS),
    }
    return records, metadata
