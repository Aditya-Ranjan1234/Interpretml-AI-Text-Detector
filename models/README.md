# Frozen detector: verified source contract

**Repository:** [`dinisds/stylometric-ai-detector`](https://huggingface.co/dinisds/stylometric-ai-detector)  
**Pinned model revision:** `de4e8a8099533b6af8a46f004d5966a7594b14a6` (checked 2026-11-22)  
**Artifact:** `random_forest_stylometric_model.joblib`  
**Artifact size:** 418,937,049 bytes  
**SHA-256:** `386cfbd171eeae2f532f86f34b170644f32d59cdd9daa18b7144179a5347b57c`  
**License:** MIT  
**Model card:** Random Forest, 100 estimators, `random_state=42`; the card's 96.03% is its own reported evaluation result, not this project's measured result.

The Hugging Face card says to load this artifact with `joblib.load`, gives the eight feature names/order, and defines `0 = Human`, `1 = AI`. The maintainer's linked `stylometric-ai-detector` package (source release 0.2.4) specifies the extraction formulas reproduced in `src/feature_extraction.py`. Its package metadata requires Python >=3.9 and scikit-learn >=1.0, but neither the model card nor repository states the exact scikit-learn version used to serialize the artifact. Therefore `requirements.txt` pins a tested-in-notebook working environment, and the loader verifies class order, feature count, estimator type, and probabilities at runtime; it does not claim those requirements were the training versions.

Feature contract and units, in exact order:

1. `char_count`: Python `len(text)` (Unicode code points).
2. `word_count`: count of items from `text.split()`.
3. `avg_word_len`: sum of those whitespace-delimited item lengths divided by count.
4. `punct_count`: characters contained in Python `string.punctuation` (ASCII punctuation).
5. `sentence_count`: nonempty pieces after splitting on one or more `.`, `!`, or `?`; the published implementation changes zero sentences to one for division.
6. `avg_sentence_len`: `word_count / sentence_count`.
7. `upper_case_count`: whitespace-delimited words for which `.isupper()` and `.isalpha()` are true.
8. `title_case_count`: whitespace-delimited words for which `.istitle()` is true.

Counts and averages are passed as numeric values in one row of shape `(1, 8)`; `predict_proba` is expected to have columns in the estimator's `classes_` order. The repository/package describes the confidence for AI as class 1, and the notebook explicitly maps probability columns from `classes_` rather than assuming a column index.

## Serialization safety

Joblib uses Python pickle semantics. Hugging Face marks the file as a Joblib model and its scanner reports NumPy and scikit-learn imports, including `RandomForestClassifier` and decision trees. A checksum protects integrity; it does not neutralize pickle execution. The loader refuses to deserialize unless `ALLOW_PICKLE_MODEL=1` is explicitly set. Set that only after reviewing the fixed public repository/revision and artifact digest. The loader downloads the model from that immutable revision and checks SHA-256 before deserialization. Never point this code at an untrusted artifact.

The model does not expose a separately versioned inference API. The supplied notebook's small compatibility gate checks actual prediction labels, probability values, InterpretML calls, and local explanations before processing the full evaluation set.
