# Explainable AI Text Detector

Reproducible evaluation and explanation of the frozen `dinisds/stylometric-ai-detector` Random Forest on an independently sampled, balanced RAID dataset. This is a research demonstration. A detector result is not proof that a person used AI and must not be used to make authorship claims about individuals.

## Status

The runnable pipeline has completed end to end. Its measured outputs are committed below; the 419 MB model and full multi-gigabyte RAID dataset are downloaded at runtime and are not committed. The selected 10,000-record text cache is local and ignored by Git; only its IDs and provenance are published.

On the pinned balanced RAID sample, measured accuracy was **60.99%**, AI precision **59.66%**, AI recall **67.90%**, AI F1 **63.51%**, and AI ROC-AUC **0.645**. The model card's 96.03% is a separate author-reported evaluation and was not reproduced here. See the [final report](outputs/reports/final_report.md) and [HTML build log](outputs/reports/build_log.html) for details.

The fixed model artifact, feature formulas, model labels, license, dataset splits/schema, and source revisions are recorded in [models/README.md](models/README.md). The author-released feature extractor is followed verbatim. The artifact is Joblib/pickle, so only load the pinned artifact after reviewing its source and checksum. `ALLOW_PICKLE_MODEL=1` is required for the loader to unpickle it.

## Setup

The primary environment is Google Colab with CPU execution. Put this repository in Google Drive at `MyDrive/Ethical-AI`, then run these cells in Colab:

```ipython
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Ethical-AI
!pip install -r requirements.txt
```

Open `notebooks/interpretml_ai_text_detector.ipynb`. Before running the model-load cell, set `os.environ['ALLOW_PICKLE_MODEL'] = '1'` in a cell after reviewing the pinned artifact and checksum in `models/README.md`. The full RAID training split is streamed and scanned to obtain a reservoir sample, so the first run can take time and transfer substantial data. No AWS, GCP, GPU, credentials, or paid APIs are needed.

For a local environment, use Python 3.11 or 3.12, create a virtual environment, install `requirements.txt`, then launch Jupyter. The first dataset scan is long; a temporary AWS CPU instance was used for the completed run, but no cloud services are required by the project. The versions in `requirements.txt` are a reproducible working stack, not a claim about the undisclosed environment used to serialize the published model. The loader checks the estimator contract after deserialization and stops on incompatibility.

## Data selection

The notebook pins RAID to its published dataset revision and streams the labeled `raid` / `train` split. The RAID card identifies this split as English genres; the `extra` split includes Czech, German, and code, and `test` is unlabeled. Records with an English genre, a nonempty `generation` text, and `model == human` or an AI model name are eligible. Human and AI rows are independently reservoir-sampled to exactly 5,000 each with seed 42. The notebook stops with observed eligible counts if either class is short. Sampling IDs and available provenance metadata are written to `data/sampled_record_ids.csv`.

This is an independent source from the model card's stated Kaggle training dataset. RAID source overlap cannot be fully ruled out from the published identifiers; the notebook preserves `source_id`, `adv_source_id`, domain, generator, attack, decoding, and repetition metadata to support audit. RAID labels do not change model parameters or thresholds.

## Outputs

Run the notebook to populate:

- `outputs/metrics/`: JSON metrics, per-class scores, and confusion matrix.
- `outputs/figures/`: confusion matrix, Morris sensitivity, and partial dependence figures.
- `outputs/explanations/`: local LIME and Kernel SHAP cases and what-if results.
- `outputs/reports/`: final Markdown and JSON reports, plus the standalone HTML build log.

The notebook performs an InterpretML compatibility gate on a small sample before the full run. It sends the original model's `predict_proba` interface (or a format-only adapter that delegates to it) to InterpretML. It never fits a replacement detector. LIME and Kernel SHAP are approximations, global sensitivity is dataset-dependent, and feature perturbations are feature-space experiments rather than causal interventions.

## Project files

- [plan.md](plan.md): the supplied implementation specification (retained under its original filename; this workspace treats case-only filenames as identical).
- [Notebook](notebooks/interpretml_ai_text_detector.ipynb): full experiment.
- [Configuration](configs/experiment.yaml): fixed IDs, revisions, seed, sample target, and explainer settings.
- `src/`: reusable loading, feature extraction, evaluation, explanation, and what-if functions.

## Limitations

This detector uses only eight surface-level stylometric statistics and was trained on a different source dataset. Performance can vary by topic, genre, generator, attack, and text length. The model card's 96.03% is the author's reported result on the author's evaluation and is never reported as this project's result. InterpretML's black-box methods approximate the frozen model's behavior; explanations do not identify text spans, prove causality, establish authorship, or validate an individual accusation.
