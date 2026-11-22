# Project Implementation Plan

## Explainability of a Pretrained AI-Generated Text Detector Using InterpretML

**Document purpose:** This file is the implementation specification for
a coding agent. Follow it as written. Do not substitute models, change
the project domain, train a new detector, or add unrelated features
without explicit approval.

------------------------------------------------------------------------

## 1. Project objective

Apply **InterpretML** to an existing, pretrained, black-box
machine-learning model that classifies text as human-written or
AI-generated. Evaluate the model on an independent dataset and
demonstrate:

-   Overall predictive performance.
-   Global feature influence.
-   Local explanations for individual predictions.
-   Correct and incorrect decisions.
-   How changes to input features affect predictions.
-   Limitations of the model and its explanations.

The project is an explainability demonstration. It is **not** intended
to establish whether a particular person used AI, and detector
predictions must not be presented as proof of authorship.

## 2. Fixed model selection

**Selected model:** `dinisds/stylometric-ai-detector`\
**Source:** https://huggingface.co/dinisds/stylometric-ai-detector

Use the existing pretrained Random Forest model from this repository.

The model is described as a Random Forest with 100 estimators and eight
stylometric input features. The model card reports 96.03% accuracy on
its own evaluation. Treat that number as the model author's reported
result, not as a result reproduced by this project or a guarantee of
performance on the independent evaluation dataset.

### Mandatory model verification before implementation

The coding agent must inspect the current Hugging Face model card,
repository files, model configuration, and inference instructions before
writing the evaluation pipeline.

Verify and record:

1.  Exact downloadable model artifact and its format.
2.  Required Python/package versions.
3.  Exact feature names, order, units, and preprocessing.
4.  Expected input shape and data types.
5.  Positive-class label and class ordering.
6.  Whether `predict_proba` is available and what each probability
    column represents.
7.  Model licence and any usage restrictions.
8.  Whether the artifact can be loaded safely in the selected
environment.

Do not guess feature order, class mapping, preprocessing, or artifact
filenames. If the repository does not provide enough information to
reproduce inference correctly, stop and report the missing information
rather than silently reverse-engineering or substituting another model.

### Model integrity rules

-   Load the published pretrained artifact.
-   Do not call `fit`, `train`, `partial_fit`, fine-tune, or otherwise
    update the detector.
-   Do not retrain or replace the detector with a newly trained
    classifier.
-   Do not use an EBM or another model as a replacement detector.
-   Keep the detector fixed throughout evaluation and explanation.
-   Record the model repository, artifact filename, revision/commit if
    available, and download date.

## 3. Fixed evaluation dataset

**Dataset:** RAID --- Robust AI Detection Benchmark\
**Hugging Face source:** https://huggingface.co/datasets/liamdugan/raid\
**Research paper:** https://aclanthology.org/2024.acl-long.674/

Use RAID as an independent evaluation source. Do not train or fine-tune
the selected detector on RAID.

### Evaluation sample

Target exactly **10,000 eligible English examples**:

-   5,000 human-written examples.
-   5,000 AI-generated examples.
-   Random sampling with seed `42`.

The dataset's actual schema and available splits must be inspected
first. Do not assume that a split named `train` is appropriate or that
it contains the required fields. Select examples from the available RAID
data while preserving the independent-evaluation principle.

If the exact 5,000/5,000 sample cannot be assembled from the accessible
data, stop and report the actual available counts. Do not silently
change the sample size or class balance.

### Leakage and provenance safeguards

-   Do not use the detector's original training data as the evaluation
    set if that can be identified.
-   Do not fit preprocessing transformations on the evaluation labels.
-   Do not use RAID labels to alter detector parameters or thresholds.
-   Keep source identifiers and available generator/domain metadata for
    analysis.
-   Save the sampled record identifiers and random seed so the
evaluation is reproducible.

## 4. Feature extraction

The selected detector expects eight numerical stylometric features. The
model card/repository is the authoritative source for their definitions
and order.

The previously identified feature concepts are:

1.  Character count.
2.  Word count.
3.  Average word length.
4.  Punctuation count.
5.  Sentence count.
6.  Average sentence length.
7.  Uppercase word count.
8.  Title-case word count.

**Important:** Treat the list above as a checklist to verify, not
permission to invent formulas. Implement the exact feature extraction
and preprocessing specified by the model author. Document any definition
that is not explicitly specified and stop for review if an ambiguity
could change model inputs.

For every record:

-   Preserve the original text and dataset label.
-   Extract the eight features exactly as required by the model.
-   Handle empty text, missing values, malformed records, and
    zero-sentence cases explicitly.
-   Record excluded records and exclusion reasons.
-   Ensure feature order and data types match the model's expected
    input.
-   Do not add extra features to the detector input.

## 5. InterpretML compatibility gate

Before running the full experiment, prove that the **original pretrained
detector** can be passed to InterpretML's black-box explainers through
its actual prediction interface.

The compatibility test must establish:

-   The model accepts the exact eight-column feature matrix.
-   Its prediction function returns the expected labels.
-   Its probability function returns valid probabilities, if supported.
-   InterpretML can call the original model or a thin prediction adapter
    without changing its predictions.
-   A small local explanation can be generated for a small number of
    records.

A thin adapter is allowed only to adapt input/output format. It must
delegate predictions to the original frozen model and must not introduce
another trained predictor.

If InterpretML cannot explain the original model through a compatible
interface, stop and report the technical blocker. Do not quietly train a
surrogate and claim that it explains the original model.

## 6. Evaluation metrics

Run the frozen detector on the 10,000-record evaluation sample and
report:

-   Accuracy.
-   Precision.
-   Recall.
-   F1-score.
-   ROC-AUC, only if reliable probability scores are available.
-   Confusion matrix.
-   Per-class precision, recall, and F1.
-   Number of records excluded during preprocessing.

Clearly state the positive class and class ordering used for all
metrics.

Do not present the model card's reported 96.03% accuracy as this
project's measured result. Keep the two results separate.

## 7. InterpretML explainability

Use InterpretML's black-box explainers on the **original pretrained
detector**.

### Global explanations

1.  **Morris Sensitivity Analysis** --- examine which input features
    have the greatest effect on model outputs.
2.  **Partial Dependence** --- examine how predicted output changes as
    one feature varies, while accounting for the other input features
    according to the explainer's method.

### Local explanations

1.  **LIME Tabular** --- explain individual predictions locally.
2.  **SHAP Kernel** --- estimate feature contributions for individual
    predictions.

Use the same frozen model and same feature definitions for all
explainers.

### Explanation examples

Select four records from the evaluation set:

1.  Correctly classified AI-generated text.
2.  Correctly classified human-written text.
3.  Human-written text incorrectly classified as AI-generated.
4.  AI-generated text incorrectly classified as human-written.

If a category has no examples, report that fact and do not fabricate an
example.

For each selected record, show:

-   Ground-truth label.
-   Predicted label.
-   Predicted class probabilities, if available.
-   Eight input feature values.
-   Local explanation output.
-   A plain-language interpretation that does not claim causality.

### Explanation correctness

-   State that LIME and Kernel SHAP are approximations.
-   Do not describe feature contributions as causal effects.
-   Do not claim that the highlighted words or text spans were
    identified by InterpretML; this project explains the detector's
    numerical stylometric inputs.
-   Do not confuse global feature sensitivity with feature importance in
    every possible dataset.
-   Report explainer settings and random seeds.
-   Check that explanations are stable enough to interpret; flag
    unstable results.

## 8. Feature perturbation / what-if analysis

Demonstrate how the frozen detector's output changes when one or more
stylometric inputs are changed.

Rules:

-   Start from real evaluation records.
-   Change one feature at a time for the primary analysis.
-   Keep all other feature values fixed.
-   Use plausible values within the observed evaluation-data range.
-   Show original and modified feature values, predictions, and
    probabilities.
-   Do not modify the original text and then pretend the altered feature
    vector represents a valid new text.
-   Label this as a feature-space sensitivity experiment, not a causal
    intervention or guaranteed editing recommendation.

## 9. Project structure

Create the following repository structure:

``` text
interpretml-ai-text-detector/
├── README.md
├── PLAN.md
├── requirements.txt
├── configs/
│   └── experiment.yaml
├── data/
│   ├── README.md
│   └── sampled_record_ids.csv
├── models/
│   └── README.md
├── notebooks/
│   └── interpretml_ai_text_detector.ipynb
├── src/
│   ├── model_loader.py
│   ├── dataset_loader.py
│   ├── feature_extraction.py
│   ├── evaluation.py
│   ├── explainability.py
│   └── what_if_analysis.py
└── outputs/
    ├── metrics/
    ├── figures/
    ├── explanations/
    └── reports/
```

Do not commit large model files or the full dataset to Git. Document how
to download them reproducibly. Keep any required credentials out of
source control.

## 10. Runtime and resource constraints

-   Primary environment: Google Colab.
-   Language: Python.
-   Use CPU execution first; this is a small tabular Random Forest
    inference workload.
-   Do not provision AWS, GCP, paid APIs, a domain, or a hosted
    application.
-   Pin tested package versions in `requirements.txt`.
-   Record Python and package versions in the experiment report.
-   If the Hugging Face artifact requires a package/runtime incompatible
    with Colab, stop and report the issue before changing the model.

## 11. Final deliverables

The coding agent must produce:

1.  A reproducible Jupyter notebook containing the complete experiment.
2.  A verified model card/inference summary, including exact artifact
    and feature contract.
3.  A reproducible RAID sample manifest with record IDs and seed.
4.  A model evaluation report with metrics and confusion matrix.
5.  Global explanation visualizations.
6.  Four local explanation case studies, where available.
7.  Feature perturbation / what-if results.
8.  A limitations section.
9.  A concise final report suitable for a technical review meeting.
10. A README with exact setup and execution instructions.

All reported numbers and charts must be generated by the notebook from
the selected model and sampled dataset. Do not fabricate results or use
screenshots from unrelated tools.

## 12. Acceptance criteria

The project is complete only when all of the following are true:

-   [ ] The exact pretrained Hugging Face artifact is identified and
    loaded.
-   [ ] The model remains frozen; no detector training or fine-tuning
    occurs.
-   [ ] The eight-feature input contract is verified against the model
    repository.
-   [ ] The RAID sample is reproducible and has the required class
    counts, or the shortfall is explicitly documented.
-   [ ] The original detector's predictions are reproducible.
-   [ ] The original detector is successfully callable by InterpretML
    explainers.
-   [ ] Global explanations are generated.
-   [ ] Local explanations are generated for available correct and
    incorrect cases.
-   [ ] Feature perturbation results are generated and correctly
    qualified.
-   [ ] Metrics, figures, and conclusions are traceable to notebook
    outputs.
-   [ ] Limitations and non-causal nature of explanations are clearly
    stated.
-   [ ] No substitute model or surrogate detector is presented as the
    selected original model.

## 13. Mandatory stop conditions

The coding agent must stop and report, rather than make an unapproved
substitution, if:

-   The Hugging Face repository no longer contains the described
    pretrained artifact.
-   The model's required feature definitions or order cannot be
    verified.
-   The model cannot be loaded safely or its licence prevents the
    intended use.
-   The RAID data cannot supply the specified independent sample.
-   The model's prediction interface is incompatible with InterpretML.
-   An explanation can only be produced by training a different model.

**No model substitutions, no detector retraining, no fine-tuning, and no
surrogate-model substitution without explicit approval.**

------------------------------------------------------------------------

## References

-   Selected model:
    https://huggingface.co/dinisds/stylometric-ai-detector
-   RAID dataset: https://huggingface.co/datasets/liamdugan/raid
-   RAID paper: https://aclanthology.org/2024.acl-long.674/
-   InterpretML documentation: https://interpret.ml/docs/
-   InterpretML black-box explainers:
    https://interpret.ml/docs/blackbox.html
-   InterpretML API:
    https://interpret.ml/docs/python/api/ExplainableMixin.html
