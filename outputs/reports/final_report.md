# RAID Detector Evaluation Report

Generated: 2026-11-22T15:34:05.258916+00:00

## Frozen model

- Repository/revision: `dinisds/stylometric-ai-detector` at `de4e8a8099533b6af8a46f004d5966a7594b14a6`
- Artifact: `random_forest_stylometric_model.joblib` (SHA-256 `386cfbd171eeae2f532f86f34b170644f32d59cdd9daa18b7144179a5347b57c`)
- Class mapping: 0 = Human, 1 = AI (positive class)

## Evaluation

- Sample: 10,000 records (5,000 Human, 5,000 AI), RAID seed 42
- Accuracy: 0.6099
- Precision / recall / F1 for AI: 0.5966 / 0.6790 / 0.6351
- ROC-AUC for AI: 0.6445
- Confusion matrix (Human, AI): `[[2704, 2296], [1605, 3395]]`
- Exclusion counts: `{}`

The model card's 96.03% accuracy is the author's reported result on a different evaluation and is not reproduced here.

## Interpretability and limits

Morris sensitivity and partial dependence summarize the sampled feature space. LIME and Kernel SHAP approximate local behavior for the four available outcome categories; their contributions are not causal and explain numeric stylometric inputs, not text spans. What-if results modify one numerical input while holding the other seven fixed; changed vectors may not represent valid text. RAID provides an independent source from the model card's stated training set, though exact source overlap cannot be fully ruled out. A detector prediction is not proof of authorship.

See `outputs/metrics/evaluation_metrics.json`, `outputs/explanations/`, and `outputs/figures/` for generated details.
