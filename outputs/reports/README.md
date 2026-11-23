# Project reports

The report artifacts are committed to the repository in these formats:

- [`interpretml_report.html`](../../interpretml_report.html) — comprehensive standalone report with embedded charts and what-if tables; works offline.
- [`build_log.html`](build_log.html) — standalone timeline of the implementation and execution.
- [`final_report.md`](final_report.md) — concise measured evaluation and limitations.
- [`experiment_report.json`](experiment_report.json) — machine-readable run metadata, configuration, versions, metrics, and limitations.

Supporting generated results are in [`../metrics/`](../metrics/), [`../explanations/`](../explanations/), and [`../figures/`](../figures/). The executed end-to-end experiment is in [`../../notebooks/interpretml_ai_text_detector.ipynb`](../../notebooks/interpretml_ai_text_detector.ipynb). The raw selected texts remain in the ignored local cache and are not in Git; the sampled ID/provenance manifest is in [`../../data/sampled_record_ids.csv`](../../data/sampled_record_ids.csv).
