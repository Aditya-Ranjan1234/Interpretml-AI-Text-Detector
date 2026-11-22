# Evaluation data

No RAID text is stored in this repository. The notebook streams the pinned RAID `train` revision, selects 5,000 eligible human and 5,000 eligible AI records with seed 42, and writes their IDs, dataset revision, seed, and source/generator/domain metadata to `sampled_record_ids.csv`. It writes filtered record IDs and exclusion reasons to `excluded_record_ids.csv` and exclusion counts to the experiment report. The selected data are used only for evaluation and explanation; they are never used to fit or tune the detector.

The published RAID training split is multi-gigabyte. Expect the initial streaming scan to transfer substantial data. To reproduce the sample, keep the source revision, filters, seed, and Python dependencies fixed.
