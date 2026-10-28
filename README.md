# Transcriptomic signatures of pre-cachectic skeletal muscle

Computational analysis for Aim 3 of a BIOM30003 research placement
(Laboratory for Muscle Research and Therapeutics, Centre for Muscle Research,
University of Melbourne). The project identifies conserved transcriptomic
signatures of pre-cachectic skeletal muscle in the C26 mouse model of cancer
cachexia, analysed separately by sex.

Two independent feature-selection methods are run on the lab's own n=8
(4 SHAM, 4 C26, per sex) RNA-seq dataset. These are attribute weighting
(ReliefF) and LASSO/Elastic Net, each under leave-one-out cross-validation.
The resulting shortlists are compared against one another, and a curated
Elastic Net gene set is validated against an independent public cohort
(GSE276018).

## Repo structure

Run in this order.

1. **`Data_PreProcessing.ipynb`** loads the raw expression matrix, filters
   genes by differential expression (FDR < 0.05) and a minimum expression
   threshold, splits by sex, and writes the per-sex expression tables and
   filtered gene lists used by every downstream notebook.
2. **`Feature_Weighting.ipynb`** runs attribute weighting (ReliefF,
   decision-tree importance) with leave-one-pair-out stability selection
   (`attribute_weighting.py`).
3. **`Lasso_Feature_Selection.ipynb`** runs LOOCV LASSO and Elastic Net
   feature selection, an exact permutation test of LOOCV accuracy, and
   external validation of a curated Elastic Net gene set against GSE276018
   (`lasso_selection.py`).

Supporting modules

- `lasso_selection.py` and `attribute_weighting.py` hold the reusable
  pipeline logic each notebook imports.
- `synthetic_data.py` generates a synthetic expression matrix matching the
  real data's structure (column names, dtypes, label categories, value
  ranges), used for developing and testing pipeline code without real
  sample data.
- `compute_aggregate_stats.py` is a local-only utility that reproduces the
  `Data_PreProcessing.ipynb` filter against the real data and prints
  aggregate scalars (group means and SDs, not per-gene or per-sample
  values) used to parameterise `synthetic_data.py`.

`figures/` holds saved presentation figures (derived, categorical outputs,
not raw data) and is tracked.

## Reproducing

```
pip install -r requirements.txt
```

This is pinned to the versions this analysis was run with. LASSO/Elastic
Net LOOCV accuracy against the external cohort was observed to be sensitive
to solver and floating-point differences across environments, so matching
these versions matters for reproducing exact numbers. Gene selection and
direction of effect are stable across environments, but raw transfer
accuracy is not.

Real data is required locally and is not included in this repo (see below).
With it in place, run the notebooks in the order listed above. Each
notebook has a `USE_REAL_DATA` real-data switch. Set it only for your own
local, offline run. Everything here defaults to synthetic data for code
development.

## Data availability

- The lab's own RNA-seq data (`Muscle_Male_Female_All.xlsx`) is not
  included. It is lab-held data, and access requires lab authorisation.
- External validation uses GSE276018 (GEO), downloaded separately and kept
  local only, not redistributed here.
- `restructured/` (per-sex expression tables and gene lists derived from
  the raw data) is generated locally by `Data_PreProcessing.ipynb` and is
  not committed.
