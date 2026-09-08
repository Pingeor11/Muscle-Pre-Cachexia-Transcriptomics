# Attribute-Weighting Stage — Adapting Zhao et al. 2024 §2.3

## Context

The project's original ML workflow roadmap (in `Data_PreProcessing.ipynb`'s final
markdown cell) sketched a generic feature-selection → classifier → validation
pipeline aimed at "Fst-treated vs. vehicle" labels. That label doesn't exist in
the current data. What *is* on disk (`restructured/{female,male}_expr.csv`) is
already structured exactly like the target problem in Zhao et al. 2024
("Transcriptomic signature of cancer cachexia by integration of machine
learning, literature mining and meta-analysis," *Comput Biol Med* 2024): gene
expression samples labeled cachexia (C26) vs. control (SHAM). We're replacing
the roadmap's feature-selection step with a direct adaptation of that paper's
**attribute weighting** stage (§2.3): an ensemble of 7 feature-importance
models, each normalized to [0,1] per gene, summed (max 7), with genes scoring
sum > 5 selected as the "signature."

Key differences from the paper's setting that shape this plan:
- **n=8 per sex** (4 SHAM + 4 C26), vs. the paper's 88 samples across 10
  studies — first-draft rankings will be much higher-variance; no CV/LASSO is
  added yet (explicitly deferred per your direction), but this instability is
  flagged everywhere it matters rather than hidden.
- **Sex is not pooled.** Data is already split into separate male/female
  files; per CLAUDE.md and the lab's own sex-divergence findings, we run the
  full pipeline independently per sex — two separate signature gene lists,
  not one combined analysis with sex as a covariate (unlike the paper, which
  pooled all studies/sexes and used sex as one more categorical feature).
- **No synthetic dataset exists yet.** Per CLAUDE.md, no pipeline code may be
  developed/tested against real expression values in this session. A
  synthetic stand-in must be built first, parameterized only by **aggregate
  summary statistics you compute locally and paste back** — never per-gene or
  per-sample values.
- **RapidMiner internals aren't published in the paper.** Where the paper is
  silent on implementation details (binning for entropy-based models, Gini
  scoring approach), we use the most methodologically faithful interpretation
  available (see "Methodology choices" below), not arbitrary defaults.

**Scope of this draft:** attribute weighting only, ending at signature gene
selection. The paper's downstream pattern-discovery/classifier stage (§2.4,
11 ML models with 5-fold CV) is explicitly out of scope — noted as a future
step, not implemented now.

---

## File layout

Flat layout, consistent with the existing repo (no `src/` yet):

```
Transcriptomics_Coding/
├── Data_PreProcessing.ipynb          # existing — small addition: save filtered gene symbol lists (see below)
├── compute_aggregate_stats.py        # NEW — run LOCALLY ONLY by you, never in this session
├── synthetic_data.py                 # NEW — synthetic expr generator, parameterized by your pasted-back stats
├── attribute_weighting.py            # NEW — the 7 weighting models + normalize/aggregate/select
├── Feature_Weighting.ipynb           # NEW — orchestration notebook, runs on synthetic data
├── requirements.txt                  # NEW — pin pandas/numpy/scipy/scikit-learn/skrebate/openpyxl
└── restructured/
    ├── {female,male}_filtered_gene_symbols.txt   # NEW — gene names only, safe to load here
    ├── {female,male}_expr_synthetic.csv          # NEW — synthetic data
    ├── {female,male}_synthetic_injected_genes.txt # NEW — for pipeline verification
    └── {female,male}_attribute_weights.csv       # NEW — final output: weight table per sex
```

`attribute_weighting.py` is dataset-agnostic (takes any `X`/`y`), so it
already satisfies "isolate loading/cleaning logic per dataset" for this
stage. If/when a second cachexia dataset is added later for cross-dataset
Aim-3 work, `Data_PreProcessing.ipynb`'s loading logic should be extracted
into its own module then — out of scope now.

---

## Step 1 — Local aggregate-stats script (you run this, not me)

`compute_aggregate_stats.py` reproduces the existing FDR<0.05 + expression
(mean ≥50 in ≥1 group) filter already coded in `Data_PreProcessing.ipynb`,
then:
1. Writes gene **symbols only** (no values) to
   `restructured/{sex}_filtered_gene_symbols.txt` — safe to bring into this
   session.
2. Prints **aggregate scalars only** (per sex, per group): count of genes,
   count of samples, mean/SD of `log1p(per-gene mean)` across genes, 1st/99th
   percentile of per-gene means, median/IQR of per-gene coefficient of
   variation, and fraction of zero entries. Never a full dataframe, never a
   per-gene or per-sample number.

You run this locally/offline and paste back only the printed scalar blocks.
Expected gene counts to sanity-check against: male 4257, female 2881
(already known from the existing notebook run).

---

## Step 2 — Synthetic data generator (`synthetic_data.py`)

Uses (a) the real gene symbol list (names only) and (b) your pasted-back
aggregate stats to generate `{sex}_expr_synthetic.csv`: correct shape (8
samples × filtered gene count + Sample/Group/Label), correct dtypes,
lognormal-drawn per-gene baselines (clipped to your reported percentiles),
negative-binomial (falling back to Poisson where CV implies it) counts per
replicate, and the real 4-SHAM/4-C26 sample-naming convention. A configurable
number of genes (default 25) get an artificial mean shift injected in C26
(default 4×) purely to give the verification step something recoverable —
written to `{sex}_synthetic_injected_genes.txt` and clearly commented as
synthetic, not biological signal.

**Guard rail:** a `USE_REAL_DATA = False` flag at the top of
`Feature_Weighting.ipynb`, with a comment that it may only be flipped to
`True` for a local/offline run outside this session.

---

## Step 3 — Attribute weighting module (`attribute_weighting.py`)

Seven per-gene weight functions, each `(X: pd.DataFrame, y: pd.Series) ->
pd.Series` of raw scores, mapped to the paper's models as the closest
available Python equivalents:

| Paper model | Python implementation |
|---|---|
| Weight by Information Gain | `sklearn.feature_selection.mutual_info_classif` (with `random_state`) |
| Weight by Information Gain Ratio | Custom: IG(feature)/H(feature), using Fayyad-Irani MDL discretization |
| Weight by Rule | Custom OneR-style: best single-threshold split, training accuracy as score |
| Weight by Chi Squared Statistic | `sklearn.feature_selection.chi2` (counts are already non-negative) |
| Weight by Gini Index | Univariate per-gene depth-1 `DecisionTreeClassifier` stump, Gini importance |
| Weight by Uncertainty | Custom symmetric uncertainty: SU = 2·IG/(H(feature)+H(label)), same MDL discretization as IG Ratio |
| Weight by Relief | `skrebate.ReliefF`, `n_neighbors=3` (bounded by min class size − 1 = 3 at n=4/group) |

**Methodology choices (flagged, not silently assumed):**
- *Discretization for IG Ratio / Uncertainty*: the paper doesn't publish
  RapidMiner's internal binning parameters. RapidMiner's "Weight by X"
  operators share naming/concepts with Weka's `AttributeSelection`
  evaluators, whose standard convention for these three entropy-based
  methods is **supervised MDL discretization (Fayyad & Irani, 1993)** —
  implemented here rather than an arbitrary fixed bin count. This is an
  informed inference, not a verified fact about RapidMiner's closed-source
  internals — documented as such in the module docstring.
- *Gini Index*: no standard external equivalent to point to; implemented as
  a univariate CART-style Gini-impurity-decrease-from-a-single-split (depth-1
  stump per gene), matching how Gini is used everywhere else in the ML
  literature and consistent with the paper's per-attribute (not joint/
  ensemble) weighting philosophy.
- *Rule scoring*: single-pass training accuracy (no CV), consistent with the
  paper's single-pass design and the earlier decision to defer CV to a later
  iteration — flagged as more overfit-prone at n=8 than at the paper's n=88.
- *Normalization edge case*: if a model's raw scores are constant for a sex
  (no signal), all genes get 0.0 for that column (not 0.5) to avoid
  artificially inflating WeightSum.

Aggregation: min-max normalize each of the 7 raw score columns to [0,1],
sum → `WeightSum` (0–7), rank descending, `Selected = WeightSum > 5` (same
relative cutoff as the paper). Run independently per sex via
`run_attribute_weighting(expr_df, gene_cols, sex=...)`.

**Limitations documented in the module docstring and the notebook's top
markdown cell:** single-pass weighting at n=8/sex (no CV/LASSO stability
averaging yet — flagged as a TODO for a later iteration, not built now);
candidate gene set is FDR<0.05 + expression filtered (a deviation from the
paper's expression-only filter, kept because it's already implemented and
more sensible at this sample size); sex run independently, not pooled;
ReliefF's neighbor count is bounded by class size and must be re-derived if
a future dataset has a different balance.

---

## Step 4 — Orchestration notebook (`Feature_Weighting.ipynb`)

1. Top markdown cell with all limitations/caveats above, plus the
   `USE_REAL_DATA` guard note.
2. Load `{sex}_expr_synthetic.csv` for both sexes.
3. Run `run_attribute_weighting` per sex.
4. Save full weight tables to `restructured/{sex}_attribute_weights.csv`
   (columns: `Gene, IG, IGRatio, Rule, Chi2, Gini, Uncertainty, Relief,
   WeightSum, Rank, Selected` — mirrors the paper's Table 3), and the
   `Selected` gene names to `restructured/{sex}_signature_genes.txt`.
5. Verification cells (below), run against synthetic data before any real
   run is ever attempted.
6. Closing note: pattern-discovery/classifier stage (paper §2.4) is the next
   stage, explicitly deferred.

**Small addition to `Data_PreProcessing.ipynb`:** save the already-computed
filtered gene lists (currently computed but discarded) to
`restructured/{sex}_filtered_gene_symbols.txt`, so this becomes the single
source of truth for both the local stats script and the synthetic generator
instead of duplicating the filter logic.

---

## Verification plan (synthetic data only)

1. Shape/dtype assertions on `{sex}_expr_synthetic.csv` (8 samples, expected
   gene count, 4/4 Group split, Label consistent with Group).
2. No-NaN/Inf assertions after each raw-score function and after
   normalization; fail loudly with model name + affected gene count rather
   than silently `fillna`.
3. Toy unit-test cell: a few hand-constructed genes with known perfect
   separation vs. pure noise, asserting all 7 models score them near 1.0 /
   near 0.0 respectively — catches bugs in the 3 custom (non-library)
   functions before running at scale.
4. **Positive control:** after running the full pipeline, check what
   fraction of the ~25 injected genes land in `Selected`/top-50 ranks —
   this is the main end-to-end pass/fail signal.
5. **Negative control:** confirm non-injected genes have a visibly lower/
   more spread WeightSum distribution than injected genes.
6. Confirm male vs. female weight tables are genuinely different (not
   accidentally pooled/duplicated).
7. Runtime sanity check on the full ~4257/2881-gene pass (expected: seconds
   at n=8; note vectorizing the entropy loops as a later optimization if not).

---

## Critical files

- `attribute_weighting.py` — new
- `synthetic_data.py` — new
- `compute_aggregate_stats.py` — new (run locally by user only)
- `Feature_Weighting.ipynb` — new
- `Data_PreProcessing.ipynb` — small addition (save filtered gene symbol lists)
- `requirements.txt` — new (pandas, numpy, scipy, scikit-learn, skrebate, openpyxl)

## Next steps after this draft (not built now)

- Paper §2.4 pattern-discovery/classifier stage (11 models, CV) on the
  selected signature genes.
- k-fold weight-stability averaging and/or LASSO as an 8th weighting signal,
  once the single-pass version is validated.
- If a second cachexia dataset is added for cross-dataset Aim-3 work, extract
  `Data_PreProcessing.ipynb`'s loading/cleaning logic into its own module.
