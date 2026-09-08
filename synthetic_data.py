"""
Synthetic expression-matrix generator for the attribute-weighting pipeline.

Uses ONLY:
  (a) the real gene-symbol lists (names only — restructured/{sex}_filtered_gene_symbols.txt)
  (b) aggregate scalars computed locally by compute_aggregate_stats.py and pasted back
      (hardcoded below in GENE_STATS)

No real expression values, sample values, or per-gene numbers are used anywhere
in this module. Safe to develop/run inside an AI-assisted session per CLAUDE.md.

Per-gene generation:
  1. Draw a per-gene log1p(mean) from Normal(log1p_mean, log1p_sd) for the gene's
     group, then expm1 it and clip to [p1, p99] of that group's gene-mean percentiles.
  2. Draw a per-gene CV from a lognormal matched to the group's reported
     median/IQR of gene CV.
  3. Convert (mean, cv) to a negative-binomial (mean, dispersion r); if the implied
     variance <= mean (CV too small for NB), fall back to Poisson.
  4. Draw one count per replicate (4 SHAM + 4 C26) independently.

A configurable number of genes get their C26-group mean multiplied by
INJECTION_FOLD before step 3 — purely an artificial recoverable signal for
pipeline verification, NOT biological signal. Written to
{sex}_synthetic_injected_genes.txt and never treated as ground truth.
"""

import os

import numpy as np
import pandas as pd

OUT_DIR = "restructured"
N_REPLICATES_PER_GROUP = 4
N_INJECTED_DEFAULT = 25
INJECTION_FOLD_DEFAULT = 4.0

# Pasted back from compute_aggregate_stats.py — aggregate scalars only, no
# per-gene/per-sample values. Re-paste here if the local run is ever redone.
GENE_STATS = {
    "female": {
        "SHAM": {"log1p_mean": 6.0248, "log1p_sd": 1.3300, "p1": 29.20, "p99": 10369.65,
                  "cv_median": 0.0988, "cv_q1": 0.0659, "cv_q3": 0.1533, "zero_frac": 0.0002},
        "C26":  {"log1p_mean": 5.9948, "log1p_sd": 1.4703, "p1": 22.15, "p99": 12506.70,
                  "cv_median": 0.1726, "cv_q1": 0.1149, "cv_q3": 0.2666, "zero_frac": 0.0000},
    },
    "male": {
        "SHAM": {"log1p_mean": 6.0135, "log1p_sd": 1.3656, "p1": 26.89, "p99": 12887.14,
                  "cv_median": 0.1801, "cv_q1": 0.1410, "cv_q3": 0.2371, "zero_frac": 0.0001},
        "C26":  {"log1p_mean": 6.0314, "log1p_sd": 1.3989, "p1": 24.50, "p99": 12980.75,
                  "cv_median": 0.1788, "cv_q1": 0.1250, "cv_q3": 0.2572, "zero_frac": 0.0000},
    },
}

# Expected gene counts, for a sanity check against the symbol-list files.
EXPECTED_GENE_COUNTS = {"male": 4257, "female": 2881}


def load_gene_symbols(sex, out_dir=OUT_DIR):
    path = os.path.join(out_dir, f"{sex}_filtered_gene_symbols.txt")
    with open(path) as f:
        genes = [line.strip() for line in f if line.strip()]
    expected = EXPECTED_GENE_COUNTS[sex]
    if len(genes) != expected:
        print(f"WARNING: {sex} gene symbol count {len(genes)} != expected {expected}")
    return genes


def _cv_lognormal_sigma(q1, q3):
    """Sigma of a lognormal whose IQR matches (q1, q3): IQR_log = 2 * 0.6745 * sigma."""
    return (np.log(q3) - np.log(q1)) / (2 * 0.6745)


def _draw_group_means(rng, n_genes, stats):
    log_means = rng.normal(stats["log1p_mean"], stats["log1p_sd"], size=n_genes)
    means = np.expm1(log_means)
    return np.clip(means, stats["p1"], stats["p99"])


def _draw_cvs(rng, n_genes, stats):
    sigma = _cv_lognormal_sigma(stats["cv_q1"], stats["cv_q3"])
    return rng.lognormal(mean=np.log(stats["cv_median"]), sigma=sigma, size=n_genes)


def _draw_counts(rng, means, cvs, n_replicates):
    """NB(mean, cv) per gene, one draw per replicate; Poisson fallback if var <= mean."""
    n_genes = len(means)
    out = np.empty((n_replicates, n_genes))

    variances = (cvs * means) ** 2
    use_nb = variances > means

    # NB branch: mean-dispersion parameterization, r = mean^2 / (var - mean)
    if use_nb.any():
        mu = means[use_nb]
        var = variances[use_nb]
        r = mu ** 2 / (var - mu)
        p = r / (r + mu)
        for rep in range(n_replicates):
            out[rep, use_nb] = rng.negative_binomial(r, p)

    # Poisson fallback where CV implies underdispersion relative to NB's floor
    if (~use_nb).any():
        mu = means[~use_nb]
        for rep in range(n_replicates):
            out[rep, ~use_nb] = rng.poisson(mu)

    return out


def generate_synthetic_expr(sex, seed=0, n_injected=N_INJECTED_DEFAULT,
                              injection_fold=INJECTION_FOLD_DEFAULT, out_dir=OUT_DIR):
    """Returns (expr_df, injected_genes) for one sex. expr_df mirrors the real
    {sex}_expr.csv layout: Sample, Group, Label, <gene columns>."""
    rng = np.random.default_rng(seed)
    genes = load_gene_symbols(sex, out_dir)
    n_genes = len(genes)

    sham_means = _draw_group_means(rng, n_genes, GENE_STATS[sex]["SHAM"])
    c26_means = _draw_group_means(rng, n_genes, GENE_STATS[sex]["C26"])
    sham_cvs = _draw_cvs(rng, n_genes, GENE_STATS[sex]["SHAM"])
    c26_cvs = _draw_cvs(rng, n_genes, GENE_STATS[sex]["C26"])

    injected_idx = rng.choice(n_genes, size=min(n_injected, n_genes), replace=False)
    injected_genes = [genes[i] for i in injected_idx]
    c26_means_injected = c26_means.copy()
    c26_means_injected[injected_idx] *= injection_fold

    sham_counts = _draw_counts(rng, sham_means, sham_cvs, N_REPLICATES_PER_GROUP)
    c26_counts = _draw_counts(rng, c26_means_injected, c26_cvs, N_REPLICATES_PER_GROUP)

    counts = np.vstack([sham_counts, c26_counts])
    sample_ids = (
        [f"SHAM_{i + 1}" for i in range(N_REPLICATES_PER_GROUP)]
        + [f"C26_{i + 1}" for i in range(N_REPLICATES_PER_GROUP)]
    )
    groups = ["SHAM"] * N_REPLICATES_PER_GROUP + ["C26"] * N_REPLICATES_PER_GROUP

    expr_df = pd.DataFrame(counts, columns=genes)
    expr_df.insert(0, "Sample", sample_ids)
    expr_df["Group"] = groups
    expr_df["Label"] = (expr_df["Group"] == "C26").astype(int)

    return expr_df, injected_genes


def save_synthetic(sex, out_dir=OUT_DIR, **kwargs):
    expr_df, injected_genes = generate_synthetic_expr(sex, out_dir=out_dir, **kwargs)

    expr_path = os.path.join(out_dir, f"{sex}_expr_synthetic.csv")
    genes_path = os.path.join(out_dir, f"{sex}_synthetic_injected_genes.txt")

    expr_df.to_csv(expr_path, index=False)
    with open(genes_path, "w") as f:
        f.write("\n".join(injected_genes))

    print(f"{sex}: saved {expr_df.shape} synthetic matrix -> {expr_path}")
    print(f"{sex}: saved {len(injected_genes)} injected genes -> {genes_path}")


if __name__ == "__main__":
    for sex in ("female", "male"):
        save_synthetic(sex)
