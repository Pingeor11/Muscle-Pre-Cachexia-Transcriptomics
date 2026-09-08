"""
Run this locally/offline only — NEVER inside an AI-assisted session.

Reproduces the FDR<0.05 + expression (mean >= EXPR_THRESHOLD in >=1 group)
filter from Data_PreProcessing.ipynb directly against Muscle_Male_Female_All.xlsx,
then prints AGGREGATE SCALARS ONLY per sex/group — no per-gene or per-sample
values, no dataframe. Also (re)writes the gene-symbol-only lists so they stay
in sync with the filter reproduced here.

Paste the printed "=== aggregate stats ===" blocks back into the AI session;
nothing else from this script's output should be shared there.
"""

import os

import numpy as np
import pandas as pd

SIG_THRESHOLD = 0.05
EXPR_THRESHOLD = 50
XLSX_PATH = "Muscle_Male_Female_All.xlsx"
OUT_DIR = "restructured"

# Sanity-check counts from the existing Data_PreProcessing.ipynb run.
EXPECTED_GENE_COUNTS = {"male": 4257, "female": 2881}


def restructure(df):
    stats = df[["logFC", "FDR"]].copy()
    counts = df.drop(columns=["logFC", "FDR"])

    expr = counts.T
    expr.index.name = "Sample"
    expr = expr.reset_index()
    expr["Group"] = expr["Sample"].apply(lambda s: "SHAM" if "SHAM" in s else "C26")
    expr["Label"] = (expr["Group"] == "C26").astype(int)

    return expr, stats


def filter_low_expression(expr_df, sig_genes, threshold=EXPR_THRESHOLD):
    group_means = expr_df.groupby("Group")[sig_genes].mean()
    keep_genes = []
    for gene in sig_genes:
        if (group_means.loc["SHAM", gene] >= threshold) or (
            group_means.loc["C26", gene] >= threshold
        ):
            keep_genes.append(gene)
    return keep_genes


def print_aggregate_stats(sex, expr_df, keep_genes):
    """Print aggregate scalars only — per sex, per group. No per-gene/per-sample values."""
    print(f"\n=== aggregate stats: {sex} ===")
    print(f"n_genes: {len(keep_genes)}")
    print(f"n_samples: {len(expr_df)}")

    for group in ("SHAM", "C26"):
        sub = expr_df.loc[expr_df["Group"] == group, keep_genes]
        gene_means = sub.mean(axis=0)
        gene_cv = sub.std(axis=0, ddof=1) / gene_means.replace(0, np.nan)

        log_means = np.log1p(gene_means)
        zero_frac = (sub == 0).to_numpy().mean()

        print(f"-- group: {group} --")
        print(f"  n_samples_in_group: {len(sub)}")
        print(f"  log1p(gene_mean): mean={log_means.mean():.4f} sd={log_means.std(ddof=1):.4f}")
        print(
            "  gene_mean percentiles: "
            f"p1={np.percentile(gene_means, 1):.4f} p99={np.percentile(gene_means, 99):.4f}"
        )
        cv_valid = gene_cv.dropna()
        q1, q3 = np.percentile(cv_valid, [25, 75])
        print(
            "  gene_cv: median="
            f"{np.median(cv_valid):.4f} iqr=({q1:.4f}, {q3:.4f})"
        )
        print(f"  zero_fraction: {zero_frac:.4f}")


def main():
    raw = pd.read_excel(XLSX_PATH, sheet_name="Male_Female_All")

    female = raw.iloc[:, 0:11].copy().set_index("Gene")
    male = raw.iloc[:, 12:23].copy()
    male.columns = male.columns.str.replace(".1", "", regex=False)
    male = male.set_index("Gene")

    os.makedirs(OUT_DIR, exist_ok=True)

    for sex, block in (("female", female), ("male", male)):
        expr, stats = restructure(block)

        sig_genes = stats.index[stats["FDR"] < SIG_THRESHOLD]
        expr_sig = expr[["Sample", "Group", "Label"] + list(sig_genes)]

        gene_cols = [c for c in expr_sig.columns if c not in ("Sample", "Group", "Label")]
        keep_genes = filter_low_expression(expr_sig, gene_cols)

        expected = EXPECTED_GENE_COUNTS[sex]
        if len(keep_genes) != expected:
            print(
                f"WARNING: {sex} filtered gene count {len(keep_genes)} != "
                f"expected {expected} — filter logic may have drifted from "
                "Data_PreProcessing.ipynb, double-check before trusting these stats."
            )

        with open(os.path.join(OUT_DIR, f"{sex}_filtered_gene_symbols.txt"), "w") as f:
            f.write("\n".join(keep_genes))

        print_aggregate_stats(sex, expr_sig, keep_genes)


if __name__ == "__main__":
    main()
