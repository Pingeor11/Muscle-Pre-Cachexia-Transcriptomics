# Ideas from Gruber, Olsen, Hernando, Cheng et al. 2026 (Nature) — "Integrated signatures define mutational processes in prostate cancer"

## Context

PPCG consortium paper, 959 human prostate cancer whole genomes. Integrates
four types of mutational signature (SBS, indel, copy-number, structural
variant) into 8 "integrated mutational footprints" (IMFs) via hierarchical
clustering, links them to clinical outcomes.

Different domain (human WGS mutational signatures vs. our mouse RNA-seq
attribute weighting, n=8) — most of the core methodology (NMF signature
extraction, chromothripsis calling, Cox survival modelling) doesn't transfer.
Reviewed on 2026-09-19 for anything worth borrowing. Decision at the time:
**no action taken — logged for possible later reference.**

## What we already do similarly

- Combine several independent methods into one consensus (their 4 signature
  types → 8 IMFs; our 7 attribute-weighting methods + LASSO → shortlist)
- Check robustness by resampling (their cluster stability/silhouette scores;
  our leave-one-pair-out stability selection in `attribute_weighting.py`)
- Plan to validate against external, independently-published data (their
  TCGA-PRAD replication; our planned overlap check against Zhao et al. 2024 /
  Zhang et al. 2024 gene lists)

## What could help make our gene list more secure

- **Report effect sizes, not just ranks/p-values.** They only trust an
  association if it clears a minimum effect size (Cohen's d), not just
  significance. Cheap to add to our final gene table.
- **Pre-declare exclusion criteria before looking at results.** They have a
  strict flowchart (Extended Data Fig. 1) of which samples get dropped and
  why, decided up front. Worth doing the same for any gene/sample exclusions.
- **Use a real permutation-based null**, not an off-the-shelf test's p-value,
  for checking whether our gene list is "enriched" for known pathways —
  consistent with the earlier caution that naive GSEA/ORA has known
  false-positive inflation.
- **Build the "known biology" gene list from multiple documented sources**,
  not ad hoc. They merged several curated databases (cancer driver
  compendia, Reactome, DDR pathway lists) into one reference set
  (Methods: "Selection of 1,747 genes of interest"). We should do the same
  for the cachexia-pathway annotation layer instead of hand-picking genes.

All four are cheap, presentation-strengthening additions, not new analysis
pipelines — worth revisiting if time allows, but not scoped into current work.
