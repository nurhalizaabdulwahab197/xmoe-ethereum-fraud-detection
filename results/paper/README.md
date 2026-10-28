# Manuscript Result Summaries

This directory is the manuscript-facing result index for the X-MoE paper.

The files here are curated summaries of the result values used in the manuscript. They are intended to provide a single, unambiguous reference point for tables, figures, captions, and claims in the paper. They are not a replacement for raw per-seed outputs or checkpoints.

## Canonical clean result set

The manuscript uses the fresh-PC deterministic 10-seed clean results:

- seeds: 100-109
- epochs: 300
- protocol: clean
- feature scaling: fitted on training nodes only
- released BLTE topology retained
- X-MoE Normal-Phishing penalty: 0.30

Primary clean Macro-F1 values:

- GCN: 0.363689 +/- 0.020485
- GAT: 0.266492 +/- 0.017115
- GraphSAGE: 0.491619 +/- 0.033561
- X-MoE: 0.657494 +/- 0.071187

X-MoE vs GraphSAGE:

- absolute improvement: 0.165875
- percentage-point improvement: 16.59
- relative improvement: approximately 33.74%

## Files

- `clean_baseline_10seed.csv`: clean baseline accuracy and Macro-F1 summary.
- `minority_class_summary.csv`: clean and strict minority-class F1/AP values used in the manuscript.
- `ablation_summary.csv`: A0-A7 mean Macro-F1 and change relative to A0.
- `connectivity_analysis.csv`: connected-node and isolated-Normal comparison for A0 vs A4.
- `strict_inductive_10seed.csv`: STRICT-INDUCTIVE-V1 Macro-F1 summary and matched comparisons.
- `statistical_tests.csv`: manuscript-level statistical results that have been explicitly verified.
- `efficiency_summary.csv`: parameter count, observed training time, and Macro-F1 for A0-A7.

## Provenance and interpretation

Historical result summaries produced before the final fresh-PC alignment remain recoverable through Git history. The current manuscript should use the values in this directory and the aligned files under `results/reference/`.

Do not infer unreported p-values, per-seed values, confidence intervals, or class metrics from these summary files. Where a cell is blank, the corresponding value has not been promoted here as a verified manuscript result.

The clean protocol is topology-visible and must not be described as strict inductive. STRICT-INDUCTIVE-V1 is reported separately and restricts graph visibility to training-training and training-target edges while excluding target-target message passing.
