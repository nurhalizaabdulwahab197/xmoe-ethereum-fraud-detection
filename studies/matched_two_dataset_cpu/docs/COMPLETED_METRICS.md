# Audited completed metrics

Audit publication prepared on 2026-11-12. Values below are percentages, mean ± sample SD over ten seeds. Full per-class metrics, AP, weighted metrics, balanced accuracy and MCC are in the linked CSV files.

The 100 original matched clean study runs are complete: 50 BLTE + 50 RQ1.
The ten fresh seed-100 runs are reproduction checks of that study, not a new
independent ten-seed study; their newly added AUC has not been separately audited
here. BLTE and RQ1 are trained/evaluated separately; there is no merged-dataset
score or transferred checkpoint. Features remain dataset-native (23 versus five).

**RQ1 journal 230-run suite: in progress, not yet fully verified.** This status
reflects the author's report on 12 November 2026; no completed-run count is
asserted from this cloud checkout. Do not use smoke, synthetic or partially
completed runs as final journal performance results. The implementation-validation
record describes software tests, not scientific completion. The planned 230
runs comprise clean/strict 100 + A0-A7 80 + sensitivity 50; optional GPU parity is
a separate additional scope. See [RQ1_JOURNAL_PARITY.md](RQ1_JOURNAL_PARITY.md).

See [metric definitions, provenance and verification](METRIC_METHODS.md).

## BLTE matched clean: five models, ten seeds, 300 epochs.

Original saved predictions; all 100 prediction hashes match published run records.

| Entry | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Macro ROC-AUC OvR |
|---|---:|---:|---:|---:|---:|
| MLP | 96.72 ± 0.81 | 35.72 ± 0.51 | 61.27 ± 6.53 | 37.15 ± 0.85 | 91.24 ± 6.09 |
| GCN | 88.11 ± 3.29 | 36.22 ± 1.41 | 74.41 ± 3.89 | 36.37 ± 2.05 | 94.29 ± 2.25 |
| GAT | 59.17 ± 3.36 | 34.35 ± 0.65 | 58.35 ± 8.57 | 26.65 ± 1.71 | 81.79 ± 3.38 |
| GraphSAGE | 98.82 ± 0.21 | 44.39 ± 3.79 | 74.55 ± 7.34 | 49.16 ± 3.36 | 95.58 ± 3.29 |
| X-MoE | 99.69 ± 0.12 | 63.93 ± 6.68 | 75.41 ± 7.15 | 65.75 ± 7.12 | 99.05 ± 0.81 |

[Every seed](../results/metrics_completion/blte_clean/per_seed.csv) · [All means/SD and per-class metrics](../results/metrics_completion/blte_clean/summary.csv) · [Audit record](../results/metrics_completion/blte_clean/validation.json)

## RQ1 matched clean: five models, ten seeds, 300 epochs.

Original saved predictions; all 100 prediction hashes match published run records.

| Entry | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Macro ROC-AUC OvR |
|---|---:|---:|---:|---:|---:|
| MLP | 63.91 ± 2.65 | 66.98 ± 1.92 | 69.84 ± 2.09 | 65.77 ± 1.40 | 81.42 ± 2.03 |
| GCN | 80.85 ± 1.26 | 74.93 ± 2.23 | 73.51 ± 1.77 | 74.08 ± 1.83 | 89.65 ± 0.83 |
| GAT | 72.35 ± 2.11 | 61.82 ± 2.45 | 56.76 ± 2.20 | 58.34 ± 2.09 | 80.34 ± 2.26 |
| GraphSAGE | 89.32 ± 0.60 | 85.18 ± 1.05 | 87.43 ± 0.66 | 86.19 ± 0.71 | 96.71 ± 0.31 |
| X-MoE | 91.71 ± 0.95 | 88.64 ± 1.32 | 90.07 ± 0.87 | 89.23 ± 1.00 | 98.02 ± 0.30 |

[Every seed](../results/metrics_completion/rq1_clean/per_seed.csv) · [All means/SD and per-class metrics](../results/metrics_completion/rq1_clean/summary.csv) · [Audit record](../results/metrics_completion/rq1_clean/validation.json)

## What is not changed

Historical source result tables, training code, hyperparameters and checkpoint selection are retained. There is no test-score-based selection of seeds/variants, no new AUC significance claim, and no change to the live demo or manuscript. Publication scope is described per series; counts across branches/ablation/sensitivity overlap and must not be summed as independent trials.
