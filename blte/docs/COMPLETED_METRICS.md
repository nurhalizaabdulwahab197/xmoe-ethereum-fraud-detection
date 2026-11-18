# Audited completed metrics

Audit publication prepared on 2026-11-12. Values below are percentages, mean ± sample SD over ten seeds. Full per-class metrics, AP, weighted metrics, balanced accuracy and MCC are in the linked CSV files.

The 120 primary reconstruction runs and 40 PCGrad on/off runs now have
completed probability-based metrics. The six fixed-mask pilots still have the
previously documented Macro-F1 evidence and are not claimed to have a newly
completed metric audit. Public wrapper GPU execution remains separately pending
validation; importing original audit results is not a new wrapper execution.
`author` means released-code configuration; `paper` means paper-described
configuration. These are not the clean/strict main-branch protocols.

See [metric definitions, provenance and verification](METRIC_METHODS.md).

## 120 GPU reconstruction runs: six method entries, two configurations, ten saved splits.

Original WSL probabilities; 360 prediction/metric/metadata hashes match the GitHub evidence manifest.

### Released-code

| Entry | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Macro ROC-AUC OvR |
|---|---:|---:|---:|---:|---:|
| MLP | 97.04 ± 0.73 | 35.89 ± 0.78 | 60.87 ± 8.02 | 37.45 ± 1.36 | 89.61 ± 9.87 |
| GCN | 88.84 ± 2.81 | 36.21 ± 1.36 | 75.95 ± 5.51 | 36.53 ± 2.16 | 94.00 ± 2.51 |
| GAT | 61.83 ± 11.13 | 34.04 ± 0.79 | 52.47 ± 10.09 | 26.64 ± 2.50 | 77.60 ± 10.52 |
| GraphSAGE | 98.83 ± 0.23 | 43.90 ± 2.49 | 75.05 ± 7.97 | 48.92 ± 2.74 | 95.05 ± 3.79 |
| X-MoE–CE | 98.96 ± 0.19 | 43.28 ± 3.01 | 74.87 ± 7.42 | 48.57 ± 3.35 | 95.07 ± 3.88 |
| Full X-MoE | 99.68 ± 0.13 | 65.57 ± 8.00 | 78.03 ± 6.57 | 67.70 ± 6.95 | 98.92 ± 1.03 |

### Paper-described

| Entry | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Macro ROC-AUC OvR |
|---|---:|---:|---:|---:|---:|
| MLP | 96.75 ± 0.44 | 36.01 ± 0.66 | 66.88 ± 5.82 | 37.71 ± 1.18 | 87.24 ± 5.79 |
| GCN | 96.10 ± 1.37 | 38.55 ± 1.79 | 77.32 ± 4.11 | 41.43 ± 2.26 | 95.80 ± 1.80 |
| GAT | 71.86 ± 16.34 | 33.87 ± 0.94 | 52.35 ± 8.34 | 28.58 ± 4.58 | 79.04 ± 9.59 |
| GraphSAGE | 98.98 ± 0.30 | 47.66 ± 3.49 | 73.80 ± 5.91 | 52.06 ± 3.15 | 91.83 ± 4.33 |
| X-MoE–CE | 99.64 ± 0.10 | 55.58 ± 4.20 | 72.84 ± 7.44 | 60.74 ± 5.32 | 96.81 ± 2.81 |
| Full X-MoE | 99.82 ± 0.05 | 68.96 ± 5.24 | 77.20 ± 7.33 | 71.19 ± 4.98 | 99.08 ± 2.06 |

[Every seed](../results/metrics_completion/six_method/per_seed.csv) · [All means/SD and per-class metrics](../results/metrics_completion/six_method/summary.csv) · [Audit record](../results/metrics_completion/six_method/validation.json)

## 40 GPU PCGrad on/off runs: two configurations, ten matched seeds.

Original WSL probabilities; supplied metadata hashes and public Macro-F1 values reconcile, with matching initialization hashes for all 20 pairs. PCGrad metadata is not independently anchored to a public file-hash manifest.

| Entry | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Macro ROC-AUC OvR |
|---|---:|---:|---:|---:|---:|
| released-code / pcgrad_on | 99.69 ± 0.13 | 65.59 ± 7.91 | 78.03 ± 6.56 | 67.78 ± 6.83 | 98.94 ± 1.02 |
| released-code / pcgrad_off | 99.55 ± 0.10 | 51.63 ± 3.59 | 75.41 ± 6.80 | 58.08 ± 4.32 | 98.94 ± 1.03 |
| paper-described / pcgrad_on | 99.82 ± 0.08 | 70.00 ± 5.73 | 78.05 ± 6.07 | 72.19 ± 5.65 | 99.27 ± 1.03 |
| paper-described / pcgrad_off | 99.73 ± 0.09 | 59.95 ± 6.25 | 74.17 ± 5.72 | 64.18 ± 5.22 | 99.35 ± 0.39 |

[Every seed](../results/metrics_completion/pcgrad/per_seed.csv) · [All means/SD and per-class metrics](../results/metrics_completion/pcgrad/summary.csv) · [Audit record](../results/metrics_completion/pcgrad/validation.json)

## What is not changed

Historical source result tables, training code, hyperparameters and checkpoint selection are retained. There is no test-score-based selection of seeds/variants, no new AUC significance claim, and no change to the live demo or manuscript. Publication scope is described per series; counts across branches/ablation/sensitivity overlap and must not be summed as independent trials.
