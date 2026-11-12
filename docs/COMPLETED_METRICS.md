# Audited completed metrics

Audit publication prepared on 2026-11-12. Values below are percentages, mean ± sample SD over ten seeds. Full per-class metrics, AP, weighted metrics, balanced accuracy and MCC are in the linked CSV files.

The main BLTE clean/strict, A0-A7, coefficient sensitivity and four-model
historical CPU/GPU comparison now have audited overall and per-class Accuracy,
Precision, Recall, F1, ROC-AUC and AP. This does not assert completion of every
pilot or every subgroup AUC. Fixed-mask pilots and the RQ1 journal suite are
separate scopes. Hardware/parameter-count and routing tables remain descriptive
analyses of their documented runs, not extra independent experiments.

See [metric definitions, provenance and verification](METRIC_METHODS.md).

## BLTE clean: five models, ten seeds, 300 epochs.

Original saved predictions from the matched CPU series; scores coincide with the main clean reference.

| Entry | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Macro ROC-AUC OvR |
|---|---:|---:|---:|---:|---:|
| MLP | 96.72 ± 0.81 | 35.72 ± 0.51 | 61.27 ± 6.53 | 37.15 ± 0.85 | 91.24 ± 6.09 |
| GCN | 88.11 ± 3.29 | 36.22 ± 1.41 | 74.41 ± 3.89 | 36.37 ± 2.05 | 94.29 ± 2.25 |
| GAT | 59.17 ± 3.36 | 34.35 ± 0.65 | 58.35 ± 8.57 | 26.65 ± 1.71 | 81.79 ± 3.38 |
| GraphSAGE | 98.82 ± 0.21 | 44.39 ± 3.79 | 74.55 ± 7.34 | 49.16 ± 3.36 | 95.58 ± 3.29 |
| X-MoE | 99.69 ± 0.12 | 63.93 ± 6.68 | 75.41 ± 7.15 | 65.75 ± 7.12 | 99.05 ± 0.81 |

[Every seed](../results/metrics_completion/clean/per_seed.csv) · [All means/SD and per-class metrics](../results/metrics_completion/clean/summary.csv) · [Audit record](../results/metrics_completion/clean/validation.json)

## BLTE strict: five models, ten seeds, 300 epochs.

Original GitHub prediction CSVs; the published utility recomputed scores without training.

| Entry | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Macro ROC-AUC OvR |
|---|---:|---:|---:|---:|---:|
| MLP | 96.72 ± 0.81 | 35.72 ± 0.51 | 61.27 ± 6.53 | 37.15 ± 0.85 | 91.24 ± 6.09 |
| GCN | 88.59 ± 4.07 | 35.84 ± 1.22 | 73.67 ± 5.23 | 35.83 ± 2.06 | 92.79 ± 2.03 |
| GAT | 61.17 ± 4.63 | 34.66 ± 0.88 | 60.57 ± 7.93 | 27.71 ± 2.36 | 82.45 ± 3.89 |
| GraphSAGE | 98.10 ± 0.39 | 40.27 ± 1.71 | 71.54 ± 3.47 | 44.01 ± 2.04 | 94.32 ± 4.56 |
| X-MoE | 99.28 ± 0.62 | 51.27 ± 7.52 | 71.61 ± 2.91 | 54.78 ± 5.82 | 95.84 ± 3.28 |

[Every seed](../results/metrics_completion/strict/per_seed.csv) · [All means/SD and per-class metrics](../results/metrics_completion/strict/summary.csv) · [Audit record](../results/metrics_completion/strict/validation.json)

## BLTE A0-A7 component ablation: 80 runs, 300 epochs.

Supplementary CPU rerun; 1,280 original recorded metric values and all 80 confusion matrices reproduced exactly. AUC and F1 use the same supplementary predictions.

| Entry | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Macro ROC-AUC OvR |
|---|---:|---:|---:|---:|---:|
| a0_full | 99.69 ± 0.12 | 63.93 ± 6.68 | 75.41 ± 7.15 | 65.75 ± 7.12 | 99.05 ± 0.81 |
| a1_conflict_only | 99.61 ± 0.18 | 63.81 ± 6.58 | 77.24 ± 7.16 | 65.25 ± 7.02 | 99.10 ± 0.53 |
| a2_graphsage_only | 99.77 ± 0.05 | 61.22 ± 4.91 | 74.20 ± 7.61 | 65.58 ± 4.95 | 96.00 ± 3.51 |
| a3_global_routing | 99.78 ± 0.09 | 66.42 ± 7.73 | 76.71 ± 5.61 | 69.17 ± 6.26 | 99.55 ± 0.32 |
| a4_no_isolate_mask | 99.83 ± 0.05 | 67.87 ± 6.67 | 75.95 ± 6.34 | 70.71 ± 6.19 | 97.28 ± 1.67 |
| a5_no_scam_specialist | 99.68 ± 0.14 | 63.74 ± 6.52 | 75.23 ± 6.74 | 65.57 ± 6.14 | 99.26 ± 0.50 |
| a6_no_normal_phishing_penalty | 99.39 ± 0.18 | 48.86 ± 4.13 | 76.16 ± 5.13 | 55.26 ± 4.51 | 98.82 ± 1.00 |
| a7_fixed_50 | 99.79 ± 0.04 | 66.50 ± 6.31 | 75.80 ± 5.96 | 69.12 ± 5.26 | 99.24 ± 1.05 |

[Every seed](../results/metrics_completion/ablation/per_seed.csv) · [All means/SD and per-class metrics](../results/metrics_completion/ablation/summary.csv) · [Audit record](../results/metrics_completion/ablation/validation.json)

## BLTE Normal-Phishing coefficient sensitivity: five coefficients, ten seeds.

30 new CPU runs (lambda 0.10, 0.50, 1.00) plus 20 attributed reused ablation runs (A6=lambda 0.00, A0=lambda 0.30). All 60 available original summary values match within 1e-12. Reused rows are not independent extra runs.

| Entry | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Macro ROC-AUC OvR |
|---|---:|---:|---:|---:|---:|
| 0.0 | 99.39 ± 0.18 | 48.86 ± 4.13 | 76.16 ± 5.13 | 55.26 ± 4.51 | 98.82 ± 1.00 |
| 0.1 | 99.64 ± 0.14 | 62.65 ± 6.38 | 76.25 ± 7.15 | 64.89 ± 6.62 | 99.21 ± 0.53 |
| 0.3 | 99.69 ± 0.12 | 63.93 ± 6.68 | 75.41 ± 7.15 | 65.75 ± 7.12 | 99.05 ± 0.81 |
| 0.5 | 99.65 ± 0.19 | 64.31 ± 7.32 | 76.61 ± 8.79 | 65.80 ± 7.46 | 98.84 ± 1.22 |
| 1.0 | 99.65 ± 0.19 | 64.14 ± 7.40 | 75.79 ± 8.35 | 65.23 ± 7.76 | 98.84 ± 1.03 |

[Every seed](../results/metrics_completion/sensitivity/per_seed.csv) · [All means/SD and per-class metrics](../results/metrics_completion/sensitivity/summary.csv) · [Audit record](../results/metrics_completion/sensitivity/validation.json)

## BLTE historical CUDA comparison: four GNNs, ten seeds.

30 supplementary GPU baseline runs reproduce all 480 original classification metric values and confusion matrices; X-MoE uses ten original explanation-probability files. Each AUC/F1 pair uses one prediction file.

| Entry | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Macro ROC-AUC OvR |
|---|---:|---:|---:|---:|---:|
| GCN | 88.44 ± 3.03 | 36.15 ± 1.17 | 75.82 ± 3.89 | 36.37 ± 1.76 | 94.10 ± 2.31 |
| GAT | 59.85 ± 4.42 | 34.75 ± 1.20 | 59.46 ± 7.54 | 27.50 ± 2.95 | 83.09 ± 3.92 |
| GraphSAGE | 98.84 ± 0.20 | 44.52 ± 3.12 | 75.75 ± 6.48 | 49.58 ± 3.14 | 95.09 ± 3.69 |
| X-MoE | 99.64 ± 0.17 | 65.00 ± 7.42 | 76.83 ± 6.46 | 65.78 ± 6.56 | 99.05 ± 1.03 |

[Every seed](../results/metrics_completion/historical_gpu/per_seed.csv) · [All means/SD and per-class metrics](../results/metrics_completion/historical_gpu/summary.csv) · [Audit record](../results/metrics_completion/historical_gpu/validation.json)

## What is not changed

Historical source result tables, training code, hyperparameters and checkpoint selection are retained. There is no test-score-based selection of seeds/variants, no new AUC significance claim, and no change to the live demo or manuscript. Publication scope is described per series; counts across branches/ablation/sensitivity overlap and must not be summed as independent trials.
