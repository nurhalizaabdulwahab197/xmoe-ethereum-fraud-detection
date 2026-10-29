# Reproducibility Guide

This guide records the verified environment, dataset artifacts, experiment runners, and manuscript-aligned evaluation protocols for the X-MoE Ethereum fraud-detection study.

## 1. Public experimental scope

The manuscript compares five models:

- MLP
- GCN
- GAT
- GraphSAGE
- X-MoE

The main reproduction entry point is:

```text
src/reproduce.py
```

Model definitions are stored in:

```text
src/models/
```

with:

```text
gcn.py
gat.py
graphsage.py
mlp.py
xmoe.py
__init__.py
```

`__init__.py` only exports model classes and is not an experiment runner.

Additional manuscript experiments use:

```text
experiments/ablation.py
experiments/strict_inductive.py
```

The ablation runner also reports connectivity-stratified metrics, so no separate connectivity runner is required.

## 2. Verified CPU environment

```text
Python:              3.11.9
pip:                 24.0
PyTorch:             2.14.1+cpu
PyTorch Geometric:   2.8.0.post1
NumPy:               2.4.6
Pandas:              3.0.6
scikit-learn:        1.9.1
SciPy:               1.17.1
CUDA available:      False
```

The tested dependency snapshot is stored in `requirements-cpu-tested.txt`.

## 3. Dataset artifacts

The study uses processed BLTE artifacts released by the upstream project.

Required local files:

```text
upstream-paper/BLTE_multiclass_graph.pt
upstream-paper/BLTE_multiclass_v2.csv
```

Verified SHA-256 hashes:

```text
BLTE_multiclass_graph.pt
2859A8A3A815EDD02B5CD85D46C00D730B2B66C4E8D047145DB2E843C9E38456

BLTE_multiclass_v2.csv
02E376DDD98BA595372F19731FE5E51BBC90BF897B2A52B3C8BB98E786423B11
```

The repository does not reconstruct the raw Ethereum data-collection or labelling pipeline.

## 4. Main clean protocol

For each seed, the clean protocol:

1. generates a fresh stratified 70/15/15 train/validation/test split;
2. fits `StandardScaler` on training nodes only;
3. applies the scaler to validation and test nodes;
4. retains the released BLTE topology across partitions;
5. trains for 300 epochs;
6. selects the checkpoint with the highest validation Macro-F1;
7. evaluates once on the held-out test partition.

Final seeds:

```text
100,101,102,103,104,105,106,107,108,109
```

This clean protocol is topology-visible. It should not be described as strict node-inductive.

## 5. Reference X-MoE configuration

```text
base_loss = weighted_ce
expert_loss_weight = 0.25
scam_specialist_loss_weight = 0.5
x_moe_expert_task_loss_weight = 1.0
x_moe_normal_phishing_penalty = 0.30
```

The value `0.30` is the manuscript reference setting, not a uniquely optimal coefficient.

## 6. Main reproduction commands

Check all model choices:

```powershell
python .\src\reproduce.py --help
```

Expected model choices:

```text
{gcn,gat,graphsage,mlp,x_moe_gnn}
```

### MLP

```powershell
python .\src\reproduce.py `
  --model mlp `
  --protocol clean `
  --seeds 100,101,102,103,104,105,106,107,108,109 `
  --epochs 300 `
  --device cpu `
  --deterministic `
  --upstream .\upstream-paper `
  --output-dir .\results\mlp
```

### GCN

```powershell
python .\src\reproduce.py `
  --model gcn `
  --protocol clean `
  --seeds 100,101,102,103,104,105,106,107,108,109 `
  --epochs 300 `
  --device cpu `
  --deterministic `
  --upstream .\upstream-paper `
  --output-dir .\results\gcn
```

### GAT

```powershell
python .\src\reproduce.py `
  --model gat `
  --protocol clean `
  --seeds 100,101,102,103,104,105,106,107,108,109 `
  --epochs 300 `
  --device cpu `
  --deterministic `
  --upstream .\upstream-paper `
  --output-dir .\results\gat
```

### GraphSAGE

```powershell
python .\src\reproduce.py `
  --model graphsage `
  --protocol clean `
  --seeds 100,101,102,103,104,105,106,107,108,109 `
  --epochs 300 `
  --device cpu `
  --deterministic `
  --upstream .\upstream-paper `
  --output-dir .\results\graphsage
```

### X-MoE

```powershell
python .\src\reproduce.py `
  --model x_moe_gnn `
  --protocol clean `
  --seeds 100,101,102,103,104,105,106,107,108,109 `
  --epochs 300 `
  --device cpu `
  --deterministic `
  --upstream .\upstream-paper `
  --x-moe-normal-phishing-penalty 0.30 `
  --output-dir .\results\xmoe
```

## 7. Verified clean-protocol results

| Model | Accuracy mean | Accuracy std | Macro-F1 mean | Macro-F1 std |
|---|---:|---:|---:|---:|
| GCN | 0.881123 | 0.032853 | 0.363689 | 0.020485 |
| GAT | 0.591693 | 0.033604 | 0.266492 | 0.017115 |
| MLP | 0.967175 | 0.008147 | 0.371498 | 0.008483 |
| GraphSAGE | 0.988188 | 0.002071 | 0.491619 | 0.033561 |
| **X-MoE** | **0.996942** | **0.001153** | **0.657494** | **0.071187** |

Machine-readable clean result summaries are stored under:

```text
results/paper/
results/reference/
```

## 8. Paired statistical comparison

The manuscript uses paired seed-level Macro-F1, two-sided exact Wilcoxon signed-rank tests, Holm correction across the four baseline comparisons, paired effect size `d_z`, and paired mean-difference confidence intervals.

| Comparison | Mean difference | Holm-adjusted p | d_z | X-MoE wins |
|---|---:|---:|---:|---:|
| X-MoE vs GCN | 0.293805 | 0.007812 | 4.309086 | 10/10 |
| X-MoE vs GAT | 0.391002 | 0.007812 | 5.298819 | 10/10 |
| X-MoE vs MLP | 0.285995 | 0.007812 | 4.323348 | 10/10 |
| X-MoE vs GraphSAGE | 0.165875 | 0.007812 | 2.680799 | 10/10 |

Detailed values are stored in `results/reference/clean_baseline_statistics.csv` and `results/paper/statistical_tests.csv`.

## 9. Normal–Phishing coefficient sensitivity

Evaluated values:

```text
0.00, 0.10, 0.30, 0.50, 1.00
```

| λ_NP | Macro-F1 | Phishing F1 | Scamming F1 |
|---:|---:|---:|---:|
| 0.00 | 0.5526 ± 0.0451 | 0.2379 ± 0.0766 | 0.4226 ± 0.1082 |
| 0.10 | 0.6489 ± 0.0662 | 0.5438 ± 0.1363 | 0.4044 ± 0.1007 |
| 0.30 | 0.6575 ± 0.0712 | 0.5374 ± 0.1433 | 0.4363 ± 0.1010 |
| 0.50 | 0.6580 ± 0.0746 | 0.5551 ± 0.1647 | 0.4203 ± 0.1174 |
| 1.00 | 0.6523 ± 0.0776 | 0.5402 ± 0.1569 | 0.4180 ± 0.1255 |

The main sensitivity occurs when the objective is removed. Positive values from 0.10 to 1.00 form a broad high-performing region.

Machine-readable summary:

```text
results/lambda_sensitivity/summary.csv
```

The sensitivity experiments can be reproduced using `src/reproduce.py` by changing `--x-moe-normal-phishing-penalty` while keeping the remaining clean-protocol settings fixed.

## 10. Ablation study

Runner:

```text
experiments/ablation.py
```

This runner implements A0–A7 and also reports connected/isolated subgroup metrics.

Reference results:

| Variant | Mean Macro-F1 | Δ vs A0 |
|---|---:|---:|
| A0 Full X-MoE | 0.6575 | — |
| A1 Conflict-Aware only | 0.6525 | -0.0050 |
| A2 GraphSAGE-expert configuration | 0.6558 | -0.0017 |
| A3 Shared routing | 0.6917 | +0.0342 |
| A4 No isolate masking | 0.7071 | +0.0496 |
| A5 No Scamming specialist | 0.6557 | -0.0018 |
| A6 No Normal–Phishing objective | 0.5526 | -0.1049 |
| A7 Fixed 50:50 outer mixture | 0.6912 | +0.0337 |

Only A6 shows a family-wise significant overall degradation after Holm correction. The reference A0 configuration should not be described as numerically optimal.

Check runner options with:

```powershell
python .\experiments\ablation.py --help
```

## 11. Connectivity-stratified analysis

The ablation runner contains the connectivity-stratified evaluation used for the A0 vs A4 analysis.

Connected-node results:

```text
A0 Macro-F1 = 0.6524
A4 Macro-F1 = 0.7273
```

For isolated Normal nodes:

```text
A0 false-positive rate = 0.000000
A4 false-positive rate = 0.000307
```

All fraud test nodes in this analysis are connected. Therefore the isolated subgroup contains only Normal test nodes, and the A0/A4 result is interpreted as a performance–safety trade-off.

## 12. STRICT-INDUCTIVE protocol

Runner:

```text
experiments/strict_inductive.py
```

Display label:

```text
STRICT-INDUCTIVE
```

Code protocol key:

```text
strict_inductive
```

The protocol preserves the released edge direction and does not synthesise reverse edges.

- Training: training features and train–train topology only.
- Validation: train + validation features; train–train, train–validation and validation–train edges; no validation–validation edges and no test nodes/edges.
- Test: train + test features; train–train, train–test and test–train edges; no test–test edges and no validation nodes/edges.

Hard leakage assertions are included in the runner.

Verified Macro-F1:

| Model | Mean ± std |
|---|---:|
| GCN | 0.358331 ± 0.020646 |
| GAT | 0.277051 ± 0.023603 |
| GraphSAGE | 0.440139 ± 0.020355 |
| X-MoE | 0.547783 ± 0.058151 |

X-MoE exceeds GraphSAGE in all ten matched seeds. The Holm-adjusted p-value for the X-MoE vs GraphSAGE comparison is 0.005859.

This experiment is a topology-visibility sensitivity check. It must not be presented as evidence of temporal, cross-chain, or cross-dataset generalisation, and the clean-to-strict change should not be described as proof of leakage.

Check runner options with:

```powershell
python .\experiments\strict_inductive.py --help
```

## 13. Result locations

Manuscript-facing summaries:

```text
results/paper/
```

Reference tables:

```text
results/reference/
```

Normal–Phishing sensitivity summary:

```text
results/lambda_sensitivity/summary.csv
```

Important manuscript-facing files include:

```text
results/paper/clean_baseline_10seed.csv
results/paper/minority_class_summary.csv
results/paper/ablation_summary.csv
results/paper/connectivity_analysis.csv
results/paper/efficiency_summary.csv
results/paper/strict_inductive_10seed.csv
results/paper/clean_to_strict_summary.csv
results/paper/statistical_tests.csv
```

## 14. Reproducibility expectations

Exact bit-for-bit equality across all hardware and operating systems is not guaranteed. Small numerical differences may arise from CPU architecture, operating system, numerical libraries, thread scheduling, and floating-point implementation details.

The principal reproduced clean-protocol ordering is:

```text
X-MoE > GraphSAGE > MLP ≈ GCN > GAT
```

with X-MoE obtaining the highest mean Macro-F1 and winning all ten paired clean-protocol seeds against each evaluated baseline.

## 15. GPU reference results

The repository also contains a previously verified GPU environment and reference results:

```text
requirements-gpu-tested.txt
REPRODUCTION_METADATA_GPU.txt
results/reference/gpu_10seed_summary.csv
```

These GPU results predate the manuscript-aligned MLP comparison and therefore cover GCN, GAT, GraphSAGE, and X-MoE only.

## 16. Demo

https://xmoe-ethereum-fraud-detection.streamlit.app/

The demo is intended for research communication and model exploration. Quantitative claims in the manuscript are based on controlled multi-seed experiments rather than individual demo predictions.
