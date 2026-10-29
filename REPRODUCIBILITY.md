# Reproducibility Guide

This guide records the verified environment and manuscript-aligned evaluation protocol for the X-MoE Ethereum fraud-detection study.

## 1. Public experimental scope

The manuscript comparison includes five models:

- MLP
- GCN
- GAT
- GraphSAGE
- X-MoE

The original runner remains available at `src/reproduce.py`. The manuscript-synchronised entry point is `src/reproduce_manuscript.py`; it extends the same runner with the feature-only MLP baseline.

Use:

```powershell
python .\src\reproduce_manuscript.py --help
```

Expected model choices:

```text
{gcn,gat,graphsage,mlp,x_moe_gnn}
```

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

The study uses the processed BLTE artifacts released by the upstream project.

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

The repository does not independently reconstruct the raw Ethereum data-collection or labelling pipeline.

## 4. Topology-visible clean protocol

For each seed, the clean protocol:

1. generates a fresh stratified 70/15/15 train/validation/test split;
2. fits `StandardScaler` on training nodes only;
3. applies the scaler to validation and test nodes;
4. retains the released BLTE topology;
5. trains for 300 epochs;
6. selects the checkpoint with the highest validation Macro-F1;
7. evaluates once on the held-out test partition.

Final seeds:

```text
100,101,102,103,104,105,106,107,108,109
```

## 5. Reference X-MoE configuration

```text
base_loss = weighted_ce
expert_loss_weight = 0.25
scam_specialist_loss_weight = 0.5
x_moe_expert_task_loss_weight = 1.0
x_moe_normal_phishing_penalty = 0.30
```

The value `0.30` is the manuscript reference setting, not a uniquely optimal coefficient.

## 6. Full clean-protocol commands

### MLP

```powershell
python .\src\reproduce_manuscript.py `
  --model mlp `
  --protocol clean `
  --seeds 100,101,102,103,104,105,106,107,108,109 `
  --epochs 300 `
  --device cpu `
  --deterministic `
  --upstream .\upstream-paper `
  --output-dir .\results\final_mlp
```

### GCN

```powershell
python .\src\reproduce_manuscript.py `
  --model gcn `
  --protocol clean `
  --seeds 100,101,102,103,104,105,106,107,108,109 `
  --epochs 300 `
  --device cpu `
  --deterministic `
  --upstream .\upstream-paper `
  --output-dir .\results\final_gcn
```

### GAT

```powershell
python .\src\reproduce_manuscript.py `
  --model gat `
  --protocol clean `
  --seeds 100,101,102,103,104,105,106,107,108,109 `
  --epochs 300 `
  --device cpu `
  --deterministic `
  --upstream .\upstream-paper `
  --output-dir .\results\final_gat
```

### GraphSAGE

```powershell
python .\src\reproduce_manuscript.py `
  --model graphsage `
  --protocol clean `
  --seeds 100,101,102,103,104,105,106,107,108,109 `
  --epochs 300 `
  --device cpu `
  --deterministic `
  --upstream .\upstream-paper `
  --output-dir .\results\final_graphsage
```

### X-MoE

```powershell
python .\src\reproduce_manuscript.py `
  --model x_moe_gnn `
  --protocol clean `
  --seeds 100,101,102,103,104,105,106,107,108,109 `
  --epochs 300 `
  --device cpu `
  --deterministic `
  --upstream .\upstream-paper `
  --x-moe-normal-phishing-penalty 0.30 `
  --output-dir .\results\final_xmoe
```

## 7. Verified clean-protocol results

| Model | Accuracy mean | Accuracy std | Macro-F1 mean | Macro-F1 std |
|---|---:|---:|---:|---:|
| GCN | 0.881123 | 0.032853 | 0.363689 | 0.020485 |
| GAT | 0.591693 | 0.033604 | 0.266492 | 0.017115 |
| MLP | 0.967175 | 0.008147 | 0.371498 | 0.008483 |
| GraphSAGE | 0.988188 | 0.002071 | 0.491619 | 0.033561 |
| **X-MoE** | **0.996942** | **0.001153** | **0.657494** | **0.071187** |

The machine-readable reference table is stored in:

```text
results/reference/manuscript_clean_results.csv
```

## 8. Paired statistical comparison

The manuscript uses paired seed-level Macro-F1, two-sided exact Wilcoxon signed-rank tests, Holm correction across the four clean baseline comparisons, paired effect size `d_z`, and 95% confidence intervals of the paired differences.

| Comparison | Mean difference | Holm-adjusted p | d_z | X-MoE wins |
|---|---:|---:|---:|---:|
| X-MoE vs GCN | 0.293805 | 0.007812 | 4.309086 | 10/10 |
| X-MoE vs GAT | 0.391002 | 0.007812 | 5.298819 | 10/10 |
| X-MoE vs MLP | 0.285995 | 0.007812 | 4.323348 | 10/10 |
| X-MoE vs GraphSAGE | 0.165875 | 0.007812 | 2.680799 | 10/10 |

Full values are stored in `results/reference/clean_baseline_statistics.csv`.

## 9. Normal–Phishing coefficient sensitivity

The evaluated values are:

```text
0.00, 0.10, 0.30, 0.50, 1.00
```

The main finding is that removal of the objective causes the largest degradation, whereas positive values from 0.10 to 1.00 remain within a broad high-performing region.

Machine-readable results are stored in:

```text
results/reference/lambda_np_sensitivity.csv
```

## 10. Strict node-inductive and connectivity analyses

The manuscript additionally reports:

- connectivity-stratified analysis of connected vs isolated nodes;
- an A0 vs A4 isolate-masking comparison;
- STRICT-INDUCTIVE-V1, which excludes target–target and cross-evaluation message passing;
- the A0–A7 ablation study.

The numerical results are documented in the manuscript-aligned README and reference result files. The advanced experiment runners used during the refactor stage are being migrated separately from the local research workspace; they should not be inferred to exist in `src/reproduce.py` until that migration is complete.

This distinction is intentional so that the public repository does not claim reproducibility for scripts that have not yet been committed.

## 11. Reproducibility expectations

Exact bit-for-bit equality across all hardware and operating systems is not guaranteed. Small numerical differences may arise from CPU architecture, operating system, numerical libraries, thread scheduling, and floating-point implementation details.

The principal reproduced finding under the verified CPU protocol is:

```text
X-MoE > GraphSAGE > MLP ≈ GCN > GAT
```

with X-MoE obtaining the highest mean Macro-F1 and winning all ten paired clean-protocol seeds against every evaluated baseline.

## 12. GPU results

The repository also contains the previously verified GPU environment and reference results in:

```text
requirements-gpu-tested.txt
REPRODUCTION_METADATA_GPU.txt
results/reference/gpu_10seed_summary.csv
```

These GPU results predate the manuscript-aligned MLP addition and therefore cover GCN, GAT, GraphSAGE, and X-MoE only.

## 13. Demo

The public research demo is available at:

https://xmoe-ethereum-fraud-detection.streamlit.app/

The demo is for research communication and model exploration. Quantitative claims in the manuscript are based on controlled multi-seed experiments rather than individual demo predictions.
