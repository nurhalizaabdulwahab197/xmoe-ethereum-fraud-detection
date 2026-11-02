# X-MoE BLTE experiments

Run experimental commands from this folder (`main` branch). `src/runner.py` replaces `src/reproduce.py`; model code and training settings are unchanged. Keep the special Python filename `__init__.py`.

## Files and results

| Path | Purpose |
|---|---|
| src/runner.py | Main training/evaluation entry point |
| src/models/ | Five model implementations |
| experiments/strict_inductive.py | Restricted-topology evaluation |
| experiments/ablation.py | A0-A7 component analyses |
| experiments/summarize_results.py | Recompute curated tables without training |
| ENVIRONMENT_CPU.txt, ENVIRONMENT_GPU.txt | Historical tested environments; four-model records |
| requirements-cpu.txt, requirements-gpu.txt | Tested dependencies |
| results/clean/reference/ | Five-model clean seed records |
| results/strict/reference/ | Audited 50-run strict series and statistics |
| results/strict/runs/ | Detailed metrics, summaries and predictions for five models across seeds 100?109 |
| results/ablation/reference/ | Ablation evidence |
| results/cpu_gpu/ | CPU/GPU comparison summaries; GPU records cover four models |
| results/sensitivity/ | Parameter-sensitivity outputs |
| results/tables/ | Curated tables and generated protocol metrics |
| figures/ | Available research figures |
| RESULT_SOURCE_HASHES.json | Current locations and byte hashes of evidence files |
| FILE_RELOCATION_MAP.json | Old-to-new path mapping |

## Dataset

Obtain processed BLTE artifacts from https://github.com/salam-ammari/Graph-Neural-Network-Based-Fraudulent-Address-Classification-in-Ethereum-Network . Required files: BLTE_multiclass_graph.pt and BLTE_multiclass_v2.csv. This workspace does not contain an audited reconstruction of the raw Ethereum collection/labels. Supply their actual directory with --upstream.

## Commands that do not train

```powershell
python src/runner.py --help
python experiments/strict_inductive.py --help
python experiments/ablation.py --help
python experiments/summarize_results.py
```

The summarizer prints/checks tables by default. Its --apply flag writes only this worktree's results/tables; publication to sibling branches is separate. Existing individual run filenames remain unchanged to preserve traceability.

## Workflow

The three local folders are separate Git worktrees. Do not merge entire role branches. Update reviewed outputs selectively. Original result CSV contents and historical absolute provenance fields are preserved; use FILE_RELOCATION_MAP.json for current locations.

Scientific status: clean CPU covers five models; strict CPU covers five models and 50 audited runs. Archived GPU clean summaries cover four models, not MLP.


## Detailed experimental protocol

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
src/runner.py
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

The tested dependency snapshot is stored in `requirements-cpu.txt`.

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
python .\src\runner.py --help
```

Expected model choices:

```text
{gcn,gat,graphsage,mlp,x_moe_gnn}
```

### MLP

```powershell
python .\src\runner.py `
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
python .\src\runner.py `
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
python .\src\runner.py `
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
python .\src\runner.py `
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
python .\src\runner.py `
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
results/tables/
```

## 8. Paired statistical comparison

The manuscript uses paired seed-level Macro-F1, two-sided exact Wilcoxon signed-rank tests, Holm correction across the four baseline comparisons, paired effect size `d_z`, and paired mean-difference confidence intervals.

| Comparison | Mean difference | Holm-adjusted p | d_z | X-MoE wins |
|---|---:|---:|---:|---:|
| X-MoE vs GCN | 0.293805 | 0.007812 | 4.309086 | 10/10 |
| X-MoE vs GAT | 0.391002 | 0.007812 | 5.298819 | 10/10 |
| X-MoE vs MLP | 0.285995 | 0.007812 | 4.323348 | 10/10 |
| X-MoE vs GraphSAGE | 0.165875 | 0.007812 | 2.680799 | 10/10 |


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
results/sensitivity/lambda/summary.csv
```

The sensitivity experiments can be reproduced using `src/runner.py` by changing `--x-moe-normal-phishing-penalty` while keeping the remaining clean-protocol settings fixed.

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
| MLP | 0.371498 ± 0.008483 |
| GCN | 0.358331 ± 0.020646 |
| GAT | 0.277051 ± 0.023603 |
| GraphSAGE | 0.440139 ± 0.020355 |
| X-MoE | 0.547783 ± 0.058151 |

X-MoE exceeds MLP, GCN, GAT and GraphSAGE in all ten matched strict seeds. Holm correction across four baselines gives p=0.0078125 for each comparison. MLP Macro-F1 is identical between clean and strict for every seed.

This experiment is a topology-visibility sensitivity check. It must not be presented as evidence of temporal, cross-chain, or cross-dataset generalisation, and the clean-to-strict change should not be described as proof of leakage.

Check runner options with:

```powershell
python .\experiments\strict_inductive.py --help
```

## 13. Result locations

Manuscript-facing summaries:

```text
results/tables/
```

Reference tables:

```text
```

Normal–Phishing sensitivity summary:

```text
results/sensitivity/lambda/summary.csv
```

Important manuscript-facing files include:

```text
results/tables/clean_baseline_10seed.csv
results/tables/minority_class_summary.csv
results/tables/ablation_summary.csv
results/tables/connectivity_analysis.csv
results/tables/efficiency_summary.csv
results/tables/strict_inductive_10seed.csv
results/tables/clean_to_strict_summary.csv
results/tables/statistical_tests.csv
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
requirements-gpu.txt
ENVIRONMENT_GPU.txt
results/cpu_gpu/gpu_10seed_summary.csv
```

These GPU results predate the manuscript-aligned MLP comparison and therefore cover GCN, GAT, GraphSAGE, and X-MoE only.

## 16. Demo

https://xmoe-ethereum-fraud-detection.streamlit.app/

The demo is intended for research communication and model exploration. Quantitative claims in the manuscript are based on controlled multi-seed experiments rather than individual demo predictions.

## Five-model result alignment

See results/tables/AUDIT_STATUS.md for evidence limitations. Historical alignment documents are retained in the local backup. Run `python experiments/summarize_results.py` to check the archived five-model tables without training. Clean-to-strict Holm correction covers five models: GAT p=0.05859375 is not significant; GraphSAGE p=0.009765625 and X-MoE p=0.015625 indicate decreases. The strict minority family contains four F1/AP metrics; recomputed Scamming F1 p=0.275390625 remains non-significant. The four-model CPU/GPU metadata files are historical environment comparisons and are not extended to imply MLP GPU runs.


## Ablation and evaluation details

# Experiment Runners

This directory contains the manuscript-specific validation runners that complement the main reproduction entry point in `src/runner.py`.

## Entry points

- `ablation.py` — A0–A7 X-MoE ablation study, including connectivity-stratified evaluation.
- `strict_inductive.py` — five-model STRICT-INDUCTIVE evaluation with restricted topology visibility.
- `summarize_five_model_results.py` — checks/rebuilds paper and demo tables from frozen reference CSVs without training.

The main five-model clean-protocol comparison is run through `src/runner.py` using:

```text
{gcn,gat,graphsage,mlp,x_moe_gnn}
```

## Ablation variants

| ID | Display label | Change from A0 |
|---|---|---|
| A0 | Full X-MoE | Reference configuration |
| A1 | Conflict-Aware only | Removes the external GraphSAGE Expert |
| A2 | GraphSAGE-expert configuration | Removes the Conflict-Aware Multitask Expert |
| A3 | Shared routing | Replaces class-specific routing with shared routing |
| A4 | No isolate masking | Removes isolate-safe suppression of graph contribution |
| A5 | No Scamming specialist | Removes the Scamming-vs-rest specialist objective |
| A6 | No Normal-Phishing objective | Removes the connected-Normal Phishing objective |
| A7 | Fixed 50:50 outer mixture | Uses fixed outer expert weights on connected nodes |

The manuscript does not claim that every component is individually necessary. A6 is the strongest harmful ablation and the only overall degradation that remains family-wise significant after Holm correction. A3, A4, and A7 can obtain higher mean Macro-F1 than A0, so they must be interpreted as architecture trade-offs rather than evidence that A0 is numerically optimal.

## Main experimental settings

Unless otherwise stated, manuscript experiments use:

- stratified 70/15/15 train/validation/test splits regenerated per seed;
- seeds 100–109;
- 300 epochs;
- training-only feature standardisation;
- deterministic execution;
- validation checkpoint selection by highest Macro-F1;
- released BLTE graph topology for the clean protocol.

Reference X-MoE objective weights:

```text
expert_loss_weight = 0.25
scam_specialist_loss_weight = 0.50
x_moe_expert_task_loss_weight = 1.00
x_moe_normal_phishing_penalty = 0.30
```

## Connectivity analysis

Connectivity-stratified evaluation is implemented in `ablation.py`; a separate connectivity runner is not required.

The A0 vs A4 analysis is interpreted as a trade-off. Removing isolate masking improves connected-node discrimination in the reported experiment, but introduces a small false-positive cost among isolated Normal addresses.

## STRICT-INDUCTIVE

`strict_inductive.py` implements the manuscript's strict node-inductive sensitivity protocol.

For each evaluation stage, only admissible nodes and edges are exposed:

- training: training-node features and train–train topology;
- validation: training + validation features, with train–train and train↔validation topology, excluding validation–validation and all test information;
- test: training + test features, with train–train and train↔test topology, excluding test–test and all validation information.

The released edge direction is preserved and reverse edges are not synthesised.

STRICT-INDUCTIVE is a sensitivity analysis of topology visibility. It should not be described as evidence of temporal, cross-chain, or cross-dataset generalisation.

## Normal–Phishing coefficient sensitivity

The reference coefficient is `lambda_NP = 0.30`. The evaluated values are:

```text
0.00, 0.10, 0.30, 0.50, 1.00
```

The primary finding is that removing the objective (`lambda_NP = 0`) causes a substantial reduction, whereas positive values from 0.10 to 1.00 form a broad high-performing region. The value 0.30 is therefore retained as the reference setting, not claimed as a unique optimum.

Machine-readable sensitivity results are available in `results/sensitivity/lambda/summary.csv`.

## Dataset scope

The BLTE dataset is not redistributed by this repository. The study uses the released processed BLTE artifacts and does not independently reconstruct the raw Ethereum collection or label-generation pipeline.

See the root `README.md` and `README.md` for dataset hashes, environment details, reproduction commands, and manuscript result tables.

MLP is included in both protocols and ignores edges. All ten matched strict MLP seeds were run and match clean Macro-F1 exactly. See the root README for commands and the explicit Holm test families.


## Local archive policy




## Active evidence and local archives

The canonical strict series is results/strict/reference/strict_50_runs.csv.
GCN seed 108 is one of the ten primary GCN runs, located under
results/strict/runs/gcn/seed_108/. It is not an additional control run.

Historical intermediate aggregates, redundant documentation and the
additional MLP demo-validation outputs are archived outside this repository.
They are not required for the main five-model comparison.
The demo branch retains inference assets for all five models.

<!-- BEGIN COMBINED FIVE-MODEL ARCHITECTURE -->
## Model architecture

![MLP, GCN, GAT, GraphSAGE and X-MoE architecture comparison](figures/architecture_comparison.png)

The figure adds the feature-only MLP baseline alongside the original GCN, GAT, GraphSAGE and X-MoE layout. MLP uses two 64-unit linear layers with LayerNorm, ReLU and dropout 0.5, followed by a three-class classifier. Graph edges are unused by MLP. The figure documents the implementation; it does not alter any model or result.
<!-- END COMBINED FIVE-MODEL ARCHITECTURE -->
