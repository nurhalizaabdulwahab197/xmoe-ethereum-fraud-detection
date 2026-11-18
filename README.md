# X-MoE for Ethereum Fraud Detection

Code, experiment records and reproducibility guides for multi-class Ethereum address classification (Normal, Phishing, Scamming) on **two separate datasets: BLTE and ScamSweeper/RQ1**. Dataset-specific code, protocols, results and environments are separated below. Tables are not pooled across datasets.

## Choose the dataset first

| | BLTE | ScamSweeper / RQ1 |
|---|---|---|
| Graph | 73,033 addresses; 6,588 released directed edges | 1,322,252 nodes; 3,856,855 directed edge entries |
| Labelled cohort | 73,033: Normal 72,865 / Phishing 69 / Scamming 99 | 4,913: Normal 636 / Phishing 1,165 / Scamming 3,112 |
| Unlabelled context | See released BLTE artifacts | 1,317,339 nodes; excluded from supervision |
| Features | 23 released features | 5 observed-graph activity/counterparty features |
| Clean isolation | Approximately 92.32% | No degree-zero nodes in the prepared clean graph |
| Code | [experiments/blte](experiments/blte/README.md) | [experiments/scamsweeper](experiments/scamsweeper/README.md) |
| Results | [results/blte](results/blte/README.md) | [results/scamsweeper](results/scamsweeper/README.md) |
| Guides | [BLTE reproduction](docs/blte/REPRODUCTION.md) | [GPU reproduction](docs/scamsweeper/REPRODUCE_GPU.md) |

`rq1` is the internal name preserved in the ScamSweeper snapshot. It does not mean the BLTE ablation, and is not the separate window5/session preprocessing dataset.

## Experiments completed and their scope

Both datasets have clean comparison, strict evaluation, A0–A7 component ablation and Normal–Phishing coefficient sensitivity. Seeds are 100–109, with 300 epochs for the principal matrices. See the [complete experiment inventory](docs/EXPERIMENT_INVENTORY.md) for exact counts, reuse and additional analyses.

| Family | BLTE evidence | ScamSweeper locked GPU suite |
|---|---|---|
| Clean: MLP, GCN, GAT, GraphSAGE, X-MoE | 50 model/seed rows | 50 fresh runs |
| Strict: same five models | 50 model/seed rows | 50 fresh runs |
| X-MoE A0–A7 | 80 variant/seed rows | 80 fresh runs |
| Lambda = 0, .1, .3, .5, 1 | 50 rows; metric audit attributes 20 reused A0/A6 controls | 50 fresh runs |
| Historical four-GNN CPU/GPU study | Separate, documented BLTE evidence | Not a controlled CPU/GPU comparison |

Connectivity, minority metrics, routing, timing, convergence and paired statistics are analyses of these runs, not automatically additional training experiments. Smoke tests are not paper results. The **reproduction branch** separately contains BLTE reconstruction, CE-only controls and **PCGrad ON/OFF**; it must not be merged conceptually with the **A0–A7** ablation here.

## Locked ScamSweeper/RQ1 results

All 230 full runs completed on NVIDIA H200 NVL with CUDA; preflight passed exact input/mask/scaler hashes for all ten seeds. Mean ± sample SD of Macro-F1:

| Model | Clean | Strict |
|---|---:|---:|
| MLP | 0.6584 ± 0.0160 | 0.6584 ± 0.0160 |
| GCN | 0.7426 ± 0.0159 | 0.6372 ± 0.0273 |
| GAT | 0.5775 ± 0.0263 | 0.5385 ± 0.0500 |
| GraphSAGE | 0.8618 ± 0.0078 | 0.7526 ± 0.0162 |
| X-MoE | 0.8910 ± 0.0098 | 0.7828 ± 0.0158 |

X-MoE exceeds GraphSAGE by 2.93 percentage points clean and 3.03 strict (Holm-adjusted p=.0078125 in each family). Only A0 vs A2 is significant in the seven-comparison ablation family; no significant incremental advantage over A1 or A3–A7 was established. A0 remains the reference, not necessarily the numerically highest ablation mean. [Full ablation/statistics](docs/scamsweeper/RESULTS.md).

This is an **exploratory** extension. Strict restricts feature/topology visibility using the same fixed-snapshot feature values; it is not a temporal or fully feature-recomputed leakage-isolation test. No isolated nodes occur in the clean graph, so A4 cannot establish the benefit of isolate masking for isolated accounts. [Protocol and limitations](docs/scamsweeper/PROTOCOL.md).

## BLTE results and reproduction

The historical five-model BLTE clean reference is 0.3715 (MLP), 0.3637 (GCN), 0.2665 (GAT), 0.4916 (GraphSAGE), 0.6575 (X-MoE), as mean Macro-F1; SDs, class metrics and provenance are in the [BLTE guide](docs/blte/REPRODUCTION.md) and [completed metric audit](docs/COMPLETED_METRICS.md). These values are BLTE historical CPU evidence, not the H200 Sweeper series.

The [BLTE guide](docs/blte/REPRODUCTION.md) retains complete setup, model architecture, clean/strict commands, BLTE A0–A7 table, lambda sensitivity and historical GTX 1660 SUPER GPU results. The [dataset-labelled BLTE launchers](experiments/blte/README.md) make defaults and output paths explicit. Root requirements-cpu/gpu and ENVIRONMENT_CPU/GPU describe BLTE; use the separate Sweeper freeze for H200 experiments.

## Repository organization

| Path | Scope |
|---|---|
| experiments/blte/ | Dataset-labelled BLTE entry points |
| experiments/scamsweeper/ | Sweeper launcher, family guides and exact frozen source subtree |
| results/blte/ | Dataset-labelled views of historical BLTE evidence |
| results/scamsweeper/ | Locked GPU tables, separated families and per-run provenance |
| docs/blte/, docs/scamsweeper/ | Dataset-specific reproduction and writing guidance |
| docs/EXPERIMENT_INVENTORY.md | Full inventory, counts, reuse and branch boundaries |
| docs/DATASET_PATH_MAP.json | Explicit path/provenance mapping |
| src/ and experiments/ablation.py, strict_inductive.py | Original BLTE scientific implementations retained for compatibility |
| results/clean, strict, ablation, sensitivity, cpu_gpu, metrics_completion, tables | Historical BLTE compatibility paths; never write Sweeper results here |

This organization preserves old result bytes, manifests, paths and implementation imports. New experiments should use a fresh dataset-labelled output directory. The captured Sweeper engine keeps its original relative tree so that its source inventory and frozen dependency hashes remain verifiable. It does not use the current root BLTE engine by accident.

## Reproduce and verify

- BLTE: [setup and commands](experiments/blte/README.md); [original metric verification](docs/METRIC_METHODS.md).
- Sweeper: [setup and preflight → smoke → full](experiments/scamsweeper/README.md); [independent audit](results/scamsweeper/provenance/INDEPENDENT_AUDIT.json).
- Run the repository organization check: `python scripts/check_dataset_organization.py` (standard library only).
- Run existing BLTE completed-metric checks in the recorded BLTE environment: `python scripts/check_completed_metrics.py`.
- [Paper-writing guidance for Sweeper](docs/scamsweeper/MANUSCRIPT_GUIDE.md).

GitHub organization did not retrain models or modify locked numerical outcomes. Public tables do not bundle raw Sweeper arrays, address mapping, checkpoint binaries or all archived predictions/histories. The verified full private snapshot and its SHA256 are documented in results/scamsweeper/README.md. Original completion flags describe the unused optional CPU/GPU comparison, not run devices; all 230 stored device values are cuda.

## Related workstreams and attribution

[reproduction branch](https://github.com/nurhalizaabdulwahab197/xmoe-ethereum-fraud-detection/tree/reproduction): separate reconstructed BLTE study (120 primary runs, 40 PCGrad rows, six separate fixed-mask pilots); archived evidence verified, public-wrapper GPU validation caveat retained. [two-dataset-comparison branch](https://github.com/nurhalizaabdulwahab197/xmoe-ethereum-fraud-detection/tree/two-dataset-comparison): historical independent workstream; not automatically rewritten by this integration.

Source code is MIT licensed; see LICENSE. BLTE inputs must be obtained from the upstream project [salam-ammari/Graph-Neural-Network-Based-Fraudulent-Address-Classification-in-Ethereum-Network](https://github.com/salam-ammari/Graph-Neural-Network-Based-Fraudulent-Address-Classification-in-Ethereum-Network). Dataset rights are separate from code rights. The Sweeper/RQ1 snapshot documents label/archive provenance, but a canonical external dataset citation/access and redistribution statement still needs to be established before data release or manuscript submission. No ownership of upstream datasets is claimed.

Previously published [research demo](https://xmoe-ethereum-fraud-detection.streamlit.app/) is an external deployment reference; this update does not alter it or establish its deployed version. The manuscript LaTeX is maintained separately.
