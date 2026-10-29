# X-MoE for Multi-Class Ethereum Fraud Detection

This repository provides a reproducible implementation for three-class Ethereum address classification under severe class imbalance and sparse graph connectivity.

The task is:

- Normal
- Phishing
- Scamming

The manuscript evaluates five clean-protocol models:

- MLP — feature-only control
- GCN
- GAT
- GraphSAGE
- X-MoE — proposed task-aware mixture-of-experts framework

## Main manuscript result

All reported clean-protocol results use:

- stratified 70/15/15 train/validation/test splits regenerated for every seed;
- seeds 100–109;
- 300 epochs;
- train-only feature standardisation;
- deterministic execution;
- checkpoint selection by highest validation Macro-F1.

| Model | Accuracy (mean ± std) | Macro-F1 (mean ± std) |
|---|---:|---:|
| GCN | 0.8811 ± 0.0329 | 0.3637 ± 0.0205 |
| GAT | 0.5917 ± 0.0336 | 0.2665 ± 0.0171 |
| MLP | 0.9672 ± 0.0081 | 0.3715 ± 0.0085 |
| GraphSAGE | 0.9882 ± 0.0021 | 0.4916 ± 0.0336 |
| **X-MoE** | **0.9969 ± 0.0012** | **0.6575 ± 0.0712** |

Relative to GraphSAGE, X-MoE improves mean Macro-F1 by 0.1659, equivalent to 16.59 percentage points or approximately 33.74% relative improvement.

## Clean-protocol paired statistics

The main clean comparison uses paired seed-level Macro-F1, a two-sided exact Wilcoxon signed-rank test, Holm correction across the four baseline comparisons, and paired effect size d_z.

| Comparison | Δ Macro-F1 | 95% CI | Holm-adjusted p | d_z | X-MoE wins |
|---|---:|---:|---:|---:|---:|
| X-MoE vs GCN | 0.2938 | [0.2450, 0.3426] | 0.007812 | 4.309 | 10/10 |
| X-MoE vs GAT | 0.3910 | [0.3382, 0.4438] | 0.007812 | 5.299 | 10/10 |
| X-MoE vs MLP | 0.2860 | [0.2387, 0.3333] | 0.007812 | 4.323 | 10/10 |
| X-MoE vs GraphSAGE | 0.1659 | [0.1216, 0.2101] | 0.007812 | 2.681 | 10/10 |

## Class-specific performance

| Model | Normal F1 | Phishing F1 | Scamming F1 |
|---|---:|---:|---:|
| GCN | 0.9367 | 0.0132 | 0.1412 |
| GAT | 0.7430 | 0.0036 | 0.0528 |
| MLP | 0.9837 | 0.0434 | 0.0874 |
| GraphSAGE | 0.9944 | 0.1023 | 0.3782 |
| **X-MoE** | **0.9988** | **0.5374** | **0.4363** |

The largest class-specific improvement occurs for Phishing.

## X-MoE architecture

X-MoE combines two complementary predictors:

1. a Conflict-Aware Multitask Expert containing a feature branch and an internal GraphSAGE branch; and
2. an independently parameterised GraphSAGE Expert.

The framework uses:

- class-specific inner feature–graph routing;
- class-specific outer expert fusion;
- expert disagreement as a routing signal;
- node degree/connectivity as a routing signal;
- isolate-safe suppression of graph contribution for isolated nodes;
- weighted multi-class supervision;
- class-specific one-versus-rest objectives;
- a Scamming-specialist objective;
- a dedicated Normal–Phishing objective;
- PCGrad for multi-objective optimisation.

The reference Normal–Phishing coefficient is λ_NP = 0.30.

![Architecture comparison for Ethereum fraud detection](docs/figures/architecture_comparison.png)

## Key mechanism result

Removing the dedicated Normal–Phishing objective reduces mean Macro-F1 from 0.6575 to 0.5526 and Phishing F1 from 0.5374 to 0.2379. This is the strongest harmful ablation in the reported study and the only overall ablation degradation that remains family-wise significant after Holm correction.

## Connectivity-stratified result

Removing isolate masking improves connected-node discrimination but introduces a small false-positive cost among isolated Normal addresses.

| Metric | Full X-MoE | No isolate masking |
|---|---:|---:|
| Connected-node Macro-F1 | 0.6524 | 0.7273 |
| Connected Phishing F1 | 0.5374 | 0.5576 |
| Connected Scamming F1 | 0.4363 | 0.6301 |
| Isolated Normal FPR | 0.000000 | 0.000307 |

This is interpreted as a performance–safety trade-off rather than simple superiority of either configuration.

## STRICT-INDUCTIVE-V1

A stricter node-inductive protocol removes target–target and cross-evaluation message passing.

| Model | Strict Macro-F1 |
|---|---:|
| GCN | 0.3583 ± 0.0206 |
| GAT | 0.2771 ± 0.0236 |
| GraphSAGE | 0.4401 ± 0.0204 |
| **X-MoE** | **0.5478 ± 0.0582** |

X-MoE remains the strongest graph-dependent model and outperforms GraphSAGE in all ten matched seeds. The Holm-adjusted p-value for the X-MoE vs GraphSAGE strict comparison is 0.005859.

This protocol is node-inductive within BLTE; it does not establish temporal, cross-chain, or cross-dataset generalisation.

## λ_NP sensitivity

| λ_NP | Macro-F1 | Phishing F1 | Scamming F1 |
|---:|---:|---:|---:|
| 0.00 | 0.5526 ± 0.0451 | 0.2379 ± 0.0766 | 0.4226 ± 0.1082 |
| 0.10 | 0.6489 ± 0.0662 | 0.5438 ± 0.1363 | 0.4044 ± 0.1007 |
| **0.30** | 0.6575 ± 0.0712 | 0.5374 ± 0.1433 | **0.4363 ± 0.1010** |
| 0.50 | **0.6580 ± 0.0746** | **0.5551 ± 0.1647** | 0.4203 ± 0.1174 |
| 1.00 | 0.6523 ± 0.0776 | 0.5402 ± 0.1569 | 0.4180 ± 0.1255 |

The main sensitivity occurs when the objective is removed. Positive values from 0.10 to 1.00 produce a broad high-performing region, so λ_NP = 0.30 is retained as the reference configuration rather than claimed as a uniquely optimal value.

## Verified CPU environment

- Python 3.11.9
- pip 24.0
- PyTorch 2.14.1+cpu
- PyTorch Geometric 2.8.0.post1
- NumPy 2.4.6
- Pandas 3.0.6
- scikit-learn 1.9.1
- SciPy 1.17.1
- CPU execution

Install the verified CPU dependencies with:

```powershell
python -m pip install -r .\requirements-cpu-tested.txt
```

## Dataset artifacts

The BLTE dataset is not redistributed here. Obtain the processed artifacts from the upstream project:

```text
salam-ammari/Graph-Neural-Network-Based-Fraudulent-Address-Classification-in-Ethereum-Network
```

Required local files:

```text
upstream-paper/BLTE_multiclass_graph.pt
upstream-paper/BLTE_multiclass_v2.csv
```

Expected SHA-256 hashes:

```text
BLTE_multiclass_graph.pt
2859A8A3A815EDD02B5CD85D46C00D730B2B66C4E8D047145DB2E843C9E38456

BLTE_multiclass_v2.csv
02E376DDD98BA595372F19731FE5E51BBC90BF897B2A52B3C8BB98E786423B11
```

## Reproduction entry points

The original runner remains available as:

```text
src/reproduce.py
```

The manuscript-synchronised runner adds the MLP baseline while inheriting the same training and evaluation logic:

```text
src/reproduce_manuscript.py
```

Check available model choices:

```powershell
python .\src\reproduce_manuscript.py --help
```

The manuscript-synchronised choices are:

```text
{gcn,gat,graphsage,mlp,x_moe_gnn}
```

### MLP clean-protocol reproduction

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

### X-MoE clean-protocol reproduction

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

## Repository scope

This repository reproduces experiments from the released processed BLTE artifacts. It does not independently reconstruct the raw Ethereum data-collection or label-generation pipeline.

The topology-visible clean protocol retains the released graph topology across node partitions. STRICT-INDUCTIVE-V1 is used separately to test graph-dependent models under more restrictive topology visibility.

## Reproducibility resources

Additional environment and reproduction details are documented in:

- `REPRODUCIBILITY.md`
- `REPRODUCTION_METADATA_CPU.txt`
- `REPRODUCTION_METADATA_GPU.txt`
- `requirements-cpu-tested.txt`
- `requirements-gpu-tested.txt`
- `results/reference/`

## Live research demo

A public demonstration is available at:

https://xmoe-ethereum-fraud-detection.streamlit.app/

The demo is intended for research communication and model exploration. Quantitative claims in the manuscript are based on the controlled multi-seed experiments rather than individual demo predictions.

## License and dataset attribution

The source code in this repository is released under the MIT License.

The BLTE dataset is not redistributed. Dataset ownership, licensing, attribution requirements, and upstream data-processing decisions remain with the original providers.
