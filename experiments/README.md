# Experiment Runners

This directory contains the manuscript-specific validation runners that complement the main reproduction entry point in `src/reproduce.py`.

## Entry points

- `ablation.py` — A0–A7 X-MoE ablation study, including connectivity-stratified evaluation.
- `strict_inductive.py` — STRICT-INDUCTIVE evaluation with restricted topology visibility.

The main five-model clean-protocol comparison is run through `src/reproduce.py` using:

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

Machine-readable sensitivity results are available in `results/lambda_sensitivity/summary.csv`.

## Dataset scope

The BLTE dataset is not redistributed by this repository. The study uses the released processed BLTE artifacts and does not independently reconstruct the raw Ethereum collection or label-generation pipeline.

See the root `README.md` and `REPRODUCIBILITY.md` for dataset hashes, environment details, reproduction commands, and manuscript result tables.
