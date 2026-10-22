# X-MoE Ablation Experiments

This directory contains the journal-validation ablation protocol for X-MoE
for multi-class Ethereum fraud detection.

## Purpose

The ablation study evaluates whether individual components of X-MoE
contribute to performance under severe class imbalance and sparse graph
connectivity.

The primary task contains three classes:

- Normal
- Phishing
- Scamming

The ablation experiment is separate from the public four-model
reproduction runner in `src/reproduce.py`.

## Ablation Variants

| ID | Variant | Experimental question |
|---|---|---|
| A0 | Full X-MoE | Reference architecture containing all X-MoE components |
| A1 | Conflict-aware expert only | Is the external GraphSAGE expert necessary? |
| A2 | GraphSAGE expert only | Is the conflict-aware expert necessary? |
| A3 | Global inner + outer routing | Does class-specific routing add value beyond shared routing? |
| A4 | No isolate masking | Does explicit suppression of graph evidence for isolated nodes add value? |
| A5 | No scam-specialist loss | Does the Scamming-specific specialist objective contribute? |
| A6 | No normal-phishing penalty | Does the connected-Normal Phishing penalty contribute? |
| A7 | Fixed 50:50 outer mixture | Does learned adaptive expert routing outperform fixed averaging? |

## Controlled Comparison Principle

Each ablation is designed so that only the component under investigation
changes relative to A0 whenever possible.

A3 preserves the A0 modules and initialization while replacing
class-specific inner and outer routing behaviour with shared routing.

A4 preserves the A0 modules and initialization while removing isolate
masking from the inner and outer routing stages.

This avoids introducing an initialization confound into the A0-vs-A4
comparison.

## Loss-Fairness Rule

For A1 and A2, the remaining expert produces the final prediction.

Therefore, its logits already receive the primary multiclass
cross-entropy loss.

The identical auxiliary multiclass cross-entropy is not applied a second
time to the same logits.

Applicable expert-specific auxiliary objectives remain active.

This prevents accidental duplication of the primary classification
objective in the single-expert ablations.

## Experimental Protocol

The journal-validation CPU experiment uses:

- protocol: clean
- seeds: 100-109
- epochs: 300
- deterministic execution
- same BLTE processed graph topology
- stratified 70/15/15 train/validation/test split regenerated per seed
- feature standardization fitted using training nodes only

Loss settings:

- expert auxiliary loss weight: 0.25
- scam-specialist loss weight: 0.50
- expert task loss weight: 1.00
- normal-phishing penalty: 0.30

The A5 and A6 variants disable their corresponding loss terms.

## Smoke Validation

Before the full experiment, run:

- seed: 100
- epochs: 10
- all A0-A7 variants

Smoke tests validate implementation only.

They must not be reported as final scientific performance evidence.

The A0 smoke regression reference is:

- Accuracy: 0.9268827019625742
- Macro-F1: 0.390384321709097

A0 should reproduce this deterministic reference under the verified CPU
environment.

## Full Journal Experiment

The full experiment evaluates all eight variants using ten seeds:

100, 101, 102, 103, 104, 105, 106, 107, 108, 109

with 300 epochs per seed.

This produces 80 variant-seed runs.

Primary metric:

- Macro-F1

Supporting metrics include:

- Accuracy
- Macro Precision
- Macro Recall
- Weighted F1
- Normal Precision / Recall / F1
- Phishing Precision / Recall / F1 / AP
- Scamming Precision / Recall / F1 / AP
- training time
- parameter count

## Why Macro-F1 Is Primary

The BLTE multiclass dataset is strongly imbalanced.

Accuracy can therefore be dominated by the Normal class.

Macro-F1 gives equal class-level contribution to the final summary and
is used as the primary comparative metric.

## Interpretation

Ablation results must be analysed across seeds rather than from one run.

A component should not be claimed as beneficial merely because one seed
or one short smoke experiment performs better.

The final interpretation will use:

1. mean and standard deviation;
2. paired seed-wise comparisons against A0;
3. per-class performance;
4. connectivity-stratified analysis;
5. effect size and statistical testing;
6. computational efficiency;
7. independent CPU/GPU reproducibility.

## Reproduction Levels

Researchers may use this repository at three levels.

### Level 1 - Reproduce the proposed model

Run only X-MoE using the public reproduction runner.

### Level 2 - Reproduce the baseline comparison

Run GCN, GAT, GraphSAGE, and X-MoE.

### Level 3 - Reproduce the complete component validation

Run A0-A7 using this ablation runner under the documented protocol.

## Independent Reproduction

A later independent experiment should be performed from a fresh clone or
fresh directory and, where available, a different hardware environment.

The same dataset hashes, seeds, split protocol, hyperparameters, metrics,
and variant definitions should be used.

Small CPU/GPU numerical differences may occur, but the principal
scientific conclusions should remain stable.

## Important

The BLTE dataset is not redistributed by this repository.

Researchers must obtain the processed artifacts from the original
upstream project and verify the documented dataset hashes before running
the experiments.
