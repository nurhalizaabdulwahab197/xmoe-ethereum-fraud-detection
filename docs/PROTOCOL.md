# Shared training and evaluation protocol

The original model/training source is pinned to repository commit
`e660a0fbf20eb24ad5d519404489e996b92eec52`. The snapshot in
`provenance/frozen_pipeline` preserves the same engine/model bytes used in the
completed 100-run study. Its `release_manifest.json` checks those bytes.
`src/runner.py` delegates to that engine, while `src/models` exposes identical
model definitions for review. `FILE_RELOCATION_MAP.json` records the layout change.
Portable data preparation,
publication checks and documentation are outside that training inventory.

## Architecture and objectives

MLP has two 64-wide hidden layers with LayerNorm, ReLU and dropout 0.5. GCN and
GraphSAGE have two 64-wide convolutions and dropout 0.5. GAT uses first-layer
width 32 with four heads, second-layer width 32 with one head, and ELU.

X-MoE combines a conflict-aware multitask expert (feature and internal GraphSAGE
branches, class-specific inner gates) with an independent GraphSAGE expert. The
outer gate receives both expert logits, absolute disagreement and log degree.
Graph contributions are suppressed for isolated nodes. There are no additional
published model families in this study. Input dimensions are 23 and five; all
other architecture settings remain the same. Fitted weights are dataset-specific.

Baselines use training-class-weighted multiclass CE. Class weights are
`(N/n_c) / sum_j(N/n_j)`, computed from training labels. X-MoE uses eight tasks:

1. Weighted CE on mixed logits.
2. Weighted CE on conflict-aware expert logits, weight 0.25.
3. Weighted CE on independent GraphSAGE logits, weight 0.25.
4. Balanced Normal one-vs-rest BCE on conflict-aware logits, weight 1.0.
5. Balanced Phishing one-vs-rest BCE, weight 1.0.
6. Balanced Scamming one-vs-rest BCE, weight 1.0.
7. GraphSAGE Scamming-vs-rest specialist BCE, weight 0.5.
8. Mean mixed Phishing probability on connected Normal training nodes, weight 0.30.

Balanced BCE normalizes inverse-frequency sample weights by their observed sum.
Fixed-order PCGrad projects negative task dot products and averages the projected
task gradients. This is the existing implementation, not a claim of inventing
PCGrad. Ordinary CE/backpropagation is used for baselines. There is no class-bias
calibration, adaptive test threshold, dataset-specific learning rate or new loss
selected on RQ1 test outcomes. The source CLI default penalty 0.0 is overridden
by the shared effective configuration **0.30**.

## Partition, preprocessing and selection

Ascending labeled global node indices are stratified with two `train_test_split`
calls: 30% remainder, then 50% of remainder for test, both using the same seed.
Seeds 100–109 also initialize NumPy/Python/Torch. Each seed's masks are shared
by every model. Context nodes are not supervised or included in scaler fitting.
StandardScaler fits raw float64 features on training rows only and transforms
all nodes to float32. No extra log transform is introduced.

Every run performs 300 Adam steps/epochs, LR 0.001 and weight decay 0.0005.
Validation Macro-F1 selects a checkpoint using a strict greater-than update;
ties retain the earliest checkpoint. Test is scored after selection. The
transductive graph is retained; validation/test target labels are excluded from
the training loss. Directed source edges are not replaced by another graph.

## CPU execution

The recorded PC has an Intel i9-13900, 32 logical processors and 31.7 GiB RAM.
Both adapters use 24 CPU intra-op threads. Actual inter-op threads were recorded
as 24. Fixed seeds do not guarantee identical floating-point reductions across
different thread counts. A 300-epoch source reconstruction at 24 threads and then
the shared memory-recomputed runner each matched all 34 tensors of the original
BLTE seed-100 X-MoE checkpoint exactly, with best epoch 179. At eight threads,
the original and modular/recomputed execution matched each other but followed a
different trajectory. Historical CSVs did not explicitly record thread count;
the checkpoint comparison is the evidence for choosing the reproduced setting.

GAT nonreentrant activation recomputation and 32,768-edge message chunks, and
X-MoE expert recomputation with preserved RNG, reduce memory without changing
model parameters or intended objectives. Equivalence checks have finite test
scope; do not infer universal cross-platform bitwise equality. Both datasets use
the same execution policy, 20 GiB sampled process-tree RSS cap, 4 GiB system RAM
floor and 21,600-second time cap per run. Sampling does not capture every
instantaneous peak. Interrupted attempts are preserved; no mid-epoch resume.

## Reporting and statistics

Accuracy, macro precision/recall/F1, weighted F1, class precision/recall/F1,
Phishing/Scamming AP and all nine confusion cells use the frozen `score` source.
Main tables aggregate all ten seeds with sample SD (`ddof=1`).

The per-dataset primary family compares X-MoE Macro-F1 against four baselines.
The conditional-exact two-sided signed-rank calculation removes exact zeros,
uses average ranks for tied absolute differences, enumerates all sign choices
and applies Holm correction across the four comparisons. Zero differences are
not assigned artificial positive wins. For ten nonzero differences, the
smallest two-sided raw p is 2/1024. RQ1 X-MoE wins against GraphSAGE on nine
seeds and loses on one; this is retained.

Overlapping partitions, very small BLTE class counts, prior test exposure and
cross-source address overlap limit inferential claims. These data do not prove
universal superiority, causal explanations or transfer of a trained checkpoint.
