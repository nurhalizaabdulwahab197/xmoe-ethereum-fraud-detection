# Manuscript Result Audit Status

This file records the current traceability status of the quantitative results used in the manuscript.

## VERIFIED

### 1. Clean baseline comparison
Source: fresh-PC deterministic 10-seed clean runs, seeds 100-109, 300 epochs.

- MLP Macro-F1: 0.371498 +/- 0.008483
- GCN Macro-F1: 0.363689 +/- 0.020485
- GAT Macro-F1: 0.266492 +/- 0.017115
- GraphSAGE Macro-F1: 0.491619 +/- 0.033561
- X-MoE Macro-F1: 0.657494 +/- 0.071187

### 2. Clean minority-class metrics
Fresh-PC 10-seed means are traced for GraphSAGE and X-MoE, including Normal F1, Phishing F1/AP, and Scamming F1/AP.

### 3. Ablation A0-A7
Matched seeds 100-109, 300 epochs. The authoritative paired analysis uses exact two-sided sign-flip tests, Holm correction across seven A0-vs-ablation comparisons, paired bootstrap 95% confidence intervals, and paired Cohen dz.

Only A6 (no Normal-Phishing penalty) is family-wise significant for overall Macro-F1 after Holm correction.

### 4. Connectivity-stratified evaluation
A0 and A4 were rerun on matched seeds 100-109 and verified to reproduce their original journal-grade overall metrics exactly. All evaluated Phishing and Scamming test nodes are connected; isolated test nodes are Normal only.

### 5. STRICT-INDUCTIVE
50/50 model-seed runs completed for MLP, GCN, GAT, GraphSAGE and X-MoE, seeds 100-109, 300 epochs. The user-PC audit checked CSV hashes, predictions, metrics and test identities. Forbidden-edge diagnostics are zero. Direct target-target edges are excluded; indirect two-hop paths through training nodes can remain.

### 6. Clean-to-strict comparison
Matched clean and STRICT-INDUCTIVE results are traced, including Holm-adjusted paired comparisons and the descriptive retention of the X-MoE-vs-GraphSAGE gap.

## PARTIAL / TO RETAIN WITH CAUTION

### 7. Efficiency and training time
Parameter counts are directly produced by the experiment runner and are stable. Full-run training-time summaries used by the manuscript are retained as descriptive hardware-dependent measurements. Training time must not be presented as a universal efficiency benchmark. If the manuscript retains the exact mean times, the corresponding 10-seed efficiency summary should be archived alongside this folder before submission.

## Interpretation rule

Files in `results/tables/` are curated manuscript-facing summaries. They are not substitutes for raw seed-level evidence. Claims in the manuscript should follow the frozen scientific summaries and statistical reports, and should not be strengthened beyond those reports.

## Five-model alignment

All five models are reported under clean and strict. The authoritative strict series is `results/strict/reference/strict_50_runs.csv`. Holm families contain four clean baseline tests, four strict baseline tests, five clean-to-strict tests and four strict minority tests against GraphSAGE. GAT clean-to-strict adjusted p=0.05859375 is not significant at 0.05. MLP Macro-F1 is identical under both protocols for all ten seeds. See `docs/RELEASE_ALIGNMENT.md` for scope and pending publication checks.
