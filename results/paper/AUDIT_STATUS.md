# Manuscript Result Audit Status

This file records the current traceability status of the quantitative results used in the manuscript.

## VERIFIED

### 1. Clean baseline comparison
Source: fresh-PC deterministic 10-seed clean runs, seeds 100-109, 300 epochs.

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

### 5. STRICT-INDUCTIVE-V1
40/40 model-seed runs completed for GCN, GAT, GraphSAGE, and X-MoE, seeds 100-109, 300 epochs. Frozen protocol audit confirms zero forbidden graph leakage and no target-target message passing.

### 6. Clean-to-strict comparison
Matched clean and STRICT-INDUCTIVE-V1 results are traced, including Holm-adjusted paired comparisons and the descriptive retention of the X-MoE-vs-GraphSAGE gap.

## PARTIAL / TO RETAIN WITH CAUTION

### 7. Efficiency and training time
Parameter counts are directly produced by the experiment runner and are stable. Full-run training-time summaries used by the manuscript are retained as descriptive hardware-dependent measurements. Training time must not be presented as a universal efficiency benchmark. If the manuscript retains the exact mean times, the corresponding 10-seed efficiency summary should be archived alongside this folder before submission.

## Interpretation rule

Files in `results/paper/` are curated manuscript-facing summaries. They are not substitutes for raw seed-level evidence. Claims in the manuscript should follow the frozen scientific summaries and statistical reports, and should not be strengthened beyond those reports.
