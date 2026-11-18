# Experiment inventory by dataset and evidence scope

## Names

`blte` means the 73,033-address, 23-feature BLTE study. `scamsweeper` means the separately prepared 1,322,252-node / 4,913-labelled-account graph; captured code/result keys retain `rq1`. Do not confuse it with the separate window5/session preprocessing artifact. No numeric result is renamed as a different dataset.

| Experiment / analysis | BLTE | ScamSweeper / RQ1 |
|---|---|---|
| Clean five-model comparison | 50 seed/model rows; historical CPU reference | 50 fresh CUDA runs |
| Strict five-model evaluation | 50 rows; restricted visibility | 50 fresh CUDA runs; unlabelled context excluded |
| X-MoE component ablation A0–A7 | 80 rows; original scores and score-reproducing supplementary evidence attributed | 80 fresh CUDA runs |
| Normal–Phishing lambda sensitivity | 50 coefficient/seed rows; completed metrics use 30 fresh + 20 reused A0/A6 rows | 50 fresh CUDA runs, including separately executed equivalent controls |
| CPU/GPU comparison | Historical four-GNN series, 40 rows per device comparison side; separately documented | NOT completed as a controlled matched CPU/GPU study; 230 CUDA runs alone do not establish parity |
| Connected/isolated performance | Analysis of runs; see connectivity tables and metric audit | Analysis in each run; no clean degree-zero nodes, so clean isolate metrics are undefined |
| Per-class precision, recall, F1, AP, confusion matrices | Completed metric audit also includes ROC-AUC, balanced accuracy, MCC | Recorded per-seed metrics in completed_runs.csv; do not claim unrecorded ROC-AUC/MCC |
| Routing / specialist diagnostics | Descriptive analyses in the historical guide/evidence | Descriptive per-run gate/weight fields where available |
| Timing / parameter counts / resource use | Historical environment-specific efficiency evidence | Training time/parameter count in results; worker RAM logs in original snapshot; no cross-hardware speed claim |
| Mean/SD, paired tests, Holm correction | Dataset-specific families and archived provenance | Dataset-specific families, original statistics preserved |
| Convergence / validation selection | Runner selection rules; archived evidence availability varies | 300-epoch histories archived; validation-selected checkpoint, earliest tie |
| Smoke / preflight | Implementation/input checks, excluded from scientific tables | 10-seed preflight and 23 one-epoch smoke runs, excluded from 230 full runs |
| Artifact / fixed-mask protocol | Supported historical runner/pilots; not the clean/strict tables | Not part of the locked 230-run matrix |
| Reconstruction + CE control + PCGrad ON/OFF | Separate reproduction branch (BLTE section): 120 primary rows, 40 PCGrad rows, six separate fixed-mask pilots; public wrapper GPU validation caveat retained | Not conducted in this locked matrix |

Counts describe each evidence series, not a grand total of independent training jobs. Reused controls, supplementary metric reruns, historical GPU series and overlapping analyses must not be added indiscriminately.

## Paths

| Dataset | Runnable entry points | Tables / run records | Interpretation |
|---|---|---|---|
| BLTE | [experiments/blte](../experiments/blte/) | [results/blte](../results/blte/) | [docs/blte](blte/) |
| ScamSweeper/RQ1 | [experiments/scamsweeper](../experiments/scamsweeper/) | [results/scamsweeper](../results/scamsweeper/) | [docs/scamsweeper](scamsweeper/) |

Separate reconstruction/PCGrad branch: [reproduction](https://github.com/nurhalizaabdulwahab197/xmoe-ethereum-fraud-detection/tree/reproduction). The earlier matched CPU tree is retained under studies/matched_two_dataset_cpu; consult studies/MATCHED_CPU_GUIDE.md for its adapted commands.

## Shared A0–A7 definitions

A0 full; A1 conflict-aware expert only; A2 GraphSAGE-expert configuration with applicable objectives; A3 shared routing in inner and outer mixtures; A4 removes both connectivity masks; A5 removes Scamming-specialist auxiliary loss; A6 removes Normal–Phishing penalty; A7 fixed 50:50 outer mixture on connected nodes, retaining isolate masking.

Definitions align; empirical conclusions differ. BLTE ablation and Sweeper ablation must have separate tables. A2 is not the standalone GraphSAGE baseline. PCGrad ON/OFF is not A0–A7.


## Branch roles and source-study reproduction status

These experiment records now live on benchmark. Main is the general project entry; reproduction is the source-study workflow. The project owner confirms BLTE reproduction completed, Sweeper source-study reproduction in progress, and BERT4ETH reproduction completed externally with repository import pending. The completed Sweeper 230-run benchmark matrix is not a substitute for the pending source-study reproduction. See BRANCH_GUIDE.md.
