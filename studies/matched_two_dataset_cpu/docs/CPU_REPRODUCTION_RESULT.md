# Selected CPU reproduction outcome

The fresh-checkout CPU reproduction passed at seed 100 for all five models on
both datasets, using the shared 300-epoch configuration. This verification was
performed on the same reference Windows computer, not a second independent PC.

The procedure began with a separate Git checkout and Python 3.11.9 environment.
The pinned principal versions and package inventory passed, as did the synthetic
pipeline/preparation checks. The actual author RQ1 outer ZIP contains three nested
ZIPs; the corrected wrapper streamed their pickles to disk and built the cohort,
features, full retained graph, ten partitions and training-only standardized data.
All four graph arrays and all 40 per-seed arrays matched the original hashes.
The configuration checker independently recomputed normalized values, masks and
scaler parameters. BLTE source input hashes also matched.

Node.js 26.13.0 was used for this fresh indexing, compared with 24.22.1 in the
earlier reference preparation check. Exact resulting array hashes establish
input equality for this tested case. Node.js does not execute model training.

Five-epoch validation-only checks passed on both datasets, including exact
checkpoint-reload validation logits and exclusion of the test phase. The final
selected reproduction then ran each model for 300 epochs at seed 100.

| Dataset | Model | Reference test Macro-F1 (%) | Fresh test Macro-F1 (%) | 25 metrics / prediction bytes / best epoch |
|---|---|---:|---:|---|
| BLTE | MLP | 36.943862 | 36.943862 | Exact match |
| BLTE | GCN | 34.660470 | 34.660470 | Exact match |
| BLTE | GAT | 25.766981 | 25.766981 | Exact match |
| BLTE | GraphSAGE | 50.858075 | 50.858075 | Exact match |
| BLTE | X-MoE | 69.894954 | 69.894954 | Exact match |
| RQ1 | MLP | 64.619232 | 64.619232 | Exact match |
| RQ1 | GCN | 74.405745 | 74.405745 | Exact match |
| RQ1 | GAT | 57.477650 | 57.477650 | Exact match |
| RQ1 | GraphSAGE | 87.067182 | 87.067182 | Exact match |
| RQ1 | X-MoE | 88.616788 | 88.616788 | Exact match |

The 25 checked fields comprise accuracy, macro precision/recall/F1, weighted F1,
Phishing/Scamming AP, nine per-class precision/recall/F1 fields and nine confusion
cells. Matching prediction hashes include the saved per-node probabilities, not
only rounded summary scores. These prediction files and checkpoints remain local.
Each fresh run's checkpoint/history/prediction hash was checked for integrity.
Serialized checkpoint-file equality with the reference is not asserted because
saved metadata includes run-specific paths and timings.

For RQ1 X-MoE, the reproduced seed-100 test accuracy was 91.316147%, Normal F1
84.422111%, Phishing F1 87.031700% and Scamming F1 94.396552%. Its selected epoch
was 290. Fresh-run elapsed time was about 3 h 30 min, with a sampled process-tree
peak of 6.46 GiB; elapsed times need not match the reference run.

These are **single-seed verification values**, not replacements for the original
ten-seed means. The main comparison remains 100 primary runs. The fresh check
adds ten verification runs, not an independent new 100-run experiment. The other
nine seeds and independent hardware have not been tested in this fresh execution.
Do not infer universal bitwise portability from this same-PC result.

Machine-readable aggregate verification:
[fresh_cpu_seed100_reproduction.json](../provenance/fresh_cpu_seed100_reproduction.json).
The experiment runner/configuration, label policies and original results were
not tuned or replaced during this verification.
