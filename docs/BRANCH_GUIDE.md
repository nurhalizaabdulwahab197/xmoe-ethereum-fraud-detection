# Branch organization and study boundaries

Use main to reproduce the principal clean/strict/component-ablation/sensitivity protocols for BOTH datasets. Dataset names belong in directories; branches here distinguish separate scientific studies, not whether a dataset happens to have an ablation.

| Recommended branch | Original compatibility branch | Pinned study head at organization | Contents |
|---|---|---|---|
| main | main | See the current main commit | Original BLTE main-study evidence plus ScamSweeper/RQ1 230-run CUDA suite; both full reproduction guides |
| study/blte-reconstruction | reproduction | 899f3dcc3581b11a61aaf5061221890f8e6e0455 | BLTE paper-described/released-code reconstruction, 120 primary runs, CE controls, 40 PCGrad ON/OFF runs and separately documented fixed-mask pilots |
| study/matched-two-dataset-cpu | two-dataset-comparison | 4b4cb987b079d99fcac01a3c34c57640b50df445 | Earlier common CPU protocol: BLTE 50 and RQ1 50 clean runs; separate ten-run seed-100 verification, original input preparation and manifests |

The descriptive study branches initially point to exactly the original study commits. Original branch names, histories, manifests and result bytes are retained. These names are not extra experiments. Future work should name its scientific scope and record the exact commit; branches can diverge later, so a historical alias is not a promise of automatic synchronization.

## What main contains

| Dataset | Clean | Strict | A0–A7 | Lambda sensitivity | Reading guide |
|---|---|---|---|---|---|
| BLTE | Five models, 50 rows | Five models, 50 rows | 80 rows | 50 rows, with reused-control provenance in metric completion | README Part I |
| ScamSweeper/RQ1 | Five models, 50 fresh CUDA runs | Five models, 50 fresh CUDA runs | 80 fresh CUDA runs | 50 fresh CUDA runs | README Part II |

BLTE code: experiments/blte (delegates to the unchanged original implementations). Sweeper code: experiments/scamsweeper (launches its captured unified suite). Results and docs have matching dataset directories.

The earlier two-dataset CPU study is scientifically useful and is not described as invalid or merely an unfinished pilot. It has a different environment and study scope from the current GPU suite. Do not silently substitute its scores for the GPU extension. Similarly, PCGrad ON/OFF under reconstructed BLTE conditions is not the A0–A7 component ablation.

## Checkout commands

Use distinct folders; do not switch branches inside a running experiment checkout.

```bash
git clone --branch main --single-branch https://github.com/nurhalizaabdulwahab197/xmoe-ethereum-fraud-detection.git xmoe-main
```

```bash
git clone --branch study/blte-reconstruction --single-branch https://github.com/nurhalizaabdulwahab197/xmoe-ethereum-fraud-detection.git xmoe-blte-reconstruction
```

```bash
git clone --branch study/matched-two-dataset-cpu --single-branch https://github.com/nurhalizaabdulwahab197/xmoe-ethereum-fraud-detection.git xmoe-matched-cpu
```

Follow each study's own environment/input commands. The historical study READMEs may still name their compatibility branch; those names remain available and refer to the original study.

## Run-family organization versus reproduction completeness

The BLTE CLI runs each model/variant/coefficient separately, or all eight variants in one ablation invocation. The locked Sweeper CLI schedules all 230 full rows after preflight/smoke. It has no family-only selector. Both include complete five-model clean reproduction. A command interface difference is not missing evidence.

Connectivity is computed during evaluation, not a separate training phase. The BLTE ablation raw CSV contains its connectivity fields; the primary summary CSV does not aggregate every subgroup field. `scripts/inspect_blte_outputs.py --view connectivity` provides a read-only view of new raw outputs. Main's historical reference table is a separate artifact and is not automatically overwritten.

No model retraining, original-source edits or frozen-result changes are performed by this branch organization.
