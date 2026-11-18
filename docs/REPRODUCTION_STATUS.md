# X-MoE — Source-Study Reproduction

This branch keeps source-study reproduction separate from our benchmark evaluation. It has three dataset folders with explicit evidence status. The complete BLTE instructions are retained below; Sweeper and BERT4ETH have honest documentation scaffolds until their remaining work/import is ready.

## 1. Choose the dataset and check status

| Folder | Reproduction status | Repository evidence / next action |
|---|---|---|
| blte/ | Completed archived reproduction evidence | Full source, saved partitions, configuration comparison, result tables and detailed commands below; retain original public-wrapper validation caveats |
| sweeper/ | IN PROGRESS | Owner is running source-study reproduction; import verified code, configuration and results when ready |
| bert4eth/ | Completed according to owner; repository import PENDING | Create the structure now; upload exact implementation, environment, input/split details and verified results later |

The 100-run matched CPU benchmark and the 230-run Sweeper GPU suite are retained on [benchmark](https://github.com/nurhalizaabdulwahab197/xmoe-ethereum-fraud-detection/tree/benchmark). They do not change Sweeper source-study reproduction to complete. BERT4ETH source reproduction success also does not establish completion of an X-MoE sequence-encoder extension.

## 2. Understand the folder layout

| Path | Role |
|---|---|
| README.md | Complete reproduction entry and BLTE step-by-step guide |
| blte/ | Byte-preserved original BLTE reproduction checkout |
| sweeper/README.md | In-progress status and import requirements |
| bert4eth/README.md | Owner-reported completed reproduction; import scaffold |
| docs/BRANCH_GUIDE.md | Project's three branch roles |
| docs/REPRODUCTION_STATUS.md | Evidence status and acceptance checklist |

The original BLTE checkout now sits one level below the branch root so its internal ROOT-relative imports and package manifests remain unchanged. Run all scientific BLTE commands from inside blte. Its original README is retained as archived evidence; the integrated guide below adapts its checkout navigation.

## 3. Clone the reproduction branch

```bash
git clone --branch reproduction --single-branch https://github.com/nurhalizaabdulwahab197/xmoe-ethereum-fraud-detection.git xmoe-reproduction
cd xmoe-reproduction
```

Before following the BLTE commands:

```bash
cd blte
git rev-parse HEAD
```

Use that folder's environment files and scripts. Do not mix its GPU environment with the separate benchmark environment merely because both use BLTE data.



## Completion rule

Status must distinguish owner-reported success, imported code, verified evidence and tested public commands. BLTE retains its existing caveats; Sweeper remains in progress; BERT4ETH remains import-pending until the successful artifacts are inspected. Benchmark outcomes are not substituted for source-study reproduction.
