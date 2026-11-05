# X-MoE: matched training on BLTE and ScamSweeper RQ1

This branch evaluates MLP, GCN, GAT, GraphSAGE and X-MoE on two Ethereum address
classification datasets using one CPU environment, one training configuration,
one experiment runner and two dataset adapters. All **100 planned runs** are
complete: two datasets × five models × ten seeds, with 300 epochs per run.

ScamSweeper reference: **ScamSweeper: Detecting Illegal Accounts in Web3 Scams via
Transactions Analysis**, IEEE Transactions on Information Forensics and Security.
Journal DOI: **[10.1109/TIFS.2025.3646027](https://doi.org/10.1109/TIFS.2025.3646027)**.
The [author artifact](https://figshare.com/s/65d00a4c50c9d5188c06) is downloaded
separately; it is not bundled in this repository.

```text
One environment + common_config.json
                 |
  MLP / GCN / GAT / GraphSAGE / X-MoE
                 |
          src/runner.py
                 |
        +--------+--------+
    BLTE adapter      RQ1 adapter
    23 features       5 features
        |                 |
    BLTE results      RQ1 results
```

Each model is trained separately on each dataset. This tests the portability of
the architecture and training method; it does not transfer a BLTE checkpoint to
RQ1. Feature semantics, topology, class balance and source annotation definitions
differ. We do not describe this as an identical-feature experiment.

## Model implementations and repository layout

The five model definitions are visible under **[src/models](src/models)**,
matching the model-directory layout of the BLTE branch:

| Model | Implementation |
|---|---|
| MLP | [src/models/mlp.py](src/models/mlp.py) |
| GCN | [src/models/gcn.py](src/models/gcn.py) |
| GAT | [src/models/gat.py](src/models/gat.py) |
| GraphSAGE | [src/models/graphsage.py](src/models/graphsage.py) |
| X-MoE | [src/models/xmoe.py](src/models/xmoe.py) |

```text
src/models/                  Five readable model implementations
src/runner.py                Public entry point for the shared experiment engine
common_config.json           Shared, frozen training configuration
preparation/                 RQ1 primitive indexing and feature/graph preparation
scripts/                     Input configuration, package and result checks
tests/                       Shared-pipeline regression test entry point
docs/                        Protocol, data provenance and CPU walkthrough
results/reference/           Aggregates and histories for 100 completed runs
provenance/frozen_pipeline/  Byte-preserved engine, adapters and execution snapshot
ENVIRONMENT_CPU.txt          Recorded CPU hardware and runtime
requirements-cpu.txt         Pinned CPU installation requirements
```

To preserve the completed study's source hashes, `src/runner.py` delegates to the
same engine now stored in `provenance/frozen_pipeline`. Its model snapshot is
under `provenance/frozen_pipeline/reference_src/models`. Visible model files are exact byte
copies, not new architectures; `scripts/check_package.py` rejects differences
between the two locations. This layout adds no second training runner and does
not change model computations, configuration or the 100 reported results.

For a fresh CPU reproduction, follow
**[docs/CPU_REPRODUCTION_WALKTHROUGH.md](docs/CPU_REPRODUCTION_WALKTHROUGH.md)**.
It separates package/setup checks, raw ZIP reconstruction and training checks.

## Branches and study boundaries

| Branch | Purpose |
|---|---|
| `main` | Existing BLTE clean/strict evaluation, ablations and supporting results |
| `reproduction` | Separate BLTE reconstruction under released-code and paper-described GPU configurations |
| `demo/streamlit` | Existing demonstration application |
| `two-dataset-comparison` | This matched CPU study on BLTE and RQ1 |

The `reproduction` branch uses a different runtime and source-fidelity
preprocessing. It is not the configuration used for these 100 runs. Updating one
branch does not update its siblings or the deployed demo.

## Results

Test results below are means over seeds 100–109. SD is the **sample** standard
deviation, in percentage points. All models use the same split and input features
within each dataset; MLP does not perform message passing.

### BLTE

| Model | Accuracy (%) | Macro-F1 mean ± SD (%) | Normal F1 (%) | Phishing F1 (%) | Scamming F1 (%) |
|---|---:|---:|---:|---:|---:|
| MLP | 96.72 | 37.15 ± 0.85 | 98.37 | 4.34 | 8.74 |
| GCN | 88.11 | 36.37 ± 2.05 | 93.67 | 1.32 | 14.12 |
| GAT | 59.17 | 26.65 ± 1.71 | 74.30 | 0.36 | 5.28 |
| GraphSAGE | 98.82 | 49.16 ± 3.36 | 99.44 | 10.23 | 37.82 |
| **X-MoE** | **99.69** | **65.75 ± 7.12** | **99.88** | **53.74** | **43.63** |

### ScamSweeper-derived RQ1

| Model | Accuracy (%) | Macro-F1 mean ± SD (%) | Normal F1 (%) | Phishing F1 (%) | Scamming F1 (%) |
|---|---:|---:|---:|---:|---:|
| MLP | 63.91 | 65.77 ± 1.40 | 79.25 | 49.56 | 68.50 |
| GCN | 80.85 | 74.08 ± 1.83 | 65.85 | 67.62 | 88.77 |
| GAT | 72.35 | 58.34 ± 2.09 | 41.45 | 48.99 | 84.57 |
| GraphSAGE | 89.32 | 86.19 ± 0.71 | 81.87 | 83.57 | 93.14 |
| **X-MoE** | **91.71** | **89.23 ± 1.00** | **85.24** | **87.98** | **94.47** |

X-MoE has the largest mean of these metrics on both datasets. Its paired
Macro-F1 advantage over GraphSAGE is 16.59 points on BLTE and 3.04 points on RQ1.
It wins 10/10 and 9/10 paired comparisons, respectively. Conditional-exact
two-sided Wilcoxon signed-rank tests give Holm-adjusted p = 0.0078125 against each
of the four baselines within each dataset's Macro-F1 family. This is not a claim
that every class/metric improves on every seed. Earlier outcomes were inspected;
the second-dataset analysis is exploratory. Repeated account partitions overlap.

Machine-readable summaries, all seed metrics, training histories
and resource reports are in [results/reference](results/reference). Absolute local
paths in published JSON were replaced with logical paths; numerical metrics and
history bytes were preserved. Original prediction/checkpoint hashes are recorded
in [provenance/evidence_inventory.json](provenance/evidence_inventory.json).
Per-node test predictions, checkpoints and raw datasets remain in the local
archive and are not distributed in Git. The public evidence contains aggregate
confusion counts, not individual node labels or prediction probabilities.

## 1. Install Git, Python and Node.js on Windows

Install [Git for Windows](https://git-scm.com/downloads/win),
[Python 3.11.9, Windows 64-bit](https://www.python.org/downloads/release/python-3119/)
and [Node.js](https://nodejs.org/en/download). Include the Python launcher during
installation and add Git/Node.js to PATH. Node.js is needed only for the RQ1
streaming indexer; the recorded preparation check used Node.js 24.22.1.

Open a new **PowerShell** terminal. These are Windows CPU commands; do not use the
WSL/GPU environment from the separate `reproduction` branch.

```powershell
git --version
py -0p
py -3.11 --version
node --version
```

Verify that `py -3.11` selects Python **3.11.9**. The library/runtime versions below
are deliberately pinned rather than replaced with the newest available versions.

## 2. Clone this branch into its own folder

```powershell
git clone --branch two-dataset-comparison https://github.com/nurhalizaabdulwahab197/xmoe-ethereum-fraud-detection.git xmoe-two-dataset
cd xmoe-two-dataset
git branch --show-current
```

Expected branch: `two-dataset-comparison`. Run all subsequent blocks from this
folder. Do not switch a checkout between branches while training is active.

## 3. Create and activate a dedicated CPU environment

```powershell
py -3.11 -m venv .venv-cpu
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv-cpu\Scripts\Activate.ps1
python --version
python -m pip install pip==26.2.2
python -m pip install -r .\requirements-cpu.txt
python -m pip check
```

The requirements contain the full recorded dependency freeze, including the CPU
PyTorch build and its package index. If a pinned wheel is unavailable, retain the
error and inspect the package source; changing a dependency changes the recorded
environment. A new internet installation has not been validated on a second PC.
The installer observed during release validation was pip 26.2.2.

Check the principal versions:

```powershell
python -c "import torch, torch_geometric, numpy, pandas, sklearn, scipy, psutil; print('torch:', torch.__version__); print('PyG:', torch_geometric.__version__); print('NumPy:', numpy.__version__); print('pandas:', pandas.__version__); print('sklearn:', sklearn.__version__); print('SciPy:', scipy.__version__); print('psutil:', psutil.__version__); print('CUDA:', torch.cuda.is_available())"
```

Expected: Torch `2.14.1+cpu`, PyG `2.8.0.post1`, NumPy `2.4.6`, pandas `3.0.6`,
scikit-learn `1.9.1`, SciPy `1.17.1`, psutil `7.2.2`, CUDA `False`.
The runner rejects a mismatched principal environment. A PyG FutureWarning about
`torch.jit.script` is not itself a training failure.

## 4. Inspect the shared protocol and verify the package

```powershell
Get-Content .\common_config.json
python .\scripts\check_package.py
python -B -m unittest discover -s .\tests -p "test_*.py" -v
python -B -m unittest discover -s .\preparation -p "test_*.py" -v
```

The package check needs only Python's standard library; model/preparation tests
need the recorded environment and Node.js. These checks use synthetic fixtures
and published predictions, not new performance training.

| Shared item | Setting for both datasets |
|---|---|
| Models | MLP, GCN, GAT, GraphSAGE, X-MoE; unchanged model source |
| Seeds | 100–109, for partition and initialization |
| Split | Stratified 70/15/15 on labeled accounts; regenerated per seed |
| Scaling | StandardScaler fitted on labeled training rows only; no added `log1p` |
| Training | 300 epochs; Adam, LR 0.001, weight decay 0.0005 |
| Class weights | Normalized inverse training-class frequency |
| Selection | Highest validation Macro-F1; earliest epoch on ties |
| X-MoE | Original objectives and fixed-order PCGrad; Normal–Phishing penalty **0.30** |
| Other X-MoE weights | Expert CE 0.25; expert one-vs-rest 1.0; Scamming specialist 0.5 |
| Calibration | Disabled; no test-selected thresholds or class biases |
| CPU | 24 intra-op threads; recorded inter-op threads also 24 |
| Memory execution | GAT recomputation + 32768-edge chunks; X-MoE expert recomputation |
| Resource guard | 20 GiB process-tree RSS cap; minimum 4 GiB system-available RAM; 6 hours/run |
| Evaluation | Native topology visible transductively; validation/test labels excluded from training loss |

Details, objective definitions and architecture source are in
[docs/PROTOCOL.md](docs/PROTOCOL.md) and [src/models](src/models).

## 5. Obtain and verify BLTE

```powershell
git clone https://github.com/salam-ammari/Graph-Neural-Network-Based-Fraudulent-Address-Classification-in-Ethereum-Network.git upstream-paper
Get-FileHash .\upstream-paper\BLTE_multiclass_graph.pt -Algorithm SHA256
Get-FileHash .\upstream-paper\BLTE_multiclass_v2.csv -Algorithm SHA256
```

Expected SHA-256:

```text
BLTE_multiclass_graph.pt
2859a8a3a815edd02b5cd85d46c00d730b2b66c4e8d047145db2e843c9e38456
BLTE_multiclass_v2.csv
02e376ddd98ba595372f19731fe5e51bbc90bf897b2a52b3c8bb98e786423b11
```

Expected: 73,033 nodes; 23 features; 6,588 directed edges; Normal/Phishing/Scamming
counts 72,865/69/99. Source artifacts remain subject to the provider's terms.

## 6. Prepare ScamSweeper RQ1

Both **RQ1** and **RQ4** archives are required for the fresh reconstruction. Obtain them from the
[ScamSweeper online artifact](https://figshare.com/s/65d00a4c50c9d5188c06).
RQ1 contains the transaction graphs; explicit source account lists are in RQ4.
The downloaded RQ1 ZIP contains three nested ZIPs: `normal_pkl.zip`,
`phish_pkl.zip` and `scams_pkl.zip`. The preparation script opens those archives
automatically and streams the selected pickle files to disk. Do not expect the
pickle files to be top-level members of the outer ZIP.
RQ4 is used only as a label-list source; the evaluated graph is built from RQ1.
RQ2&3 and Cases are not required by this frozen reconstruction workflow. The
historical case/marker review decisions are already frozen in the exclusion policy.
Place the downloaded archives in `data/raw/` and name them `RQ1.zip` and `RQ4.zip`.
The filenames in the download interface may include numbered prefixes; renaming
the ZIP does not change its internal files. Read [docs/DATASETS.md](docs/DATASETS.md)
before using these data. Do not assign unknown context nodes to the Normal class.

```powershell
New-Item -ItemType Directory -Force .\data\raw
# Place the two downloaded archives here before running the next command.
python -B .\preparation\prepare_rq1.py `
  --rq1-zip .\data\raw\RQ1.zip `
  --rq4-zip .\data\raw\RQ4.zip `
  --output-dir .\data\prepared_rq1
```

This verifies pinned pickle bytes and source-list hashes, applies the frozen 13
review exclusions, streams the graph primitives without executing a pickle,
indexes all cohort-incident observation ranges, aggregates duplicate-invariant
features, computes real context-node features, and builds ten training-only
scaled inputs. It does not train or score a model. It deliberately refuses an
existing output directory. Preserve a failed preparation directory for diagnosis.

For an existing unpacked RQ1 directory containing `normal/xlabel_normal.pickle`,
`phish/phish.pickle` and `scams/scams.pickle`, replace `--rq1-zip` with
`--rq1-unpacked <directory>`. Do not provide both options.

Expected prepared graph: **1,322,252 nodes**, **3,856,855 directed pairs**, five
features, **4,913 labeled accounts** (636 Normal, 1,165 Phishing, 3,112 Scams) and
1,317,339 unlabeled context nodes. Each seed has 3,439 training, 737 validation and
737 test accounts. The wrapper checks rebuilt graph hashes; the next step checks
every seed array against the original study.

Preparation historically took roughly an hour for the aggregation/context stages,
plus indexing, extraction and verification. Allow at least 60 GB free disk and a
32 GB RAM system. The combined portable raw-archive wrapper is new packaging:
its components have fixture checks, while a fresh complete raw rebuild must still
be validated before claiming end-to-end archive reproduction. The existing
prepared-array route below is separately verified.

## 7. Verify every prepared input and create local paths

```powershell
python -B .\scripts\configure_datasets.py `
  --blte .\upstream-paper `
  --rq1-graph .\data\prepared_rq1\graph `
  --rq1-inputs .\data\prepared_rq1\seed_inputs
```

For previously prepared study arrays, supply their graph/input directories instead.
This command verifies graph hashes, all ten split/normalized-array hashes and
independently recomputes training-only scaling. It creates `datasets.local.json`
and a portable local input lock. These generated path files are excluded from Git;
they do not alter the shared training source/configuration. No test metrics are
calculated. Do not copy historical absolute filesystem paths from another PC.

```powershell
python -B .\src\runner.py --dataset blte --mode verify --registry .\datasets.local.json --seeds 100,101,102,103,104,105,106,107,108,109
python -B .\src\runner.py --dataset rq1 --mode verify --registry .\datasets.local.json --seeds 100,101,102,103,104,105,106,107,108,109
```

## 8. Check execution and memory before a long run

Reference hardware: Windows, Intel Core i9-13900, 32 logical processors and
31.7 GiB physical RAM. CUDA is not used. Check your available memory:

```powershell
python -c "import psutil; print('Available GiB:', round(psutil.virtual_memory().available/1024**3,2))"
```

Aim for at least 12 GiB available before starting RQ1. Run one experiment process
at a time. GAT/X-MoE recomputation keeps RQ1 peak memory around 3–7 GiB on the
reference PC; it increases compute time. A short probe does not guarantee memory
availability throughout a long run.

```powershell
python -B .\src\runner.py --dataset blte --mode validate --check-epochs 5 --seeds 100 --threads 24 --registry .\datasets.local.json --output-dir .\checks\blte_validation5
python -B .\src\runner.py --dataset rq1 --mode probe --seeds 100 --threads 24 --registry .\datasets.local.json --output-dir .\checks\rq1_memory_probe
```

`validate` trains briefly and checks validation/checkpoint reloading without test
scoring. `probe` performs one optimizer step without validation/test evaluation.
These are technical checks, excluded from the 100-run final comparison.

## 9. Run all BLTE experiments first

```powershell
python -B .\src\runner.py `
  --dataset blte `
  --mode train `
  --registry .\datasets.local.json `
  --models mlp,gcn,gat,graphsage,x_moe_gnn `
  --seeds 100,101,102,103,104,105,106,107,108,109 `
  --threads 24 `
  --output-dir .\results\new_runs\blte
```

## 10. Run all RQ1 experiments

```powershell
python -B .\src\runner.py `
  --dataset rq1 `
  --mode train `
  --registry .\datasets.local.json `
  --models mlp,gcn,gat,graphsage,x_moe_gnn `
  --seeds 100,101,102,103,104,105,106,107,108,109 `
  --threads 24 `
  --output-dir .\results\new_runs\rq1
```

Do not launch these two blocks concurrently. Keep the training terminal open and
disable automatic sleep while running. Based on observed RQ1 seed 100, five models
took about 5 h 40 min; ten seeds would be roughly 57 h at that rate. This is an
estimate, not a timeout setting or guarantee. X-MoE is the slowest model.

Every final run selects its best validation checkpoint, reloads it, then evaluates
test once. It writes `result.json`, `resource_probe.json`, `best_checkpoint.pt`,
`test_predictions.csv`, `training_history.csv` and progress/console reports.
Each dataset root gets `summary.json` and `attempt_inventory.json`. Completion
requires **50/50**, not merely the final model's `passed` message.

## 11. Continue after interruption

Use exactly the same command and output directory. Completed runs are reused only
after configuration, registry and evidence-hash verification. Partial attempts
are preserved in numbered attempt directories. A failed seed restarts at epoch 1;
optimizer/RNG mid-epoch resume is not implemented. Do not change configuration or
registry paths midway through an execution series.

Closing an editor can leave a worker alive. Before restarting, inspect remaining
Python processes and their command lines:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -eq "python.exe" } |
  Select-Object ProcessId, ParentProcessId, CommandLine |
  Format-List
```

Confirm whether an old worker for the same seed is still running before launching
another. Stop only the identified abandoned experiment process, after review.
Do not terminate every Python process or lower the 4 GiB safety floor to force a
run. `stopped_low_available_ram` refers to system availability, not model quality.

## 12. Audit results and compare with the published evidence

```powershell
python .\scripts\audit_results.py --dataset blte --results-dir .\results\new_runs\blte
python .\scripts\audit_results.py --dataset rq1 --results-dir .\results\new_runs\rq1
python .\scripts\check_package.py
```

The audit reads saved predictions and histories; it does not train or run model
inference. It checks all planned seeds, unique completed attempts, hashes,
confusion counts, accuracy and per-class/Macro-F1, the earliest best validation
epoch, and shared settings. Do not choose favorable repeat attempts or seeds.

Exact floating-point equality on different hardware/backends is not guaranteed.
CPU thread count mattered in the original reproduction audit: the explicit
24-thread execution reproduced the original BLTE seed-100 X-MoE weights exactly;
an eight-thread trajectory differed. This is why 24 is fixed here. Thread count
was restored to reproduce pre-existing evidence, not selected to optimize RQ1
test scores. See [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Limitations and attribution

The RQ1 cohort and features describe available cohort-incident observations, not
complete Ethereum histories. Temporal-unit verification beyond the implemented
consistency/plausibility checks remains limited. No amount, gas or transaction
frequency feature was fabricated. Context labels remain unknown. RQ1 features
are graph-derived, so its MLP is not graph-information-free.

BLTE/RQ1 share some addresses and have known annotation disagreements. This is a
second dataset evaluation, not an address-disjoint, temporal, cross-chain or
checkpoint-transfer test. All source-native definitions and exclusions are
retained and disclosed. The transductive setting uses the full observed topology.
Routing diagnostics are model diagnostics, not causal proof.

Code is under [MIT](LICENSE). Dataset rights are separate: obtain upstream
artifacts from their providers and follow their attribution/redistribution terms.
See [docs/DATASETS.md](docs/DATASETS.md) for provenance and expected hashes.
See [docs/VALIDATION_STATUS.md](docs/VALIDATION_STATUS.md) for the exact release
checks completed and the remaining fresh-install/raw-rebuild validation scope.
