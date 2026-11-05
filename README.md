# X-MoE: matched training on BLTE and ScamSweeper RQ1

This branch evaluates MLP, GCN, GAT, GraphSAGE and X-MoE on two Ethereum address
classification datasets using one CPU environment, one training configuration,
one experiment runner and two dataset adapters. All **100 planned runs** are
complete: two datasets × five models × ten seeds, with 300 epochs per run.

## Completed study and reproduction checks

| Existing work | Scope | Status |
|---|---|---|
| Main BLTE experiment | Five models, seeds 100–109, 300 epochs | **50/50 complete** |
| Main RQ1 experiment | Five models, seeds 100–109, 300 epochs | **50/50 complete** |
| Fresh-checkout reproduction check | Five models on both datasets, seed 100, 300 epochs | **10/10 complete; exact reference match** |

The main scientific comparison uses **all 100 original runs**. X-MoE has the
highest mean accuracy, Macro-F1 and each class's F1 among the five evaluated
models on both datasets. The extra seed-100 runs verify that the published
workflow can reproduce its reference outcomes on the same PC in a fresh
environment. They do not replace the ten-seed results or add independent data.

**Selected CPU reproduction is also verified:** a fresh checkout/environment on
the reference PC rebuilt RQ1 from the original nested ZIP and reproduced all five
models at seed 100 for both datasets (ten additional verification runs, 300 epochs
each). All 25 predictive metrics per run, prediction-file bytes and best epochs
matched the original seed-100 evidence exactly. This is not a second ten-seed
study or independent-hardware validation. See
[docs/CPU_REPRODUCTION_RESULT.md](docs/CPU_REPRODUCTION_RESULT.md).

**The commands below are for a new user's own execution.** Steps 1–8 prepare/check
their environment and inputs. Steps 9–11 reproduce seed 100. Step 12 is optional
for readers who want to reproduce the full ten-seed study; their seed 100 is
already counted. **The author has completed both the main study and the selected
reproduction check; no further author training is required by this guide.**

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

## Completed main study: results across seeds 100–109

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

## Reproduction guide for new users

The following numbered steps describe how a reader creates their own outputs.
They do not describe unfinished work in the completed study above. Keep the
published `results/reference` evidence separate from the new output directories.

## 1. Install Git, Python and Node.js on Windows

Install [Git for Windows](https://git-scm.com/downloads/win),
[Python 3.11.9, Windows 64-bit](https://www.python.org/downloads/release/python-3119/)
and [Node.js](https://nodejs.org/en/download). Include the Python launcher during
installation and add Git/Node.js to PATH. Node.js is needed only for RQ1 input
preparation, not PyTorch training. Earlier checks used Node.js 24.22.1; the
successful fresh ZIP-to-input reproduction used **26.13.0** and produced the same
array hashes. Record your version; the Python/numerical environment below is pinned.

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

If `node` is not recognized, this is a command/PATH issue, not a project-folder
issue. Check the normal Windows installation before installing again:

```powershell
Test-Path "C:\Program Files\nodejs\node.exe"
```

If this returns `True`, enable it in the **current terminal** and verify:

```powershell
$env:Path = "C:\Program Files\nodejs;" + $env:Path
node --version
```

Otherwise install Node.js for Windows x64 from the official site with **Add to
PATH**, and open a new PowerShell terminal. An editor's terminal may require the
editor to be restarted after saving work. Do not rely on another application's
private runtime installation. After reopening a terminal later, return to this
checkout and reactivate `.venv-cpu` before using Python.

## 2. Clone this branch into its own folder

```powershell
git clone --branch two-dataset-comparison https://github.com/nurhalizaabdulwahab197/xmoe-ethereum-fraud-detection.git xmoe-two-dataset
cd xmoe-two-dataset
git branch --show-current
```

Expected branch: `two-dataset-comparison`. Run all subsequent blocks from this
folder. If the target directory already exists, preserve it and choose a new
name rather than overwriting it. Do not switch branches or pull code changes
while preparation/training is active. Record the exact revision:

```powershell
git rev-parse HEAD
```

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
Expected `pip check` result: `No broken requirements found`. A notice that a newer
pip exists does not require changing the pinned installer. Retain an installation
error and review it before continuing; do not silently substitute package versions.

## 4. Inspect the shared protocol and verify the package

```powershell
Get-Content .\common_config.json
python .\scripts\check_package.py
python -B -m unittest discover -s .\tests -p "test_*.py" -v
python -B -m unittest discover -s .\preparation -p "test_*.py" -v
```

The package check needs only Python's standard library; model/preparation tests
need the recorded environment and Node.js. These checks use synthetic fixtures
and published aggregate evidence, not new performance training. Expected output:
`PASS: publication inventory and all 100 saved runs`, **16** shared-pipeline tests
`OK`, and **12** preparation tests `OK`. The package check's
`training_performed: false` / `model_inference_performed: false` are intentional.

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
Get-Item .\data\raw\RQ1.zip, .\data\raw\RQ4.zip |
  Select-Object Name, Length
Get-FileHash .\data\raw\RQ1.zip, .\data\raw\RQ4.zip -Algorithm SHA256
```

Both files must exist and be nonempty. The reference downloads were about
4.46 GB (RQ1 ZIP) and 0.40 GB (RQ4 ZIP). ZIP names/compression metadata can vary;
the enforced checks are the raw pickle and label-list content hashes. Keep the
ZIP hash output as your download record. Next, rebuild the inputs:

```powershell
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
For a new execution, the `Seed 100` through `Seed 109` lines here describe that
reader's prepared split/scaling inputs. Their model-training commands follow in
steps 9–10 and, optionally, step 12. The completed author study includes **both**
data preparation and model training for all ten seeds, with results reported above.
If you choose a different preparation output directory, use that same directory
in both RQ1 path arguments in step 7.

For an existing unpacked RQ1 directory containing `normal/xlabel_normal.pickle`,
`phish/phish.pickle` and `scams/scams.pickle`, replace `--rq1-zip` with
`--rq1-unpacked <directory>`. Do not provide both options.

Expected prepared graph: **1,322,252 nodes**, **3,856,855 directed pairs**, five
features, **4,913 labeled accounts** (636 Normal, 1,165 Phishing, 3,112 Scams) and
1,317,339 unlabeled context nodes. Each seed has 3,439 training, 737 validation and
737 test accounts. The wrapper checks rebuilt graph hashes; the next step checks
every seed array against the original study.

Allow at least 60 GB free disk and a 32 GB RAM system. On 5 November 2026, a fresh
CPU checkout rebuilt the RQ1 outer ZIP through graph/features and all ten seed
inputs. All four graph-array hashes and 40 seed-array hashes matched the original
study; training-only scaling and masks were independently recomputed and equal.
This input reconstruction used Python 3.11.9 and Node.js 26.13.0 on the reference
PC. Selected 300-epoch reproduction at seed 100 has subsequently passed for all
five models on both datasets. Independent-computer validation and repeating all
ten seeds in the fresh checkout have not been performed. The input record is
[provenance/fresh_cpu_input_validation.json](provenance/fresh_cpu_input_validation.json).

### Preparation time and progress

Users starting from the original archives need to perform this preparation once.
The same verified graph/features and ten prepared splits are then reused by all
five models. Do not repeat ZIP extraction and graph construction for each model,
seed, or training restart. A deliberate fresh-input reproduction is a separate
reason to build into a new output directory.

Observed timings from the earlier preparation on the reference i9-13900/31.7 GiB
Windows PC are useful planning evidence, not a measured end-to-end time for a
fresh installation:

| Phase | Work / progress indication | Recorded time or scope |
|---|---|---|
| Nested extraction | `Opening nested archive: ...`; streams three pickles totaling about 10.7 GB | Varies with disk, compression and antivirus; no separately recorded phase time |
| Source verification | `PASS raw graph: ...`; hashes each entire pickle and verifies label lists | Full-file I/O; no separate benchmark |
| Stage 1/4 | Indexes cohort-incident source observations | Progress reports millions of source edge records |
| Stage 2/4 | Cohort feature aggregation over 1,695,923 selected range tasks | About **22 minutes** previously (1,302.8 seconds) |
| Stage 3/4 | Builds context features and graph arrays for 1,322,252 nodes | About **53 minutes** previously (3,162.8 seconds) |
| Stage 4/4 | Builds per-seed training-only scaling and masks | Prints `Seed 100` through `Seed 109`; no separate benchmark |

For planning, allow **roughly 1.5–3 hours** on comparable hardware, possibly longer
on slower or busy disks/CPUs. This is an allowance, not a guaranteed duration; a
complete end-to-end wall time was not captured. The two
historically measured aggregation/graph phases alone total about 75 minutes.
Downloading archives and model training are additional time, not included here.

The subsequent fresh CPU reconstruction recorded about **16.4 minutes** for
cohort aggregation and **21.3 minutes** for context/graph building. These are
individual phase measurements, not a full ZIP-to-input total; workload and I/O
conditions can materially change timings. Preserve both the earlier timings and
this fresh-run observation rather than presenting one as a guaranteed duration.

Keep the terminal open, prevent automatic sleep, leave at least 60 GB free disk
and run only one preparation job at a time. A quiet terminal during ZIP copying,
hashing or a SQLite query does not itself prove a hang; check CPU/disk activity
and output-file growth. Do not launch a duplicate process to obtain more logs.
Preserve errors and the output folder. Successful preparation ends with
`RQ1 graph prepared` and ten seed summaries; verify hashes/scalers with the next
section before training. Full performance tables are not produced by this step.

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
Expected: `PASS RQ1 seed 100` through `PASS RQ1 seed 109`, followed by
`PASS: registry created; no training or test evaluation performed`. A mismatch is
a stop condition; do not edit the expected hashes or fit scaling on all nodes.

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
python -B .\src\runner.py `
  --dataset blte `
  --mode validate `
  --registry .\datasets.local.json `
  --models mlp,gcn,gat,graphsage,x_moe_gnn `
  --seeds 100 `
  --check-epochs 5 `
  --threads 24 `
  --output-dir .\checks\blte_validation5
```

Wait for BLTE to finish, then run the same five-epoch check on RQ1:

```powershell
python -B .\src\runner.py `
  --dataset rq1 `
  --mode validate `
  --registry .\datasets.local.json `
  --models mlp,gcn,gat,graphsage,x_moe_gnn `
  --seeds 100 `
  --check-epochs 5 `
  --threads 24 `
  --output-dir .\checks\rq1_validation5
```

Expected: **five `passed` model statuses for each dataset**, each with
`test_evaluated: false`. This verifies short training, validation and exact
checkpoint-reload validation logits. It is not a five-epoch performance claim
and is excluded from the final 300-epoch comparison. A roughly 6.46 GiB sampled
RQ1 X-MoE peak was observed in the fresh reference check. Do not start full runs
after a failed resource check without investigating it.

## 9. First full reproduction: BLTE seed 100

This is the first full training step for a new user: five models, **one seed**,
300 epochs each. Keep new outputs separate from `results/reference`.

```powershell
python -B .\src\runner.py `
  --dataset blte `
  --mode train `
  --registry .\datasets.local.json `
  --models mlp,gcn,gat,graphsage,x_moe_gnn `
  --seeds 100 `
  --threads 24 `
  --output-dir .\results\new_runs\cpu_reproduction_blte
```

Wait for all five models to finish, then use the selected comparison in step 11.
Progress in this new output folder is separate from the published completed
study. If this folder's summary shows `5/50` and `all_complete: false`, it means
five seed-100 runs are present within its optional 50-run plan. The seed-100 check
can still be complete and successful; the author's reference study remains
**50/50 per dataset**. On the reference PC, BLTE X-MoE alone took about 12 minutes;
total time depends on hardware and other workloads.

## 10. First full reproduction: RQ1 seed 100

Run this only after BLTE seed 100 has finished. It is the same training recipe
on the rebuilt RQ1 inputs, again five models and one seed at 300 epochs.

```powershell
python -B .\src\runner.py `
  --dataset rq1 `
  --mode train `
  --registry .\datasets.local.json `
  --models mlp,gcn,gat,graphsage,x_moe_gnn `
  --seeds 100 `
  --threads 24 `
  --output-dir .\results\new_runs\cpu_reproduction_rq1
```

Do not launch these two blocks concurrently. Keep the training terminal open and
disable automatic sleep while running. Based on observed RQ1 seed 100, five models
took about 5 h 40 min; ten seeds would be roughly 57 h at that rate. This is an
estimate, not a timeout setting or guarantee. In that recorded RQ1 seed-100 CPU
run, X-MoE took approximately 3 h 54 min with expert activation recomputation
enabled to reduce memory use. This is a hardware- and execution-specific training
measurement, not a general claim about inference latency or every deployment.
Multi-expert/multitask training and memory-saving recomputation add computational
cost; predictive performance and runtime should both be considered.

Every final run selects its best validation checkpoint, reloads it, then evaluates
test once. It writes `result.json`, `resource_probe.json`, `best_checkpoint.pt`,
`test_predictions.csv`, `training_history.csv` and progress/console reports.
Each dataset root gets `summary.json` and `attempt_inventory.json`. Completion
of this selected seed requires **five verified runs per dataset**. Full ten-seed
completion later requires 50/50 per dataset.

## 11. Compare reproduced seed 100 with its reference

Compare seed 100 to **seed 100**, not to the ten-seed mean in the main tables:

```powershell
python -B .\scripts\compare_seed.py `
  --dataset blte `
  --results-dir .\results\new_runs\cpu_reproduction_blte `
  --seed 100

python -B .\scripts\compare_seed.py `
  --dataset rq1 `
  --results-dir .\results\new_runs\cpu_reproduction_rq1 `
  --seed 100
```

The checker requires exactly one verified completed attempt per model, checks
run-artifact hashes, 300 history rows, the test population and earliest maximum
validation epoch, then compares **25 predictive fields**, prediction-file bytes
and the selected epoch against the published reference. It does not train or run
model inference. It does not claim that two serialized checkpoint files must have
the same hash: their saved metadata can contain different paths/timings.

On the tested fresh checkout on the reference PC, both commands produce
`status: exact_reference_match` and `verified_completed_runs: 5`. Different
hardware may produce numerical differences: the checker then reports
`reference_differences_require_review` and exits with code 2. Preserve those
outputs and inspect environment, inputs and execution settings. Do not alter
hyperparameters, thresholds, seeds or reference scores to force a match.

Reference seed-100 Macro-F1 (percent; rounded for display):

| Model | BLTE seed 100 | RQ1 seed 100 |
|---|---:|---:|
| MLP | 36.943862 | 64.619232 |
| GCN | 34.660470 | 74.405745 |
| GAT | 25.766981 | 57.477650 |
| GraphSAGE | 50.858075 | 87.067182 |
| X-MoE | 69.894954 | 88.616788 |

If your aim is the selected reproduction check, stop after this verification and
retain the evidence. **You do not have to rerun seeds 101–109 for that scope.**

## 12. Optional: complete a new ten-seed study with seeds 101–109

These commands are for **new users choosing to reproduce the full study**. The
author's original 100 runs are already complete. Continue in the **same output
directories** used in steps 9–10, keeping configuration and registry unchanged.
Seed 100 is already counted and will not be trained again.

First finish the remaining BLTE seeds:

```powershell
python -B .\src\runner.py `
  --dataset blte `
  --mode train `
  --registry .\datasets.local.json `
  --models mlp,gcn,gat,graphsage,x_moe_gnn `
  --seeds 101,102,103,104,105,106,107,108,109 `
  --threads 24 `
  --output-dir .\results\new_runs\cpu_reproduction_blte
```

After BLTE finishes, run the remaining RQ1 seeds:

```powershell
python -B .\src\runner.py `
  --dataset rq1 `
  --mode train `
  --registry .\datasets.local.json `
  --models mlp,gcn,gat,graphsage,x_moe_gnn `
  --seeds 101,102,103,104,105,106,107,108,109 `
  --threads 24 `
  --output-dir .\results\new_runs\cpu_reproduction_rq1
```

This adds 45 runs per dataset to the five already completed: **50 per dataset,
100 total**. RQ1 can take several days for ten seeds. The preparation in step 6
is reused; do not reconstruct the graph for each model or seed. Including seed
100 again in a resume command is also supported: verified completions are skipped.

## 13. Audit the completed ten-seed results

Only after step 12 finishes, run:

```powershell
python -B .\scripts\audit_results.py `
  --dataset blte `
  --results-dir .\results\new_runs\cpu_reproduction_blte

python -B .\scripts\audit_results.py `
  --dataset rq1 `
  --results-dir .\results\new_runs\cpu_reproduction_rq1

python -B .\scripts\check_package.py
```

Expected: `verified_runs: 50` for each dataset, with `summary.json` reporting
`completed_runs: 50` and `all_complete: true`. This full-study audit intentionally
rejects a seed-100-only series; use `compare_seed.py` for step 11.

The full audit reads saved predictions and histories; it does not train or run
model inference. It checks unique completed attempts, hashes, confusion counts,
accuracy and per-class/Macro-F1, earliest best epoch and shared settings. The
runner's `summary.json` reports all ten seeds' means/sample SD and its declared
Macro-F1 comparisons. Do not select favorable repeat attempts or seeds.

## 14. Continue after interruption

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

Exact floating-point equality on different hardware/backends is not guaranteed.
CPU thread count mattered in the original reproduction audit: the explicit
24-thread execution reproduced the original BLTE seed-100 X-MoE weights exactly;
an eight-thread trajectory differed. This is why 24 is fixed here. Thread count
was restored to reproduce pre-existing evidence, not selected to optimize RQ1
test scores. See [docs/PROTOCOL.md](docs/PROTOCOL.md).

## 15. Keep your reproduction record

From the repository root, retain a local environment/revision record:

```powershell
New-Item -ItemType Directory -Force .\checks\environment | Out-Null
git rev-parse HEAD | Set-Content .\checks\environment\revision.txt -Encoding UTF8
python --version
node --version
python -m pip freeze | Set-Content .\checks\environment\requirements-freeze.txt -Encoding UTF8
Get-CimInstance Win32_ComputerSystem |
  Select-Object TotalPhysicalMemory, NumberOfLogicalProcessors
Get-CimInstance Win32_Processor | Select-Object Name
```

Keep input hashes, actual commands, resource reports, selected comparison output
and result folders alongside this record. The `checks` directory, datasets,
per-node predictions and checkpoints are local artifacts, excluded from public
publication unless separately reviewed/authorized.

## 16. If a step fails

| Symptom | Action before continuing |
|---|---|
| `node` is not recognized | Follow step 1's installation/PATH check; changing project folders is not a fix |
| Wrong Python or dependencies | Inspect `python --version`, `python -m pip --version` and `Get-Command python`; activate this checkout's environment |
| Nested pickle not found | Check the branch revision and original RQ1 ZIP layout; do not substitute different data |
| Preparation output already exists | Preserve it and choose a fresh directory; update both RQ1 paths in step 7 |
| Hash/scaler mismatch | Retain the reports and identify the source/processing difference; do not edit expected hashes |
| Long quiet preparation stage | Check CPU/disk activity and file growth; do not launch a second process |
| `stopped_low_available_ram` | Save/close unneeded workloads; inspect remaining experiment workers; keep the safety limits |
| Training interrupted | Reuse the same command/output root; completed runs skip, incomplete attempts restart at epoch 1 |
| `5/50` in a new seed-100 output folder | Five selected runs in that folder's optional ten-seed plan; use step 11 to verify them. The published study is already 50/50 per dataset |
| Reference differences | Keep all results; investigate environment/inputs/execution rather than tune on test outcomes |

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
