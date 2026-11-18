# Fresh CPU reproduction: check each stage before continuing

This is a new execution series in a separate checkout. Keep the original 100-run
archive and its environment intact. Do not start a second experiment while a
previous worker is still running. Reproducing the ScamSweeper method and live
wallet inference are outside this walkthrough.

The main README is the complete user guide. The original 100 runs and the fresh
seed-100 verification are already complete for the author. The commands here are
for a **new user's execution**, not instructions for the author to repeat them.
Stages A–F lead to ten selected seed-100 runs; continuing 101–109 is optional.

| Author's completed work | Status |
|---|---|
| BLTE main experiment, five models × seeds 100–109 | 50/50 complete |
| RQ1 main experiment, five models × seeds 100–109 | 50/50 complete |
| Fresh-checkout seed-100 reproduction, both datasets × five models | 10/10 exact reference matches |

The main reported results use all 100 original runs. This walkthrough describes
a new reader's own execution and does not require the author to repeat training.

## Stage A — new checkout and environment

Use a normal Windows PowerShell terminal. Confirm prerequisites:

```powershell
git --version
py -3.11 --version
node --version
```

Python must be 3.11.9. Install missing prerequisites using the official links in
the root README; Node.js is needed by the primitive indexer. The successful fresh
reconstruction used Node.js 26.13.0; earlier checks used 24.22.1. If `node` is not
recognized, use the installation/PATH instructions in README step 1; this is not
a project-folder problem. Do not silently change the pinned Python
or numerical libraries to get past an installation error.

Clone to a fresh, unused folder:

```powershell
Set-Location "$env:USERPROFILE\Documents"
git clone --branch two-dataset-comparison --single-branch https://github.com/nurhalizaabdulwahab197/xmoe-ethereum-fraud-detection.git xmoe-two-dataset
Set-Location .\xmoe-two-dataset
git branch --show-current
py -3.11 -m venv .venv-cpu
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv-cpu\Scripts\Activate.ps1
python -m pip install pip==26.2.2
python -m pip install -r .\requirements-cpu.txt
python -m pip check
```

If the checkout/environment directory already exists, preserve it and choose a
different new name. Stop on a dependency error and retain the complete message.
Verify package content and versions:

```powershell
python --version
python -c "import torch, torch_geometric, numpy, pandas, sklearn, scipy, psutil; print('Torch:',torch.__version__); print('PyG:',torch_geometric.__version__); print('NumPy:',numpy.__version__); print('pandas:',pandas.__version__); print('sklearn:',sklearn.__version__); print('SciPy:',scipy.__version__); print('psutil:',psutil.__version__); print('CUDA:',torch.cuda.is_available())"
python -B .\scripts\check_package.py
python -B -m unittest discover -s .\tests -p "test_*.py" -v
python -B -m unittest discover -s .\preparation -p "test_*.py" -v
```

Expected Python 3.11.9, Torch 2.14.1+cpu, PyG 2.8.0.post1, NumPy 2.4.6,
pandas 3.0.6, scikit-learn 1.9.1, SciPy 1.17.1, psutil 7.2.2, CUDA False.
Expect 16 shared-pipeline tests and 12 preparation tests to pass. The package
check verifies 100 published run aggregates; it does not retrain models or audit
undistributed per-node predictions.

## Stage B — obtain original archives and BLTE inputs

Download RQ1 and RQ4 from the source artifact link in the README. Use the original
ZIP contents rather than copying old prepared arrays for this fresh reconstruction.
The RQ4 archive supplies explicit account lists, including Scams. No RQ4 graph is
used as the evaluated graph. The frozen review exclusions are applied by the code.
The RQ1 outer ZIP holds `normal_pkl.zip`, `phish_pkl.zip` and `scams_pkl.zip`;
the script handles this nesting without requiring manual unpacking.

```powershell
New-Item -ItemType Directory -Force .\data\raw
# Place the author downloads here as RQ1.zip and RQ4.zip.
Get-Item .\data\raw\RQ1.zip, .\data\raw\RQ4.zip |
    Select-Object Name, Length
Get-FileHash .\data\raw\RQ1.zip, .\data\raw\RQ4.zip -Algorithm SHA256

git clone https://github.com/salam-ammari/Graph-Neural-Network-Based-Fraudulent-Address-Classification-in-Ethereum-Network.git upstream-paper
Get-FileHash .\upstream-paper\BLTE_multiclass_graph.pt -Algorithm SHA256
Get-FileHash .\upstream-paper\BLTE_multiclass_v2.csv -Algorithm SHA256
```

The ZIP hashes record the exact downloads. Raw pickle/list-member hashes, rather
than archive compression bytes alone, are the enforced content identity. BLTE's
expected hashes and RQ1 member hashes are in the README and dataset document.

## Stage C — rebuild RQ1 directly from ZIP

Allow at least 60 GB free disk. Close unnecessary applications after saving work.
Keep the terminal open. This stage is CPU/disk preparation, not model training.
It is required once when starting from ZIPs; the output is reused for all five
models and seeds. It is not repeated for each training run or verified restart.

```powershell
python -B .\preparation\prepare_rq1.py `
  --rq1-zip .\data\raw\RQ1.zip `
  --rq4-zip .\data\raw\RQ4.zip `
  --output-dir .\data\prepared_rq1
```

Wait for all four preparation stages: observation indexing, cohort aggregation,
context/graph arrays, and ten per-seed standardized inputs. This can take over an
hour. Do not call the source graph a full blockchain history. No unsafe graph
unpickling, invented amount feature, graph sampling or context-to-Normal labeling
is performed. Existing output directories are refused rather than overwritten.

On the reference PC, earlier cohort aggregation took about 22 minutes and the
context/graph stage about 53 minutes, before adding extraction, hashing, indexing
and scaling. Allow roughly **1.5–3 hours** for planning on comparable hardware;
slower/busy systems can take longer. A complete fresh-install total has not yet
been measured. See the root README's phase table for log messages and timing
scope. Download and training time are separate.

`Opening nested archive` and full-file hash checks may run quietly between log
messages. Check CPU/disk activity or file growth rather than starting another
copy. Keep automatic sleep disabled and do not delete output files while a job
is active. Preserve a failed directory and its last console output for diagnosis.

Expected graph: 1,322,252 nodes, 3,856,855 directed pairs, five features, 4,913
labeled nodes (636/1,165/3,112), and 1,317,339 unknown context nodes. A final graph
hash mismatch is a stop condition; do not change expected hashes to accept it.

## Stage D — verify newly rebuilt arrays and paths

```powershell
python -B .\scripts\configure_datasets.py `
  --blte .\upstream-paper `
  --rq1-graph .\data\prepared_rq1\graph `
  --rq1-inputs .\data\prepared_rq1\seed_inputs

python -B .\src\runner.py --dataset blte --mode verify --registry .\datasets.local.json --seeds 100,101,102,103,104,105,106,107,108,109
python -B .\src\runner.py --dataset rq1 --mode verify --registry .\datasets.local.json --seeds 100,101,102,103,104,105,106,107,108,109
```

All graph hashes, all ten normalized-feature/mask hashes and independently fitted
scaler/split values must match the frozen study. This verifies input reproduction
without repeating any final training. The generated registry/lock have new local
paths and their own hashes; do not use them to resume old output folders.

## Stage E — short execution and checkpoint checks

```powershell
python -c "import psutil; print('Available GiB:',round(psutil.virtual_memory().available/1024**3,2))"
python -B .\src\runner.py --dataset blte --mode validate --check-epochs 5 --seeds 100 --threads 24 --registry .\datasets.local.json --output-dir .\checks\blte_validation5
python -B .\src\runner.py --dataset rq1 --mode validate --check-epochs 5 --seeds 100 --threads 24 --registry .\datasets.local.json --output-dir .\checks\rq1_validation5
```

Aim for at least 12 GiB available before RQ1 and run only one block at a time.
Both blocks cover all five models. Validation mode saves and reloads the selected
checkpoint and checks identical validation logits after reload. It does not score
test. This does not by itself reproduce the final 300-epoch performance table.

## Stage F — reproduce seed 100, BLTE first and then RQ1

After Stages A–E pass, execute one seed for both datasets in **new** output roots:

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

Wait for all five BLTE models to finish, then run:

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

These are 300-epoch training runs. A new output folder may report `5/50` after
seed 100 because its progress counter covers the optional ten-seed plan. That
counter describes only this new folder; the published study is already 50/50 for
BLTE and 50/50 for RQ1. Compare the five selected runs before choosing whether to
perform a full new execution:

```powershell
python -B .\scripts\compare_seed.py --dataset blte --results-dir .\results\new_runs\cpu_reproduction_blte --seed 100
python -B .\scripts\compare_seed.py --dataset rq1 --results-dir .\results\new_runs\cpu_reproduction_rq1 --seed 100
```

The reference-PC result is `exact_reference_match`, five completed runs per
dataset. The checker reports 25 predictive-field comparisons per model,
prediction-byte equality and selected-epoch equality. It also checks artifact
integrity and input metadata. Differences on other hardware require review;
do not tune the recipe against reference test scores.

For the selected reproduction scope, stop here and retain the evidence. For a
new full ten-seed execution, use README step 12's explicit **101–109** commands in
these same output roots. Seed 100 is already included and will not be retrained.
This adds 90 runs to the ten already completed, for 100 total. Reuse prepared data.

The complete-study `audit_results.py` command expects 50 runs per dataset and
will correctly reject a one-seed study as incomplete. A selected reproduction
must not be reported as an independent new ten-seed experiment.

## Stage G — record the reproduction outcome

Retain the commit SHA, package hashes, environment freeze, hardware, raw-input
hashes, exact commands, input verification, resource reports and result folders.
Only after Stage C/D actually pass from ZIP should the release's validation
status be updated to claim full raw-input reconstruction. Do not mark a new
internet environment installation validated before it succeeds.

```powershell
git rev-parse HEAD
python -m pip freeze | Set-Content .\environment-reproduction.txt -Encoding UTF8
```

Keep local paths, per-node predictions and datasets out of public commits unless
their publication has been separately reviewed and authorized.
