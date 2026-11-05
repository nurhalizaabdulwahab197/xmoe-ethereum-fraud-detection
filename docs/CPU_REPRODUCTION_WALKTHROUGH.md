# Fresh CPU reproduction: check each stage before continuing

This is a new execution series in a separate checkout. Keep the original 100-run
archive and its environment intact. Do not start a second experiment while a
previous worker is still running. Reproducing the ScamSweeper method and live
wallet inference are outside this walkthrough.

## Stage A — new checkout and environment

Use a normal Windows PowerShell terminal. Confirm prerequisites:

```powershell
git --version
py -3.11 --version
node --version
```

Python must be 3.11.9. Install missing prerequisites using the official links in
the root README; Node.js is needed by the primitive indexer. The recorded current
preparation checks used Node.js 24.22.1. Do not silently change the pinned Python
or numerical libraries to get past an installation error.

Clone to a fresh, unused folder:

```powershell
Set-Location "$env:USERPROFILE\Documents"
git clone --branch two-dataset-comparison --single-branch https://github.com/nurhalizaabdulwahab197/xmoe-ethereum-fraud-detection.git xmoe-two-dataset-cpu-check
Set-Location .\xmoe-two-dataset-cpu-check
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
Expect 16 shared-pipeline tests and seven preparation tests to pass. The package
check verifies 100 published run aggregates; it does not retrain models or audit
undistributed per-node predictions.

## Stage B — obtain original archives and BLTE inputs

Download RQ1 and RQ4 from the source artifact link in the README. Use the original
ZIP contents rather than copying old prepared arrays for this fresh reconstruction.
The RQ4 archive supplies explicit account lists, including Scams. No RQ4 graph is
used as the evaluated graph. The frozen review exclusions are applied by the code.

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

```powershell
python -B .\preparation\prepare_rq1.py `
  --rq1-zip .\data\raw\RQ1.zip `
  --rq4-zip .\data\raw\RQ4.zip `
  --output-dir .\data\fresh_rq1
```

Wait for all four preparation stages: observation indexing, cohort aggregation,
context/graph arrays, and ten per-seed standardized inputs. This can take over an
hour. Do not call the source graph a full blockchain history. No unsafe graph
unpickling, invented amount feature, graph sampling or context-to-Normal labeling
is performed. Existing output directories are refused rather than overwritten.

Expected graph: 1,322,252 nodes, 3,856,855 directed pairs, five features, 4,913
labeled nodes (636/1,165/3,112), and 1,317,339 unknown context nodes. A final graph
hash mismatch is a stop condition; do not change expected hashes to accept it.

## Stage D — verify newly rebuilt arrays and paths

```powershell
python -B .\scripts\configure_datasets.py `
  --blte .\upstream-paper `
  --rq1-graph .\data\fresh_rq1\graph `
  --rq1-inputs .\data\fresh_rq1\seed_inputs

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
python -B .\src\runner.py --dataset blte --mode validate --check-epochs 5 --seeds 100 --threads 24 --registry .\datasets.local.json --output-dir .\checks\fresh_blte_validation5
python -B .\src\runner.py --dataset rq1 --mode validate --check-epochs 5 --seeds 100 --threads 24 --registry .\datasets.local.json --output-dir .\checks\fresh_rq1_validation5
```

Aim for at least 12 GiB available before RQ1 and run only one block at a time.
Both blocks cover all five models. Validation mode saves and reloads the selected
checkpoint and checks identical validation logits after reload. It does not score
test. This does not by itself reproduce the final 300-epoch performance table.

## Stage F — selected final-run reproduction

After Stages A–E pass, execute one seed for both datasets in **new** output roots:

```powershell
python -B .\src\runner.py --dataset blte --mode train --registry .\datasets.local.json --models mlp,gcn,gat,graphsage,x_moe_gnn --seeds 100 --threads 24 --output-dir .\results\new_runs\fresh_blte
python -B .\src\runner.py --dataset rq1 --mode train --registry .\datasets.local.json --models mlp,gcn,gat,graphsage,x_moe_gnn --seeds 100 --threads 24 --output-dir .\results\new_runs\fresh_rq1
```

Run BLTE first, then RQ1. These are 300-epoch training runs, not smoke tests.
Compare seed 100 with `results/reference/<dataset>/per_seed.csv`, accounting for
documented cross-machine numerical limits. Preserve unfavorable outcomes. Full
ten-seed reproduction is a separate commitment: use the complete seed list from
the root README in these same new roots; verified seed 100 will be skipped.

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
