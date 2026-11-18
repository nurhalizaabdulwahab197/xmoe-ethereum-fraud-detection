# Reproduce the locked GPU matrix

## Recorded environment

Linux, Python 3.10.12, NVIDIA H200 NVL, driver 570.148.08, PyTorch 2.11.0+cu128, PyG 2.8.1, NumPy 2.2.6, pandas 2.3.3, sklearn 1.7.2, SciPy 1.15.3, psutil 7.2.2; 24 CPU threads. Exact package list: ../../results/scamsweeper/provenance/requirements-freeze.txt. CPU reference used different package versions; this is not a controlled CPU/GPU parity comparison.

## Setup

From the package root, on a compatible Linux CUDA host:

```bash
python3.10 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r results/scamsweeper/provenance/requirements-freeze.txt --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip check
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

This reproduces recorded versions where those distributions remain available. Hardware/driver compatibility and full runs must be validated on the target machine; this documentation package was audited without retraining.

## Required data

Obtain the original prepared graph through an authorized source. Set GRAPH to its directory, containing features_raw.npy, labels.npy, candidate_labeled_mask.npy and edge_index.npy. Demo normalized arrays are insufficient. No data download URL or redistribution permission is established by this package.

## Execution

```bash
cd experiments/scamsweeper/source
GRAPH=/absolute/path/to/original/graph
OUT=/absolute/path/to/NEW/results-gpu-230
python experiments/rq1_parity/run.py --stage preflight --graph "$GRAPH" --output "$OUT" --threads 24
python experiments/rq1_parity/run.py --stage smoke --graph "$GRAPH" --output "$OUT" --threads 24
python experiments/rq1_parity/run.py --stage full --graph "$GRAPH" --output "$OUT" --threads 24
```

Run each next stage only after the previous stage passes. Do NOT add --include-gpu: matrix() already creates 230 CUDA rows. The historical optional flag adds four clean GNN runs per seed and is not the all-GPU switch. CPU-only chunked GAT messages are bypassed on CUDA; activation recomputation remains enabled. Frozen pipeline files are unchanged. The exact local changes are visible against backup-before-cuda in the complete private snapshot.

## Monitoring and recovery

Use nohup with the environment's absolute Python path for a detached full run; start only one runner per output directory. The root log reports run IDs, while jobs/full/*.log contains epoch progress. Ctrl+C on tail does not stop training. A container restart can still stop the job.

Completed runs are audited before being skipped. There is no epoch-level resume checkpoint in this implementation: an interrupted run must be preserved/moved aside and restarted from epoch 1. Never treat a partial folder as complete. Changed source, environment or configuration requires a NEW output directory. Guards: >20 GiB process RAM, <4 GiB host available RAM, or six hours per worker. GPU VRAM is not the host-RAM guard.

After completion:

```bash
python experiments/rq1_parity/run.py --stage summarize --output "$OUT"
```

This audits predictions, controls, missing runs and statistics; it does not train. In the original completion JSON, gpu_included=false/gpu_parity_completed=false refer to the unused optional CPU/GPU comparison. They do not describe actual run devices. Verify device=cuda in all 230 result records. Preserve original metadata; record this explanation instead of rewriting historical evidence.
