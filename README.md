# X-MoE for Ethereum Fraud Detection

This repository provides a reproducible implementation for multiclass Ethereum address fraud detection using three graph neural network baselines and the proposed X-MoE model.

## Models

The public runner exposes four models:

- GCN
- GAT
- GraphSAGE
- X-MoE

GCN, GAT, and GraphSAGE are used as baseline models. X-MoE is the proposed mixture-of-experts model.

## Task

The models perform three-class classification:

- Normal
- Phishing
- Scamming

## CPU Reproduction Status

The CPU experiments in this repository were reproduced using:

- Python 3.11.9
- pip 24.0
- PyTorch 2.14.1+cpu
- PyTorch Geometric 2.8.0.post1
- NumPy 2.4.6
- Pandas 3.0.6
- scikit-learn 1.9.1
- SciPy 1.17.1
- CPU execution
- clean protocol
- 300 epochs
- seeds 100-109

## Final CPU Results

| Model | Accuracy (mean ± std) | Macro-F1 (mean ± std) |
|---|---:|---:|
| GCN | 0.881178 ± 0.032824 | 0.363702 ± 0.020403 |
| GAT | 0.587321 ± 0.033070 | 0.266464 ± 0.020491 |
| GraphSAGE | 0.988206 ± 0.002095 | 0.493079 ± 0.034075 |
| **X-MoE** | **0.996887 ± 0.001234** | **0.662575 ± 0.072090** |

Under this reproduced CPU configuration, X-MoE improves Macro-F1 over GraphSAGE by:

- **+0.169496 absolute Macro-F1**
- **+16.95 percentage points**
- approximately **+34.4% relative improvement**

Detailed CPU reference results are available in:

```text
results/reference/

```

## Repository Structure

```text
xmoe-ethereum-fraud-detection/
|
|-- README.md
|-- REPRODUCIBILITY.md
|-- REPRODUCTION_METADATA_CPU.txt
|-- requirements-cpu-tested.txt
|-- .gitignore
|
|-- src/
|   `-- reproduce.py
|
|-- results/
|   `-- reference/
|       |-- cpu_10seed_summary.csv
|       `-- CPU_REFERENCE_RESULTS.txt
|
`-- upstream-paper/        # created locally; not committed
```

## 1. Clone This Repository

```powershell
git clone <THIS-REPOSITORY-URL>
cd xmoe-ethereum-fraud-detection
```

Replace `<THIS-REPOSITORY-URL>` with the final GitHub URL after the repository is published.

## 2. Python Environment

The verified CPU environment uses Python 3.11.9.

Check installed Python versions:

```powershell
py -0p
```

Create the virtual environment:

```powershell
py -3.11 -m venv .venv-cpu
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Activate:

```powershell
.\.venv-cpu\Scripts\Activate.ps1
```

Verify:

```powershell
python --version
python -m pip --version
```

Verified versions:

```text
Python 3.11.9
pip 24.0
```

For strict reproduction of the tested CPU environment, do not upgrade pip before installing the tested requirements.

## 3. Install CPU Dependencies

```powershell
python -m pip install -r .\requirements-cpu-tested.txt
```

Verify the main packages:

```powershell
python -c "import torch, torch_geometric, numpy, pandas, sklearn, scipy; print('torch=' + torch.__version__); print('torch-geometric=' + torch_geometric.__version__); print('numpy=' + numpy.__version__); print('pandas=' + pandas.__version__); print('sklearn=' + sklearn.__version__); print('scipy=' + scipy.__version__); print('cuda=' + str(torch.cuda.is_available()))"
```

Expected tested CPU configuration:

```text
torch=2.14.1+cpu
torch-geometric=2.8.0.post1
numpy=2.4.6
pandas=3.0.6
sklearn=1.9.1
scipy=1.17.1
cuda=False
```

A PyTorch Geometric `torch.jit.script` FutureWarning may appear. It does not prevent the experiments from running.

## 4. Obtain the BLTE Dataset Artifacts

Clone the upstream repository:

```powershell
git clone https://github.com/salam-ammari/Graph-Neural-Network-Based-Fraudulent-Address-Classification-in-Ethereum-Network.git upstream-paper
```

Required artifacts:

```text
upstream-paper/BLTE_multiclass_graph.pt
upstream-paper/BLTE_multiclass_v2.csv
```

Verify SHA256:

```powershell
Get-FileHash `
  .\upstream-paper\BLTE_multiclass_graph.pt, `
  .\upstream-paper\BLTE_multiclass_v2.csv `
  -Algorithm SHA256
```

Expected hashes:

```text
BLTE_multiclass_graph.pt
2859A8A3A815EDD02B5CD85D46C00D730B2B66C4E8D047145DB2E843C9E38456

BLTE_multiclass_v2.csv
02E376DDD98BA595372F19731FE5E51BBC90BF897B2A52B3C8BB98E786423B11
```

The upstream dataset directory is intentionally excluded from this repository.

## 5. Verify Available Models

```powershell
python .\src\reproduce.py --help
```

The public model choices should be:

```text
{gcn,gat,graphsage,x_moe_gnn}
```

## 6. Smoke Tests

Smoke tests use seed 100 and 10 epochs. They verify that each training pipeline runs successfully. They are not final performance measurements.

### GCN

```powershell
python .\src\reproduce.py `
  --model gcn `
  --protocol clean `
  --seeds 100 `
  --epochs 10 `
  --device cpu `
  --upstream .\upstream-paper `
  --output-dir .\results\smoke_gcn `
  --no-save-checkpoints `
  --no-save-predictions
```

### GAT

```powershell
python .\src\reproduce.py `
  --model gat `
  --protocol clean `
  --seeds 100 `
  --epochs 10 `
  --device cpu `
  --upstream .\upstream-paper `
  --output-dir .\results\smoke_gat `
  --no-save-checkpoints `
  --no-save-predictions
```

### GraphSAGE

```powershell
python .\src\reproduce.py `
  --model graphsage `
  --protocol clean `
  --seeds 100 `
  --epochs 10 `
  --device cpu `
  --upstream .\upstream-paper `
  --output-dir .\results\smoke_graphsage `
  --no-save-checkpoints `
  --no-save-predictions
```

Verified GraphSAGE smoke-test summary:

```text
accuracy         0.001461
macro_precision  0.005986
macro_recall     0.377778
macro_f1         0.011117
weighted_f1      0.000032
```

### X-MoE

```powershell
python .\src\reproduce.py `
  --model x_moe_gnn `
  --protocol clean `
  --seeds 100 `
  --epochs 10 `
  --device cpu `
  --upstream .\upstream-paper `
  --x-moe-normal-phishing-penalty 0.30 `
  --output-dir .\results\smoke_xmoe `
  --no-save-checkpoints `
  --no-save-predictions
```

## 7. Full 10-Seed CPU Reproduction

All final CPU experiments use:

```text
protocol = clean
seeds = 100,101,102,103,104,105,106,107,108,109
epochs = 300
device = cpu
deterministic = enabled
```

### GCN

```powershell
python .\src\reproduce.py `
  --model gcn `
  --protocol clean `
  --seeds 100,101,102,103,104,105,106,107,108,109 `
  --epochs 300 `
  --device cpu `
  --deterministic `
  --upstream .\upstream-paper `
  --output-dir .\results\final_gcn `
  --save-checkpoints `
  --save-predictions
```

### GAT

```powershell
python .\src\reproduce.py `
  --model gat `
  --protocol clean `
  --seeds 100,101,102,103,104,105,106,107,108,109 `
  --epochs 300 `
  --device cpu `
  --deterministic `
  --upstream .\upstream-paper `
  --output-dir .\results\final_gat `
  --save-checkpoints `
  --save-predictions
```

### GraphSAGE

```powershell
python .\src\reproduce.py `
  --model graphsage `
  --protocol clean `
  --seeds 100,101,102,103,104,105,106,107,108,109 `
  --epochs 300 `
  --device cpu `
  --deterministic `
  --upstream .\upstream-paper `
  --output-dir .\results\final_graphsage `
  --save-checkpoints `
  --save-predictions
```

### X-MoE

```powershell
python .\src\reproduce.py `
  --model x_moe_gnn `
  --protocol clean `
  --seeds 100,101,102,103,104,105,106,107,108,109 `
  --epochs 300 `
  --device cpu `
  --deterministic `
  --upstream .\upstream-paper `
  --x-moe-normal-phishing-penalty 0.30 `
  --output-dir .\results\final_xmoe `
  --save-checkpoints `
  --save-predictions
```

## 8. Clean Protocol

The `clean` protocol regenerates a stratified 70/15/15 train/validation/test split for every seed.

Feature standardization is fitted on the training nodes only.

The released BLTE graph topology is retained.

## 9. Interpretation of the CPU Results

The verified 10-seed CPU results are:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| GCN | 0.881178 | 0.363702 |
| GAT | 0.587321 | 0.266464 |
| GraphSAGE | 0.988206 | 0.493079 |
| **X-MoE** | **0.996887** | **0.662575** |

The main comparison is X-MoE versus GraphSAGE.

X-MoE increases mean Macro-F1 from:

```text
0.493079
```

to:

```text
0.662575
```

corresponding to:

```text
absolute improvement = 0.169496
percentage-point improvement = 16.95
relative improvement ˜ 34.4%
```

## 10. Reproducibility Scope

This repository reproduces the experiments using the processed BLTE artifacts released by the upstream project.

It does not independently reconstruct the complete BLTE dataset from raw blockchain data.

The `clean` protocol addresses feature-scaling leakage by fitting the scaler using training nodes only. The released graph topology is nevertheless retained.

Because results can be affected by numerical libraries, operating system, hardware, and low-level implementation differences, independently reproduced values should be expected to closely match the documented results rather than necessarily be bit-for-bit identical on every machine.

## 11. Generated Outputs

When checkpoints and predictions are enabled, the runner generates per-seed model checkpoints and prediction files.

These generated files are intentionally excluded from version control.

The curated CPU reference summaries are stored in:

```text
results/reference/
```

## 12. Reproduction Metadata

The exact verified CPU configuration is also recorded in:

```text
REPRODUCTION_METADATA_CPU.txt
```

Additional reproducibility details are documented in:

```text
REPRODUCIBILITY.md
```

## GPU Reproduction

GPU reproduction will be added to this repository after validation on the GPU environment.

The GPU section will document:

- GPU model
- driver and CUDA configuration
- PyTorch/CUDA build
- dependency versions
- exact commands
- seeds
- runtime
- final GCN, GAT, GraphSAGE, and X-MoE results
- comparison with the verified CPU results

## Demo Application

A separate demonstration application will be developed after CPU and GPU reproducibility validation is complete.

The demo will be intended for model demonstration and reference use and will be linked from this repository when available.

