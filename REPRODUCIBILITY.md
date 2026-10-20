# Reproducibility Guide

This document records the reproducibility protocol used for the CPU experiments in this repository.

The purpose is to allow an independent user to recreate the environment, obtain the same processed BLTE artifacts, execute the same experimental protocol, and obtain results that closely match the verified reference values.

---

## 1. Public Experimental Scope

The public runner exposes four models:

- GCN
- GAT
- GraphSAGE
- X-MoE

GCN, GAT, and GraphSAGE are the baseline models.

X-MoE is the proposed model.

The public command-line interface is:

```text
--model {gcn,gat,graphsage,x_moe_gnn}
```

Some internal classes remain in `src/reproduce.py` because they are implementation components required by X-MoE. They are not exposed as standalone public experimental models.

---

## 2. Verified CPU Environment

The CPU reproduction reported in this repository was verified using:

```text
Python:              3.11.9
pip:                 24.0
PyTorch:             2.14.1+cpu
PyTorch Geometric:   2.8.0.post1
NumPy:               2.4.6
Pandas:              3.0.6
scikit-learn:        1.9.1
SciPy:               1.17.1
CUDA available:      False
```

The complete tested Python dependency snapshot is stored in:

```text
requirements-cpu-tested.txt
```

The reproduction metadata recorded during verification is stored in:

```text
REPRODUCTION_METADATA_CPU.txt
```

For strict CPU reproduction, use Python 3.11 and do not upgrade pip before installing the tested requirements.

---

## 3. Windows Environment Setup

Create a Python 3.11 environment:

```powershell
py -3.11 -m venv .venv-cpu
```

If PowerShell prevents activation:

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

Expected:

```text
Python 3.11.9
pip 24.0
```

Install dependencies:

```powershell
python -m pip install -r .\requirements-cpu-tested.txt
```

Verify the main scientific stack:

```powershell
python -c "import torch, torch_geometric, numpy, pandas, sklearn, scipy; print('torch=' + torch.__version__); print('torch-geometric=' + torch_geometric.__version__); print('numpy=' + numpy.__version__); print('pandas=' + pandas.__version__); print('sklearn=' + sklearn.__version__); print('scipy=' + scipy.__version__); print('cuda=' + str(torch.cuda.is_available()))"
```

Expected:

```text
torch=2.14.1+cpu
torch-geometric=2.8.0.post1
numpy=2.4.6
pandas=3.0.6
sklearn=1.9.1
scipy=1.17.1
cuda=False
```

A PyTorch Geometric warning concerning `torch.jit.script` may appear. It did not prevent the verified experiments from running.

---

## 4. Dataset Artifacts

The experiments use the processed BLTE artifacts released by the upstream project.

Clone the upstream repository:

```powershell
git clone https://github.com/salam-ammari/Graph-Neural-Network-Based-Fraudulent-Address-Classification-in-Ethereum-Network.git upstream-paper
```

Required files:

```text
upstream-paper/BLTE_multiclass_graph.pt
upstream-paper/BLTE_multiclass_v2.csv
```

Verified SHA256:

```text
BLTE_multiclass_graph.pt
2859A8A3A815EDD02B5CD85D46C00D730B2B66C4E8D047145DB2E843C9E38456

BLTE_multiclass_v2.csv
02E376DDD98BA595372F19731FE5E51BBC90BF897B2A52B3C8BB98E786423B11
```

Verify on Windows:

```powershell
Get-FileHash `
  .\upstream-paper\BLTE_multiclass_graph.pt, `
  .\upstream-paper\BLTE_multiclass_v2.csv `
  -Algorithm SHA256
```

The upstream artifact directory is excluded from version control.

---

## 5. Dataset and Evaluation Scope

The classification task contains three classes:

```text
Normal
Phishing
Scamming
```

The verified experiments use the `clean` protocol.

For each seed, this protocol:

1. generates a fresh stratified 70/15/15 train/validation/test split;
2. fits `StandardScaler` on training nodes only;
3. applies the fitted scaler to validation and test nodes;
4. retains the released BLTE graph topology;
5. selects the model state according to validation performance;
6. evaluates on the held-out test nodes.

The processed BLTE artifacts are reproduced as released.

This repository does not independently reconstruct the full BLTE dataset from raw Ethereum blockchain data.

---

## 6. Random Seeds

The final experiments use ten seeds:

```text
100
101
102
103
104
105
106
107
108
109
```

Command-line form:

```text
--seeds 100,101,102,103,104,105,106,107,108,109
```

All final CPU experiments use:

```text
epochs = 300
device = cpu
deterministic = enabled
protocol = clean
```

---

## 7. X-MoE Configuration

The verified X-MoE experiment additionally uses:

```text
x_moe_normal_phishing_penalty = 0.30
```

Command-line form:

```text
--x-moe-normal-phishing-penalty 0.30
```

Other relevant runner defaults recorded during the verified experiments include:

```text
base_loss = weighted_ce
expert_loss_weight = 0.25
scam_specialist_loss_weight = 0.5
x_moe_v8_task_loss_weight = 1.0
task_loss_weight = 0.5
focal_gamma = 2.0
```

These values are generated by the released runner unless explicitly overridden.

---

## 8. Smoke-Test Validation

Before running full experiments, each of the four public models was successfully smoke-tested using:

```text
seed = 100
epochs = 10
protocol = clean
device = cpu
```

Models successfully tested:

```text
GCN        PASS
GAT        PASS
GraphSAGE  PASS
X-MoE      PASS
```

The verified GraphSAGE smoke-test result was:

```text
accuracy         0.001461
macro_precision  0.005986
macro_recall     0.377778
macro_f1         0.011117
weighted_f1      0.000032
```

The low 10-epoch performance is expected for this smoke test. The purpose is execution validation, not final model evaluation.

---

## 9. Final CPU Commands

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

---

## 10. Verified CPU Results

The final 10-seed CPU results are:

| Model | Accuracy mean | Accuracy std | Macro-F1 mean | Macro-F1 std |
|---|---:|---:|---:|---:|
| GCN | 0.881178 | 0.032824 | 0.363702 | 0.020403 |
| GAT | 0.587321 | 0.033070 | 0.266464 | 0.020491 |
| GraphSAGE | 0.988206 | 0.002095 | 0.493079 | 0.034075 |
| **X-MoE** | **0.996887** | **0.001234** | **0.662575** | **0.072090** |

The complete reference table is stored in:

```text
results/reference/cpu_10seed_summary.csv
```

A human-readable result summary is stored in:

```text
results/reference/CPU_REFERENCE_RESULTS.txt
```

---

## 11. X-MoE Improvement over GraphSAGE

The main baseline comparison is:

```text
GraphSAGE Macro-F1 = 0.493079
X-MoE Macro-F1     = 0.662575
```

Absolute improvement:

```text
0.662575 - 0.493079 = 0.169496
```

Equivalent percentage-point increase:

```text
+16.95 percentage points
```

Approximate relative improvement:

```text
+34.4%
```

The CPU reproduction therefore confirms that, under the documented protocol and environment, X-MoE achieves higher mean Macro-F1 than GCN, GAT, and GraphSAGE.

---

## 12. Generated Files

When enabled, each full experiment generates:

- validation-selected model checkpoints;
- per-seed prediction CSV files;
- metric/result files produced by the runner.

These files can be regenerated from the commands above and are therefore excluded from the public repository.

Only curated reference summaries are tracked under:

```text
results/reference/
```

---

## 13. Reproducibility Expectations

The verified environment provides a reference configuration for independent reproduction.

Exact bit-for-bit equality on all hardware and operating systems is not guaranteed.

Potential sources of small numerical differences include:

- CPU architecture;
- operating system;
- numerical libraries;
- PyTorch/PyG implementation details;
- thread scheduling;
- floating-point execution differences.

Independent runs should therefore be compared against the documented means, standard deviations, and overall ranking of the models.

The central reproduced finding is:

```text
X-MoE > GraphSAGE > GCN/GAT
```

with X-MoE providing the highest mean Macro-F1 under the verified CPU protocol.

---

## 14. Windows Path Note

The verified public reproduction was executed from a local non-OneDrive directory.

Example:

```text
C:\Users\<USER>\Desktop\Fraud\xmoe-ethereum-fraud-detection
```

During earlier environment preparation, package/path issues were encountered in a OneDrive-backed working directory.

If Windows users encounter unusual filesystem or package-installation errors, using a normal local directory outside OneDrive is recommended.

This is a troubleshooting recommendation, not a claim that OneDrive necessarily causes reproducibility failure.

---

## 15. Verified GPU Reproduction

GPU reproduction was completed using the same clean protocol, seeds 100-109, and 300 training epochs.

Verified GPU environment:

`	ext
GPU:                  NVIDIA GeForce GTX 1660 SUPER
VRAM:                 6 GB
NVIDIA Driver:        595.95
Driver-reported CUDA: 13.2
PyTorch:              2.14.1+cu126
CUDA runtime:         12.6
PyTorch Geometric:    2.8.0.post1
Python:               3.11.9
`

The final 10-seed GPU results are:

| Model | Accuracy mean | Accuracy std | Macro-F1 mean | Macro-F1 std |
|---|---:|---:|---:|---:|
| GCN | 0.884409 | 0.030307 | 0.363670 | 0.017636 |
| GAT | 0.598466 | 0.044165 | 0.275000 | 0.029484 |
| GraphSAGE | 0.988444 | 0.002025 | 0.495760 | 0.031397 |
| **X-MoE** | **0.996385** | **0.001714** | **0.657782** | **0.065579** |

The model ranking is preserved across the tested CPU and GPU environments:

`	ext
X-MoE > GraphSAGE > GCN > GAT
`

On GPU, X-MoE improves mean Macro-F1 over GraphSAGE by 0.162022, corresponding to +16.20 percentage points or approximately +32.68% relative improvement.

The CPU and GPU results are closely consistent but are not expected to be bitwise identical.

Detailed GPU environment and result files:

`	ext
requirements-gpu-tested.txt
REPRODUCTION_METADATA_GPU.txt
results/reference/gpu_10seed_summary.csv
`

---

## 16. Demo Application

A separate demonstration project will be developed only after CPU and GPU reproducibility validation is complete.

The demonstration application will be linked from the main repository when it is publicly available.
