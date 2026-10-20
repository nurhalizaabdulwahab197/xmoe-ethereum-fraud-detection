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

## Architecture Comparison

The figure below compares the architectures of the three baseline GNN models and the proposed X-MoE model used in this repository.

![Architecture comparison for Ethereum fraud detection](docs/figures/architecture_comparison.png)

All four models solve the same three-class Ethereum address classification problem:

- **Normal**
- **Phishing**
- **Scamming**

However, they differ substantially in how node features and graph information are processed.

### A) GCN

The Graph Convolutional Network (GCN) propagates and aggregates information from neighboring addresses using shared graph convolution.

The implemented pipeline is:

```text
Node features + graph edges
        |
        v
   GCNConv(64)
        |
       ReLU
        |
   Dropout(0.5)
        |
        v
   GCNConv(64)
        |
       ReLU
        |
   Dropout(0.5)
        |
 Linear classifier
        |
        v
Normal | Phishing | Scamming
```

GCN provides a simple mechanism for incorporating graph structure. However, neighborhood information is aggregated using the same convolutional mechanism for all nodes. In highly imbalanced fraud detection, minority-class signals can therefore be difficult to preserve when fraudulent addresses are sparse relative to the Normal class.

### B) GAT

The Graph Attention Network (GAT) extends neighborhood aggregation by learning attention weights over neighboring nodes.

The implemented pipeline is:

```text
Node features + graph edges
        |
        v
GATConv(32, heads=4)
        |
       ELU
        |
   Dropout(0.5)
        |
        v
GATConv(32, heads=1)
        |
       ELU
        |
 Linear classifier
        |
        v
Normal | Phishing | Scamming
```

Unlike GCN, GAT does not treat all neighbors equally. It learns which neighboring addresses should receive greater attention.

This provides additional flexibility, but attention alone does not explicitly address severe class imbalance, expert disagreement, or the fact that some Ethereum addresses may have little or no useful neighborhood structure.

### C) GraphSAGE

GraphSAGE learns node representations by aggregating information from local neighborhoods.

The implemented pipeline is:

```text
Node features + graph edges
        |
        v
   SAGEConv(64)
        |
       ReLU
        |
   Dropout(0.5)
        |
        v
   SAGEConv(64)
        |
       ReLU
        |
   Dropout(0.5)
        |
 Linear classifier
        |
        v
Normal | Phishing | Scamming
```

GraphSAGE is the strongest conventional GNN baseline in the verified CPU experiments.

Its neighborhood aggregation is effective for learning relational fraud patterns, particularly when meaningful graph connectivity exists.

However, it still relies on a single graph-learning mechanism. Its usefulness can therefore be reduced when an address is isolated or when the discriminative evidence is primarily contained in node-level features rather than neighborhood structure.

### D) X-MoE: Proposed Architecture

X-MoE differs fundamentally from the three baseline models.

Instead of relying on a single GNN backbone, X-MoE combines two complementary experts through an adaptive gating mechanism.

The architecture can be summarized as:

```text
                         Node features
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
        Conflict-aware              GraphSAGE
        multitask expert               expert
                  |                       |
                  v                       v
           Expert 1 logits         Expert 2 logits
                  |                       |
                  +-----------+-----------+
                              |
                    Expert disagreement
                              |
                 Node connectivity / degree
                              |
                              v
                       Gating network
                              |
                              v
                 Class-specific mixture
                         weights
             w_Normal | w_Phishing | w_Scamming
                              |
                              v
                  Adaptive weighted fusion
                              |
                              v
               Normal | Phishing | Scamming
```

#### Expert 1: Conflict-aware multitask expert

The first expert captures feature-level and task-specific evidence.

It produces class predictions for:

```text
Normal
Phishing
Scamming
```

and auxiliary task-specific signals.

This gives the model an alternative source of evidence when graph structure is weak or insufficient.

#### Expert 2: GraphSAGE expert

The second expert is a GraphSAGE-based graph-structure expert.

It captures relational information from neighboring Ethereum addresses and produces its own three-class prediction.

This expert is especially useful when graph connectivity provides meaningful evidence of fraudulent behavior.

#### Expert disagreement

X-MoE explicitly measures disagreement between the two experts:

```text
| Expert 1 logits - Expert 2 logits |
```

This disagreement is used as an input to the gating mechanism.

When both experts agree, the prediction has mutually supporting evidence. When they disagree, the gating network can learn which expert should receive greater influence for the current node.

#### Connectivity-aware gating

The gating network receives:

```text
Expert 1 logits
Expert 2 logits
Expert disagreement
Node connectivity / log-degree
```

It then learns adaptive mixture weights.

A key design feature is that graph-based contribution can be reduced for isolated nodes, where neighborhood aggregation provides little meaningful information.

This prevents X-MoE from assuming that every Ethereum address should rely equally on graph structure.

#### Class-specific mixture weights

X-MoE does not use one global mixture weight for the complete prediction.

Instead, it learns class-specific expert weights:

```text
w_Normal
w_Phishing
w_Scamming
```

Therefore, the importance of the feature/task-aware expert and the GraphSAGE expert can differ for each fraud class.

This is particularly relevant because Normal, Phishing, and Scamming addresses need not exhibit the same behavioral or structural patterns.

### Why the X-MoE Architecture Helps Fraud Detection

The central motivation of X-MoE is that **no single source of evidence is equally reliable for every Ethereum address**.

For a strongly connected address, graph relationships may contain useful fraud evidence.

For an isolated or weakly connected address, node-level features may be more informative than message passing.

For difficult minority-class cases, the two experts may disagree, and this disagreement itself provides useful information about prediction uncertainty.

X-MoE therefore combines four complementary signals:

1. feature/task-aware evidence;
2. graph-structural evidence;
3. disagreement between the experts; and
4. node connectivity.

The gating network uses these signals to determine how the two experts should be combined for each class.

This produces an adaptive decision mechanism rather than applying the same graph-learning rule to every address.

### Architecture-Level Interpretation by Class

For **Normal addresses**, the model must avoid incorrectly converting the large majority of legitimate-looking addresses into fraud predictions. X-MoE can reduce unnecessary graph influence when structural evidence is weak and preserve strong node-level evidence for the Normal class.

For **Phishing addresses**, discriminative evidence may arise from both account-level behavioral features and relational patterns. The mixture-of-experts design allows these two evidence sources to complement each other rather than forcing the decision through a single GNN pathway.

For **Scamming addresses**, graph relationships can provide strong structural information when suspicious interaction patterns are present. The class-specific gating mechanism allows the GraphSAGE expert to receive a different contribution for Scamming than for Normal or Phishing.

### Key Takeaway

GCN, GAT, and GraphSAGE each rely on a single graph-learning mechanism.

X-MoE instead combines a **feature/task-aware expert** and a **graph-structure expert** through a **connectivity-aware, disagreement-aware, class-specific gating network**.

This adaptive fusion is the main architectural distinction of X-MoE and provides a mechanism for improving balanced detection across Normal, Phishing, and Scamming addresses.

The verified CPU experiments support this architectural motivation:

| Model | Accuracy (mean +/- std) | Macro-F1 (mean +/- std) |
|---|---:|---:|
| GCN | 0.881178 +/- 0.032824 | 0.363702 +/- 0.020403 |
| GAT | 0.587321 +/- 0.033070 | 0.266464 +/- 0.020491 |
| GraphSAGE | 0.988206 +/- 0.002095 | 0.493079 +/- 0.034075 |
| **X-MoE** | **0.996887 +/- 0.001234** | **0.662575 +/- 0.072090** |

Relative to GraphSAGE, X-MoE improves mean Macro-F1 by:

```text
0.662575 - 0.493079 = 0.169496
```

equivalent to:

```text
+16.95 percentage points
```

or approximately:

```text
+34.4% relative improvement
```

Because the dataset is highly imbalanced, this improvement in Macro-F1 is more informative than accuracy alone for assessing balanced performance across Normal, Phishing, and Scamming.

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
|-- REPRODUCTION_METADATA_GPU.txt
|-- requirements-cpu-tested.txt
|-- requirements-gpu-tested.txt
|-- .gitignore
|
|-- src/
|   `-- reproduce.py
|
|-- docs/
|   `-- figures/
|       `-- architecture_comparison.png
|
|-- results/
|   `-- reference/
|       |-- cpu_10seed_summary.csv
|       |-- gpu_10seed_summary.csv
|       `-- CPU_REFERENCE_RESULTS.txt
|
`-- upstream-paper/        # created locally; not committed
```

## 1. Clone This Repository

```powershell
git clone https://github.com/nurhalizaabdulwahab197/xmoe-ethereum-fraud-detection.git
cd xmoe-ethereum-fraud-detection
```

Repository URL: https://github.com/nurhalizaabdulwahab197/xmoe-ethereum-fraud-detection

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

## Demo Application

A separate demonstration application will be developed after CPU and GPU reproducibility validation is complete.

The demo will be intended for model demonstration and reference use and will be linked from this repository when available.





## License and Dataset Attribution

The source code in this repository is released under the MIT License.

The BLTE dataset is **not redistributed** in this repository. The processed dataset artifacts used for reproduction must be obtained from the original upstream project:

```text
salam-ammari/Graph-Neural-Network-Based-Fraudulent-Address-Classification-in-Ethereum-Network
```

The dataset and any external data used by the upstream project remain subject to the licenses, attribution requirements, terms of use, and ethical restrictions specified by their original providers.

This repository claims authorship only for the code and methodology developed here, including the proposed X-MoE implementation and reproducibility workflow, and does not claim ownership of the upstream BLTE dataset.



## GPU Reproduction Results

GPU reproduction was independently performed using the same clean protocol, seeds 100-109, and 300 training epochs.

Tested GPU environment:

```text
GPU: NVIDIA GeForce GTX 1660 SUPER
VRAM: 6 GB
NVIDIA Driver: 595.95
PyTorch: 2.14.1+cu126
CUDA runtime: 12.6
PyTorch Geometric: 2.8.0.post1
Python: 3.11.9
```

### CPU vs GPU Macro-F1

| Model | CPU Macro-F1 | GPU Macro-F1 |
|---|---:|---:|
| GCN | 0.363702 +/- 0.020403 | 0.363670 +/- 0.017636 |
| GAT | 0.266464 +/- 0.020491 | 0.275000 +/- 0.029484 |
| GraphSAGE | 0.493079 +/- 0.034075 | 0.495760 +/- 0.031397 |
| **X-MoE** | **0.662575 +/- 0.072090** | **0.657782 +/- 0.065579** |

The model ranking is preserved across both environments:

```text
X-MoE > GraphSAGE > GCN > GAT
```

On the tested GPU environment, X-MoE improves mean Macro-F1 over GraphSAGE from 0.495760 to 0.657782, corresponding to an absolute improvement of 0.162022 (+16.20 percentage points), or approximately +32.68% relative improvement.

The CPU and GPU results are closely consistent but are not expected to be bitwise identical. Minor numerical variation can arise from differences between CPU and CUDA execution paths.

The tested GPU environment is frozen in `requirements-gpu-tested.txt`, and the complete GPU summary is stored in `results/reference/gpu_10seed_summary.csv` and `REPRODUCTION_METADATA_GPU.txt`.
