# BLTE reproduction and X-MoE evaluation

This project examines how far the multiclass BLTE experiments can be reconstructed from the published paper and released code. It evaluates an additional MLP baseline and Full X-MoE under the reconstructed data/evaluation conditions. X-MoE–CE is retained as a training-control variant to examine the contribution of the complete additional training package.

The paper and released code specify different training settings. We retain both configurations rather than silently choosing one. Original numerical split seeds and partitions could not be recovered from the audited materials, so repeated evaluation uses our documented reconstructed partitions.

**Scope:** five main methods plus one training-control variant × two configurations × ten splits = **120 primary runs**, each with 300 epochs. The six released-mask pilot runs are separate. This is not a reproduction of all seven models reported in the reference paper.

The accompanying [reproduction report](docs/REPRODUCTION_REPORT.md) explains source differences, assumptions and findings.

## Release status

The archived evidence for all 120 primary runs and 40 PCGrad ablation runs was verified locally. This branch provides source code, documentation, saved split indices, and result tables. GPU execution of the public reproduction wrapper, `scripts/runner.py`, remains pending validation.

Run `python scripts/check_package.py` to check the published package. See [LOCAL_FIRST_STEPS.md](LOCAL_FIRST_STEPS.md) for local audit and evidence import instructions. Training through the wrapper requires `--run`.

## Evaluation structure

1. **BLTE reconstruction:** audit the paper and released code, document differences and assumptions, and evaluate GCN, GAT and GraphSAGE.
2. **Additional MLP baseline:** complete the feature-only comparison using our explicitly described implementation.
3. **Full X-MoE evaluation:** introduce the proposed complete method and compare it with the four baselines under both reconstructed configurations.
4. **X-MoE–CE training control:** retain the architecture with weighted CE only to assess the combined benefit of the additional training package.
5. **Clean/strict evaluation:** extend the investigation through the separate `main` workflow to examine preprocessing and graph visibility.
6. **Component ablation:** compare Full X-MoE with gradient projection enabled and disabled.

This is the logical organization of the evaluation, not a claim about the chronology of model development. The primary results comprise 120 runs: six methods or variants evaluated under two configurations across ten reconstructed partitions. Additional implementation checks are excluded from these primary summaries.

## Source and terminology

Reference paper: **Explainable Graph Neural Networks for Multi-Class Fraudulent Address Classification in Ethereum**, *Future Internet* 2026, 18, 461. [DOI](https://doi.org/10.3390/fi18090461).

Audited upstream repository: [author implementation](https://github.com/salam-ammari/Graph-Neural-Network-Based-Fraudulent-Address-Classification-in-Ethereum-Network).

Pinned upstream commit: `56afa30511fec8bf53d1f8270d47e5793186aadb`.

The audited paper concerns the multiclass extension of BLTE; it is not the original binary BLTE dataset publication. Cite both the dataset origin and multiclass reference as appropriate; see the upstream attribution.

| Label in this guide | Meaning |
|---|---|
| `released-code` | Training settings taken from released scripts; LR 0.001 |
| `paper-described` | Settings reconstructed from Table 8; LR 0.01 and documented GAT changes |
| `X-MoE–CE` | Our X-MoE architecture with weighted multiclass cross-entropy only |
| `Full X-MoE` | Same proposed architecture with its complete multitask training package |
| Reconstructed split | Our saved train/validation/test indices; not claimed to be the author's original partition |
| Seed | Our explicit random-state choice; paper run labels are not treated as recovered RNG seeds |

Existing audit metadata uses `author` for `released-code`, `paper` for `paper-described`, `XMoE_CE` for X-MoE–CE, and `Full_XMoE` for Full X-MoE. Final packaging must retain this mapping.

## Results at a glance

Mean test Macro-F1 ± **sample SD**, in percent, over ten reconstructed splits:

| Method | Released-code configuration | Paper-described configuration |
|---|---:|---:|
| Our MLP | 37.4544 ± 1.3567 | 37.7082 ± 1.1833 |
| GCN | 36.5333 ± 2.1597 | 41.4269 ± 2.2639 |
| GAT | 26.6442 ± 2.4989 | 28.5816 ± 4.5819 |
| GraphSAGE | 48.9195 ± 2.7373 | 52.0641 ± 3.1524 |
| X-MoE–CE | 48.5667 ± 3.3531 | 60.7443 ± 5.3174 |
| **Full X-MoE** | **67.6952 ± 6.9504** | **71.1905 ± 4.9837** |

Full X-MoE exceeded every comparator on all ten matched splits in both configurations. Its mean advantage over GraphSAGE was 18.7757 and 19.1264 percentage points, respectively. The ten exploratory Full-X-MoE comparisons had Holm-adjusted p = 0.01953125. Overlapping repeated partitions and prior test exposure limit inferential interpretation.

X-MoE–CE did not establish an advantage over GraphSAGE in the released-code configuration at 300 epochs. This result is retained. Full-versus-CE improvements concern the complete additional training package; they do not isolate PCGrad.

## Beginner guide: what to prepare

- A Windows computer with WSL 2 and Ubuntu 22.04, or a comparable Linux setup.
- An NVIDIA GPU and working host driver. The recorded reproduction used a GTX 1660 SUPER with 6 GB VRAM; the paper reports an RTX 3090 with 24 GB.
- Python 3.10.12 for the recorded environment.
- The pinned upstream inputs and, after review, the released reproduction package.
- A terminal opened **inside Ubuntu**, not Windows PowerShell, for the Bash commands below.

CPU execution is not the environment used for these audit results. Do not silently fall back to CPU or change architecture when reproducing the recorded GPU experiment.

### 1. Open Ubuntu in VS Code

1. Install the VS Code WSL extension if it is not already installed.
2. Open Ubuntu from the Windows Start menu.
3. Run the commands below from Ubuntu:

```bash
cd "$HOME/blte-reproduction/BLTE_reproduction_local_candidate"
code .
```

4. In VS Code, open **Terminal → New Terminal**. Confirm that the terminal is Bash in Ubuntu:

```bash
uname -s
pwd
```

`uname -s` should print `Linux`. `pwd` prints your current directory. Keep this project separate from other experiment folders.

### 2. Check the operating system, Python and GPU

```bash
cat /etc/os-release
python3 --version
nvidia-smi
```

Recorded values: Ubuntu 22.04.3 LTS, Python 3.10.12, GTX 1660 SUPER, host NVIDIA driver 595.95. Exact GPU/driver identity with the paper is not claimed.

The `CUDA Version` displayed by `nvidia-smi` is driver capability, not the PyTorch runtime. We use a PyTorch build with CUDA **11.8**.

### 3. Create a dedicated Python environment

If Python reports a version other than 3.10.12, record the difference before comparing results. Do not silently substitute another Python version.

Install venv support if needed:

```bash
sudo apt-get update
sudo apt-get install --no-upgrade python3.10-venv git
```

Create the environment once:

```bash
python3.10 -m venv .venv-author
source .venv-author/bin/activate
python --version
python -m pip --version
```

If `.venv-author` already exists, inspect it instead of overwriting it. A failed environment without pip should be preserved or renamed before making a fresh environment.

For the recorded installer version:

```bash
python -m pip install pip==22.0.2
```

When opening a new terminal later, return to this folder and activate again:

```bash
cd "$HOME/blte-reproduction/BLTE_reproduction_local_candidate"
source .venv-author/bin/activate
```

### 4. Install the recorded principal libraries

Install PyTorch first:

```bash
python -m pip install torch==2.0.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

Install numerical, plotting and graph libraries:

```bash
python -m pip install \
  numpy==1.24.4 scipy==1.10.1 pandas==2.0.3 \
  scikit-learn==1.3.2 matplotlib==3.7.3 \
  seaborn==0.12.2 networkx==3.1
```

Install the compatible prebuilt PyG extensions:

```bash
python -m pip install \
  torch-scatter==2.1.1+pt20cu118 \
  torch-sparse==0.6.17+pt20cu118 \
  --only-binary=:all: \
  --find-links https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

Then install PyG:

```bash
python -m pip install torch-geometric==2.3.1
python -m pip check
```

These commands specify the principal versions used. They are not a substitute for the complete recorded dependency freeze, which must accompany the final package. Wheel availability and compatibility must be checked during release validation. If pip starts compiling a missing CUDA extension or cannot find a wheel, retain the error; do not silently install a newer stack.

### 5. Verify GPU computation

```bash
python - <<'PY'
import sys
import torch
import torch_geometric
import numpy
import scipy
import pandas
import sklearn
import torch_scatter
import torch_sparse

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("PyG:", torch_geometric.__version__)
print("CUDA runtime:", torch.version.cuda)
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("torch_scatter:", torch_scatter.__version__)
print("torch_sparse:", torch_sparse.__version__)
assert torch.__version__ == "2.0.1+cu118"
assert torch_geometric.__version__ == "2.3.1"
assert torch.cuda.is_available(), "CUDA is unavailable; stop and inspect."
x = torch.tensor([1., 2., 3.], device="cuda", requires_grad=True)
(x*x).sum().backward()
torch.cuda.synchronize()
assert torch.allclose(x.grad, torch.tensor([2., 4., 6.], device="cuda"))
print("GPU:", torch.cuda.get_device_name(0))
print("PASS_GPU_BACKWARD")
PY
```

Record your environment:

```bash
mkdir -p environment-records
python -m pip freeze > environment-records/requirements-freeze.txt
nvidia-smi > environment-records/nvidia-smi.txt
python --version > environment-records/python-version.txt
```

### 6. Obtain the exact upstream inputs

From the project directory, clone into a separate folder:

```bash
git clone https://github.com/salam-ammari/Graph-Neural-Network-Based-Fraudulent-Address-Classification-in-Ethereum-Network.git upstream-paper
git -C upstream-paper checkout --detach 56afa30511fec8bf53d1f8270d47e5793186aadb
git -C upstream-paper rev-parse HEAD
```

This changes only the upstream checkout. If the directory already exists, inspect its status rather than repeating the clone.

Verify the input hashes:

```bash
python - <<'PY'
from pathlib import Path
import hashlib
expected = {
    "BLTE_multiclass_graph.pt": "2859a8a3a815edd02b5cd85d46c00d730b2b66c4e8d047145db2e843c9e38456",
    "BLTE_multiclass_v2.csv": "02e376ddd98ba595372f19731fe5e51bbc90bf897b2a52b3c8bb98e786423b11",
}
for name, wanted in expected.items():
    path = Path("upstream-paper") / name
    assert path.is_file(), path
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == wanted, (name, actual)
    print("PASS:", name)
PY
```

Stop if either hash differs. Inputs are obtained from the original source; this package should not redistribute them without checking their terms.

### 7. Understand the inputs before training

The verified graph contains 73,033 nodes, 23 features and 6,588 directed edges. Labels are Normal = 0, Phishing = 1, Scamming = 2, with counts 72,865 / 69 / 99.

The released graph includes fixed train/validation/test masks. Our fixed-mask pilots use those masks. The repeated primary evaluation instead uses ten saved reconstructed stratified partitions.

The released features numerically match StandardScaler fitted on all CSV nodes. The audit retains these features for source fidelity; it does not claim train-only scaling. The `main` clean experiment uses a different preprocessing procedure.

### 8. Understand how the ten partitions are generated

For each integer seed 100 through 109, using scikit-learn 1.3.2:

1. Split all node indices into 70% train and 30% remainder, stratifying by label.
2. Split that remainder equally into validation and test, again stratifying.
3. Use the same seed for both calls and for model initialization.
4. Sort indices, save them, verify no overlaps or missing nodes, and hash the files.
5. Reuse those exact saved indices for every method and both configurations.

The split keys are `train`, `validation`, `test`. Class counts can differ by one minority node between validation and test because of integer rounding. These are our documented seed choices, not recovered author seeds.

## Portable commands: local validation required

The saved partitions are included. Training never silently rebuilds partitions. The original GNN and CE training scripts are retained separately for `author` and `paper` to preserve their tested calculations. Public CLI labels map to these historical names through `provenance/CONFIGURATIONS.json`.

### 9. Check the package and recorded GPU environment

```bash
python scripts/check_package.py
python scripts/check_environment.py
python scripts/check_models.py
```

The first command needs only the Python standard library. The others perform GPU forward/backward checks without optimizer steps. The synthetic CE checks are not substitutes for validating a complete Full-X-MoE experiment wrapper.

### 10. Verify inputs and saved partitions

```bash
python scripts/verify_inputs.py --upstream upstream-paper
python scripts/verify_splits.py --upstream upstream-paper
```

The second command reconstructs indices in memory and compares arrays with the saved partitions. It does not replace the saved files. Stop and inspect any mismatch.

### 11. Inspect a training plan without running

```bash
python scripts/runner.py \
  --model gcn --configuration released-code --seeds 100 \
  --upstream upstream-paper \
  --output-dir results/new_runs/released-code/gcn
```

This prints a plan and checks source hashes. It does not train or write run outputs. All experiment jobs use 300 epochs and CUDA. There is no CPU fallback or arbitrary epoch override in this reference wrapper.

The six fixed-mask pilots are described in the report. This package's runner targets the primary reconstructed-partition evaluation; it does not claim to rerun the original plotting/GNNExplainer pilot stages.

### 12. Run each primary method/configuration

**The following commands perform training. They are provided for review, not as instructions to start new experiments now. First complete the no-training local checks.**

Run one group at a time. Each group contains ten seeds. Output directories must not already exist; the runner does not overwrite or silently resume runs. Each seed executes in a fresh child process.

#### A. Released-code configuration — LR 0.001

**MLP**

```bash
python scripts/runner.py \
  --model mlp --configuration released-code \
  --seeds 100,101,102,103,104,105,106,107,108,109 \
  --upstream upstream-paper \
  --output-dir results/new_runs/released-code/mlp --run
```

**GCN**

```bash
python scripts/runner.py \
  --model gcn --configuration released-code \
  --seeds 100,101,102,103,104,105,106,107,108,109 \
  --upstream upstream-paper \
  --output-dir results/new_runs/released-code/gcn --run
```

**GAT**

```bash
python scripts/runner.py \
  --model gat --configuration released-code \
  --seeds 100,101,102,103,104,105,106,107,108,109 \
  --upstream upstream-paper \
  --output-dir results/new_runs/released-code/gat --run
```

**GraphSAGE**

```bash
python scripts/runner.py \
  --model graphsage --configuration released-code \
  --seeds 100,101,102,103,104,105,106,107,108,109 \
  --upstream upstream-paper \
  --output-dir results/new_runs/released-code/graphsage --run
```

**Full X-MoE — proposed complete method**

```bash
python scripts/runner.py \
  --model full-xmoe --configuration released-code \
  --seeds 100,101,102,103,104,105,106,107,108,109 \
  --upstream upstream-paper \
  --output-dir results/new_runs/released-code/full-xmoe --run
```

**X-MoE–CE — training control**

```bash
python scripts/runner.py \
  --model xmoe-ce --configuration released-code \
  --seeds 100,101,102,103,104,105,106,107,108,109 \
  --upstream upstream-paper \
  --output-dir results/new_runs/released-code/xmoe-ce --run
```

#### B. Paper-described configuration — LR 0.01

**MLP**

```bash
python scripts/runner.py \
  --model mlp --configuration paper-described \
  --seeds 100,101,102,103,104,105,106,107,108,109 \
  --upstream upstream-paper \
  --output-dir results/new_runs/paper-described/mlp --run
```

**GCN**

```bash
python scripts/runner.py \
  --model gcn --configuration paper-described \
  --seeds 100,101,102,103,104,105,106,107,108,109 \
  --upstream upstream-paper \
  --output-dir results/new_runs/paper-described/gcn --run
```

**GAT**

```bash
python scripts/runner.py \
  --model gat --configuration paper-described \
  --seeds 100,101,102,103,104,105,106,107,108,109 \
  --upstream upstream-paper \
  --output-dir results/new_runs/paper-described/gat --run
```

**GraphSAGE**

```bash
python scripts/runner.py \
  --model graphsage --configuration paper-described \
  --seeds 100,101,102,103,104,105,106,107,108,109 \
  --upstream upstream-paper \
  --output-dir results/new_runs/paper-described/graphsage --run
```

**Full X-MoE — proposed complete method**

```bash
python scripts/runner.py \
  --model full-xmoe --configuration paper-described \
  --seeds 100,101,102,103,104,105,106,107,108,109 \
  --upstream upstream-paper \
  --output-dir results/new_runs/paper-described/full-xmoe --run
```

**X-MoE–CE — training control**

```bash
python scripts/runner.py \
  --model xmoe-ce --configuration paper-described \
  --seeds 100,101,102,103,104,105,106,107,108,109 \
  --upstream upstream-paper \
  --output-dir results/new_runs/paper-described/xmoe-ce --run
```

### 13. Monitor a running experiment

Leave the training terminal open. In a separate Ubuntu terminal:

```bash
nvidia-smi -l 2
```

Use Ctrl+C in the monitoring terminal to stop monitoring. Ctrl+C in the training terminal interrupts training. Keep interrupted evidence; do not call it a completed ten-seed experiment.

The historical Full-X-MoE runner did not print epoch progress. A silent terminal did not mean it was frozen. The final wrapper should print a heartbeat without changing model computations. Memory use and training time can differ substantially across methods even at identical epoch counts.

### 14. Verify results and rebuild summaries

The public package includes the 120-run primary tables and [40 ablation scores](results/tables/pcgrad_ablation_40_runs.csv). Per-run predictions and checkpoints are archived locally. After importing the full reference evidence using `LOCAL_FIRST_STEPS.md`, verify that local evidence with the following command:

```bash
python scripts/verify_results.py \
  --results-dir results/reference --upstream upstream-paper
```

For a new execution series, keep all twelve groups beneath one dedicated series directory. Do not mix this directory with historical references or repeat runs.

```bash
python scripts/verify_results.py \
  --results-dir results/new_runs --upstream upstream-paper
```

Recompute the bundled 120-run reference summary without training:

```bash
python scripts/summarize_results.py --reference
```

After a complete new 120-run series is verified:

```bash
python scripts/summarize_results.py \
  --results-dir results/new_runs \
  --output-dir results/new_analysis
```

The summarizer requires exactly twelve method/configuration groups with seeds 100–109. It does not select the best repeat. The original two prediction schemas are normalized during verification without changing probabilities or labels.

### 15. Read the results correctly

Macro-F1 treats all three class F1 values equally. Accuracy and weighted F1 are strongly influenced by the Normal majority; inspect Phishing and Scamming separately.

| Quantity | Meaning |
|---|---|
| Mean | Average over ten runs |
| Sample SD (`ddof=1`) | Reported run-to-run spread, not a confidence interval |
| Population SD (`ddof=0`) | Additional recorded convention for explicit comparison |
| Percentage-point difference | Difference between two scores expressed as percentages |
| Paired difference | Two methods on the same saved partition/seed |
| Holm-adjusted p-value | Exploratory comparison after the specified multiplicity correction |

We use sample SD consistently in the main results table. The paper does not explicitly state the SD convention; its rounded Appendix-B values were previously found consistent with population SD. Do not silently mix conventions or interpret a paper SD as a confidence interval.

Exact equality across GPU executions is not promised. Verify input hashes, environment versions and settings when comparing executions. Exact GPU equality is not a condition for interpreting the reported mean and sample SD.

### 16. Understand why the configurations differ

| Setting | Released code | Paper description / reconstruction |
|---|---|---|
| Learning rate | 0.001 | 0.01 |
| GAT hidden width | 32 | 64 |
| GAT first-layer heads | 4 | 8 |
| GAT activation | ELU | ReLU |
| Checkpoint selection | Best validation Macro-F1 | Retained from released implementation |
| Exact repeated partitions/seeds | Not supplied in training scripts | Our saved reconstructed splits, not recovered author partitions |

The adapted GAT retains a single-head second layer with `concat=False`, its final classifier and other inherited implementation details. These choices are documented assumptions, not fully recovered author settings. GCN and GraphSAGE reconstruction changes the learning rate while retaining their released architecture.

### 17. Understand MLP, Full X-MoE and its CE-only control

Our MLP uses 23 inputs, two hidden layers of 64, LayerNorm, ReLU, dropout 0.5 and a three-class output. It ignores edges. The audited author materials did not supply a verified MLP training implementation or a fully recoverable architecture. Our MLP should not be relabeled as the author's exact MLP.

Full X-MoE is the proposed complete method. It combines the architecture with weighted CE, expert objectives, class-focused tasks, the Scamming specialist, the Normal–Phishing penalty and the existing fixed-order PCGrad procedure. Class-bias calibration is off.

X-MoE–CE is a training-control variant of that architecture. It uses weighted CE and ordinary backpropagation, with the additional objectives and PCGrad disabled. Calibration is also off. It is retained to assess the contribution of the complete additional training package, rather than as a separate proposed final method.

### PCGrad ablation

We ran 40 Full X-MoE ablation runs at 300 epochs: two gradient conditions across ten matched seeds in each of two configurations. Both conditions retain the eight training objectives, their weights, and mean aggregation of task gradients. The off condition disables gradient projection. The audit verified the run metadata and output hashes and confirmed matching initialization hashes for all 20 pairs.

| Configuration | PCGrad on, mean ± SD (%) | PCGrad off, mean ± SD (%) | Mean paired difference (pp) | On wins / ties / off wins |
|---|---:|---:|---:|---:|
| Released-code | 67.7787 ± 6.8333 | 58.0850 ± 4.3224 | +9.6937 | 10 / 0 / 0 |
| Paper-described | 72.1943 ± 5.6479 | 64.1750 ± 5.2227 | +8.0193 | 9 / 0 / 1 |

SD denotes the sample standard deviation across seeds. These 40 runs are reported separately from the 120 runs in the primary comparison.

### 18. Why extend the evaluation to clean/strict?

The BLTE reconstruction evaluates the reference methods and our proposed extensions while retaining the released feature matrix and graph topology. This preserves comparability with the available reference implementation, but leaves additional questions about preprocessing and graph visibility.

We therefore extend the evaluation through the clean and strict protocols provided in the `main` branch. The clean protocol fits preprocessing on training nodes only, while the strict protocol further restricts graph connections according to its documented visibility rules. These experiments assess whether the observed performance persists under more controlled information access. They complement the reconstruction and are reported separately because their input preparation and evaluation conditions differ.

Runtime versions also differ from the historical main-branch environment. Do not directly subtract a main-branch Full-X-MoE score from this audit's CE score and attribute the gap to PCGrad. Existing main results are not overwritten or automatically promoted by reproduction runs. Demo deployment remains on its separate branch.

### 19. Candidate directory layout

| Path | Content |
|---|---|
| `README.md` | Beginner environment and execution guide |
| `LOCAL_FIRST_STEPS.md` | No-training local verification/import steps |
| `docs/REPRODUCTION_REPORT.md` | Audit findings, assumptions and limits |
| `docs/SOURCE_REVIEW.md` | What was checked and what remains pending |
| `requirements/` | Recorded freeze and environment notes |
| `sources/author/` | Frozen upstream training scripts for provenance |
| `sources/paper_configuration/` | Audited fixed-mask paper adaptations |
| `sources/prepared/author/` and `paper/` | Unchanged primary GNN and CE scripts |
| `sources/prepared/model_source/models/` | Frozen MLP/X-MoE model modules imported by CE scripts |
| `sources/full/` | Frozen Full-X-MoE runner, adapter and model modules |
| `scripts/` | Portable wrappers, checks and summary tool |
| `splits/` | Ten unchanged partition arrays and a portable manifest |
| `results/tables/` | Primary results for 120 runs and Macro-F1 scores for 40 PCGrad ablation runs |
| `results/reference/` | Created only by verified local evidence import |
| `provenance/` | Source/input/package hashes, configurations and expected audit evidence |

Frozen CE and Full model modules are deliberately duplicated at their original relative import locations. Matching hashes establish identity. No new `pcgrad.py` is introduced: PCGrad stays in the existing Full runner. No training-loop refactor is performed to create this candidate.

Datasets, environments, backup folders, development diagnostics and unrelated manuscript/demo source are excluded. See the [upstream attribution](provenance/UPSTREAM_ATTRIBUTION.md) for the source license and dataset terms.

### 20. Troubleshooting

| Problem | Action |
|---|---|
| Terminal is PowerShell | Open VS Code through WSL and use an Ubuntu Bash terminal |
| `No module named pip` after venv creation | Check python3.10-venv support; preserve the incomplete environment |
| Wrong Python/library version | Record the mismatch; use a separate environment rather than modifying other experiments |
| CUDA unavailable | Check host driver, WSL GPU visibility and the installed Torch CUDA build |
| CUDA out of memory | Retain the error; do not silently reduce architecture or switch to CPU |
| No epoch text appears | Check GPU/process activity; the original Full runner can remain silent |
| Dataset hash differs | Stop and identify the input version |
| Output directory already contains runs | Use a new directory or a verified resume policy; never overwrite results silently |
| Scores differ slightly on a rerun | Verify settings/input hashes; exact GPU equality is not guaranteed |
| Scores differ substantially | Inspect source, splits, scaling, objectives and checkpoint selection before drawing conclusions |
| Local wrapper check fails | Save the error output and review the wrapper and runtime configuration |

### 21. Limitations and attribution

This work successfully executes released implementations and evaluates documented reconstructions. It does not establish exact recovery of the original paper experiment, the raw collection/labeling pipeline, all seven reference models, or generalization beyond the evaluated data.

Class minorities are small. Repeated partitions may overlap. Earlier test outcomes were inspected during development; later extensions are exploratory. All configurations and unfavorable outcomes are retained. Statistical evidence should be read with effect sizes, class metrics and these limits.

The upstream dataset and code remain subject to their own terms. Include the upstream license and attribution where redistribution is permitted. Cite the original BLTE dataset publication as appropriate and the multiclass paper. PCGrad is an existing technique; this repository's fixed-order implementation is not a claim of inventing PCGrad.

## Validation scope

Static package checks verify the distributed files. The local audit verifies the archived run evidence. GPU testing of the portable wrapper is a separate validation step.
