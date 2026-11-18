# ScamSweeper RQ1 artifact reproduction

This package reproduces the descriptive RQ1 analysis in *ScamSweeper:
Detecting Illegal Accounts in Web3 Scams via Transactions Analysis* from the
authors' released `RQ1.zip` artifact.

It reproduces:

- Table I node and edge counts;
- the released `isp=1` source-marker counts;
- population standard deviation of total degree;
- complete in-degree and out-degree frequency distributions;
- a Figure 6-style visualization;
- input, output, code and environment provenance.

It does **not** train ScamSweeper. Model training starts in paper RQ2. It also
does not independently recreate the authors' Ethereum crawl.

## Validated result

An independent Linux run on 2026-11-18 used Python 3.10.12, Node.js 22.23.5
and released archive SHA-256
`4481bddf4cc75cd35e8ecebf51eb20d3cd0154c1dbfa899848467219b55b8c87`.

| Graph | Nodes | Adjacency sources | Edges | Computed population SD | Paper SD | Match |
|---|---:|---:|---:|---:|---:|:---:|
| Normal | 12,042,066 | 12,042,066 | 142,750,370 | 3555.346071649701 | 3555.35 | Yes |
| Phishing | 10,159,847 | 10,159,176 | 62,011,219 | 1285.251995593048 | 1285.25 | Yes |
| Web3 Scams | 8,736,430 | 8,736,430 | 64,265,586 | 541.1005668306682 | 541.10 | Yes |

Final status:

```text
artifact_reproduction_complete_with_documented_source_count_discrepancy
```

Table I reports 4,905 phishing source nodes, while the released pickle contains
4,701 node records with `isp=1`. No missing labels are synthesized.

## Safety design

The scanners never call `pickle.load()` on the multi-gigabyte NetworkX files.
They statically interpret the required primitive pickle opcodes. Raw files are
checked for changes during scanning, and outputs are not silently overwritten.

## Package files

| File | Purpose |
|---|---|
| `prepare_rq1.py` | Verify and safely extract the nested released archives |
| `rq1_degree_scan.mjs` | Full adjacency and degree scan |
| `rq1_verify_table1.py` | Machine-verifiable Table I comparison |
| `rq1_plot_figure6.py` | Degree-frequency visualization |
| `rq1_finalize_reproduction.py` | Final status, discrepancies and hashes |
| `requirements-rq1.txt` | Minimal Python dependency pin |
| `RQ1_REPRODUCTION_PROTOCOL.md` | Scope and statistical definition |

The node-prefix scanner and primitive opcode reader are in `../tools/`.

## Complete step-by-step reproduction

All commands required for a clean Windows or Linux reproduction are included
below. The separate `RQ1_FROM_SCRATCH.md` file mirrors this procedure for
users who prefer a standalone protocol.

## 1. Hardware and software

Recommended minimum:

- 64-bit Windows or Linux;
- 16 GB RAM (the Node process is allowed an 8 GB heap);
- 25 GB free disk space;
- Python 3.10, 3.11 or 3.12;
- Node.js 22 LTS;
- Git.

A GPU is not used by RQ1. Running this on a GPU server is fine, but CPU, RAM and
local-disk speed determine the runtime. The released zip is 4.46 GB and the
three extracted pickles total approximately 10.73 GB.

Record the machine details in every reported run:

```text
OS and version:
CPU:
RAM:
Python version:
Node version:
Git commit:
RQ1.zip SHA-256:
```

## 2. Clone and freeze the code revision

```bash
git clone --branch rq1-reproduction \
  https://github.com/nurhalizaabdulwahab197/xmoe-ethereum-fraud-detection.git
cd xmoe-ethereum-fraud-detection
git branch --show-current
git rev-parse HEAD
```

Until this branch is merged into `main`, the branch name must be specified.
Expected branch: `rq1-reproduction`.

The commit used for a publication must contain these files:

- `scamsweeper_reproduction/prepare_rq1.py`;
- `scamsweeper_reproduction/rq1_degree_scan.mjs`;
- `scamsweeper_reproduction/rq1_verify_table1.py`;
- `scamsweeper_reproduction/rq1_plot_figure6.py`;
- `scamsweeper_reproduction/rq1_finalize_reproduction.py`;
- `tools/rq1_node_prefix_audit.py`;
- `tools/rq1_pickle_primitives.mjs`.

Do not commit the multi-gigabyte pickle files to Git.

## 3. Download the released RQ1 archive

Download `RQ1.zip` from the authors' released artifact and place it in a local
data directory, for example `external_data/RQ1.zip`.

Audited archive identity:

```text
Size:    4,461,502,321 bytes
SHA-256: 4481bddf4cc75cd35e8ecebf51eb20d3cd0154c1dbfa899848467219b55b8c87
```

The preparation script refuses a different archive hash. This protects the
experiment from silently using another dataset version.

## 4. Detect the available environment first

Do not assume that Python 3.12, Conda, Node, or an HPC module system exists.
Run:

```bash
command -v python3 || true
command -v python || true
command -v conda || true
command -v mamba || true
command -v node || true
python3 --version 2>/dev/null || true
node --version 2>/dev/null || true
```

Use an available Python version from 3.10 through 3.12. If `python3` reports
3.10 or newer, it is sufficient; do not call `python3.12` unless that exact
command exists.

## 5. Create an isolated Python environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv-rq1
.\.venv-rq1\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\scamsweeper_reproduction\requirements-rq1.txt
python --version
node --version
```

Linux:

```bash
python3 -m venv .venv-rq1
source .venv-rq1/bin/activate
python -m pip install --upgrade pip
python -m pip install -r scamsweeper_reproduction/requirements-rq1.txt
python --version
node --version
```

If `python3 -m venv` fails because `venv` is unavailable, use a site-provided
Conda/Mamba installation if one was detected. Do not run `conda` or `module`
commands when `command -v` did not find them.

## 6. Install Node.js when it is not provided

The scan requires Node.js, but root access, Conda and npm are not required.
First identify the CPU architecture:

```bash
uname -m
command -v curl || command -v wget
```

For `x86_64`, install the official Node 22 binary inside the project:

```bash
mkdir -p .local
curl -fL -o /tmp/node-v22.23.5-linux-x64.tar.xz \
  https://nodejs.org/dist/v22.23.5/node-v22.23.5-linux-x64.tar.xz
tar -xJf /tmp/node-v22.23.5-linux-x64.tar.xz -C .local
export PATH="$PWD/.local/node-v22.23.5-linux-x64/bin:$PATH"
node --version
npm --version
```

For `aarch64`/`arm64`, replace both `linux-x64` occurrences with `linux-arm64`.
Repeat the `export PATH=...` command after reconnecting, or place it in the
job script. Record the exact Node version used.

## 7. Confirm that the reproduction package exists

Before installing requirements, check the cloned revision:

```bash
test -f scamsweeper_reproduction/requirements-rq1.txt || echo "MISSING: reproduction package has not been pushed"
test -f scamsweeper_reproduction/prepare_rq1.py || echo "MISSING: prepare_rq1.py"
test -f tools/rq1_node_prefix_audit.py || echo "MISSING: node scanner"
test -f tools/rq1_pickle_primitives.mjs || echo "MISSING: pickle parser"
```

If any file is missing, stop. Push the reproduction package to GitHub and then
run `git pull`; installing from a nonexistent requirements file cannot work.

With all files present:

```bash
python -m pip install -r scamsweeper_reproduction/requirements-rq1.txt
```

Only Matplotlib is required from PyPI. The graph pickle is never unpickled and
NetworkX, PyTorch, CUDA and PyTorch Geometric are not required.

The requirements file pins Matplotlib 3.10.8 so the same environment works on
Python 3.10, 3.11 and 3.12. Matplotlib 3.11.x requires Python 3.11 or newer and
must not be used for the documented Python 3.10 path.

## 8. Extract and hash the released graphs

Windows PowerShell:

```powershell
python .\scamsweeper_reproduction\prepare_rq1.py `
  --archive .\external_data\RQ1.zip `
  --output-dir .\data\raw\RQ1\unpacked
```

Linux:

```bash
python scamsweeper_reproduction/prepare_rq1.py \
  --archive external_data/RQ1.zip \
  --output-dir data/raw/RQ1/unpacked
```

The output directory must not already exist. The expected layout is:

```text
data/raw/RQ1/unpacked/
  manifest.json
  normal/xlabel_normal.pickle
  phish/phish.pickle
  scams/scams.pickle
```

Keep `manifest.json` with the experiment results; it records every input hash.

## 9. Scan the complete node tables

Windows PowerShell:

```powershell
python .\tools\rq1_node_prefix_audit.py `
  --root .\data\raw\RQ1\unpacked `
  --full-node-table `
  --max-mib 8192 `
  --output .\scamsweeper_reproduction\results\rq1_node_scan.json
```

Linux:

```bash
python tools/rq1_node_prefix_audit.py \
  --root data/raw/RQ1/unpacked \
  --full-node-table \
  --max-mib 8192 \
  --output scamsweeper_reproduction/results/rq1_node_scan.json
```

This pass records node counts, `isp` values, the exact adjacency boundary and
pickle memo state. It statically interprets primitive pickle opcodes; it does
not execute pickle constructors.

Expected node counts are 12,042,066 Normal, 10,159,847 Phishing and 8,736,430
Web3 Scam. Expected `isp=1` counts in the released artifact are 636, 4,701 and
3,125 respectively.

## 10. Scan every edge and compute degree statistics

The output directory must be new. Change `run01` when repeating an experiment.

Windows PowerShell:

```powershell
node --max-old-space-size=8192 `
  .\scamsweeper_reproduction\rq1_degree_scan.mjs `
  --boundaries .\scamsweeper_reproduction\results\rq1_node_scan.json `
  --output-dir .\scamsweeper_reproduction\results\rq1_degree_run01
```

Linux:

```bash
node --max-old-space-size=8192 \
  scamsweeper_reproduction/rq1_degree_scan.mjs \
  --boundaries scamsweeper_reproduction/results/rq1_node_scan.json \
  --output-dir scamsweeper_reproduction/results/rq1_degree_run01
```

Do not interrupt the process. It scans all 269,027,175 edge records. The final
line for each graph reports nodes, adjacency sources, edges and population SD.

## 11. Verify Table I

```bash
python scamsweeper_reproduction/rq1_verify_table1.py \
  --node-report scamsweeper_reproduction/results/rq1_node_scan.json \
  --edge-report scamsweeper_reproduction/results/rq1_degree_run01/summary.json \
  --output scamsweeper_reproduction/results/rq1_table1_run01.json
```

PowerShell accepts the same command on one line, or replace each `\` line
continuation with a backtick. Required result:

```text
structural_counts_status: pass
source_marker_status: unresolved
degree_statistics_status: pass
overall_status: table_i_reproduction_complete_with_documented_source_count_discrepancy
```

`unresolved` is expected, not a failed run: Table I reports 4,905 phishing
source nodes, while the released pickle contains 4,701 `isp=1` records.

## 12. Recreate the degree-distribution figure

```bash
python scamsweeper_reproduction/rq1_plot_figure6.py \
  --input-dir scamsweeper_reproduction/results/rq1_degree_run01 \
  --output scamsweeper_reproduction/results/rq1_figure6_run01.png
```

Inspect the PNG and confirm that it contains Normal, Phishing and Web3 Scams
panels with log-scaled in-degree and out-degree frequency curves.

## 13. Produce the final machine-readable verdict

```bash
python scamsweeper_reproduction/rq1_finalize_reproduction.py \
  --table-verification scamsweeper_reproduction/results/rq1_table1_run01.json \
  --degree-dir scamsweeper_reproduction/results/rq1_degree_run01 \
  --figure scamsweeper_reproduction/results/rq1_figure6_run01.png \
  --output scamsweeper_reproduction/results/rq1_reproduction_run01.json
```

Successful output must contain:

```text
status: artifact_reproduction_complete_with_documented_source_count_discrepancy
structural_counts_reproduced: true
sd_degree_reproduced: true
degree_distributions_reproduced: true
```

The three total-degree population SD values must round to 3555.35, 1285.25 and
541.10. For Phishing, the SD denominator is the 10,159,176 serialized adjacency
source records, not all 10,159,847 node records.

## 14. Record the environment and output hashes

Linux:

```bash
{
  echo "DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "HOST=$(hostname)"
  echo "OS=$(uname -a)"
  echo "PYTHON=$(python --version 2>&1)"
  echo "NODE=$(node --version)"
  echo "GIT_BRANCH=$(git branch --show-current)"
  echo "GIT_COMMIT=$(git rev-parse HEAD)"
  echo "RQ1_ZIP_SHA256=$(sha256sum external_data/RQ1.zip | awk '{print $1}')"
} | tee scamsweeper_reproduction/results/rq1_environment_run01.txt

sha256sum \
  data/raw/RQ1/unpacked/manifest.json \
  scamsweeper_reproduction/results/rq1_node_scan.json \
  scamsweeper_reproduction/results/rq1_degree_run01/summary.json \
  scamsweeper_reproduction/results/rq1_degree_run01/*_degree_report.json \
  scamsweeper_reproduction/results/rq1_degree_run01/*_degree_frequency.csv \
  scamsweeper_reproduction/results/rq1_table1_run01.json \
  scamsweeper_reproduction/results/rq1_figure6_run01.png \
  scamsweeper_reproduction/results/rq1_reproduction_run01.json \
  scamsweeper_reproduction/results/rq1_environment_run01.txt \
  > scamsweeper_reproduction/results/rq1_output_sha256_run01.txt
```

If the archive is stored elsewhere, replace `external_data/RQ1.zip` with its
actual path.

## 15. Preserve a reproducibility bundle

Publish the small outputs, not the raw pickles:

```text
manifest.json
rq1_node_scan.json
rq1_degree_run01/summary.json
rq1_degree_run01/*_degree_report.json
rq1_degree_run01/*_degree_frequency.csv
rq1_table1_run01.json
rq1_figure6_run01.png
rq1_reproduction_run01.json
rq1_environment_run01.txt
rq1_output_sha256_run01.txt
```

Also record `git rev-parse HEAD`, `python --version`, `node --version`, OS, CPU,
RAM and wall-clock runtime. Never describe this as an independent blockchain
data-collection reproduction; it is a complete reproduction of RQ1 statistics
from the authors' released artifacts.

## Troubleshooting

- `Output directory already exists`: use a new run name; outputs are not
  overwritten intentionally.
- Node exits with heap error: confirm the 64-bit Node build and keep
  `--max-old-space-size=8192`; if needed use 12288 on a machine with sufficient
  RAM.
- Archive hash mismatch: do not bypass the check. Record the new archive as a
  separate dataset version and investigate its provenance.
- Missing `tools/rq1_pickle_primitives.mjs`: the Git commit is incomplete.
- `matplotlib==3.11.x` rejects Python 3.10: pull the current requirements file,
  which pins Python-3.10-compatible Matplotlib 3.10.8.
- `python3.12`, `conda`, or `module` is not found: use an available Python
  3.10-3.12 executable and the documented local Node binary installation.
- `cat ...rq1_output_sha256_run01.txt` reports no such file: run the preceding
  `sha256sum ... >` command first; `cat` only displays an existing file.
- GPU utilization remains zero: this is correct because RQ1 is a static graph
  statistics experiment.

## What comes next

RQ2/RQ3 model reproduction uses the separate `RQ2&3` artifact. RQ4 phishing
model reproduction uses the RQ4 artifact. An X-MoE three-class extension built
from RQ1-derived data must not be presented as the paper's RQ1 model
reproduction, because paper RQ1 contains no classifier.
