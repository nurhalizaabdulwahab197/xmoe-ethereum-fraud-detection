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

## Start here

Follow the complete Windows/Linux procedure in
[`RQ1_FROM_SCRATCH.md`](RQ1_FROM_SCRATCH.md). Do not skip directly to a later
command: extraction, node-boundary discovery and full edge scanning are
separate validation stages.

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

## What comes next

RQ2/RQ3 model reproduction uses the separate `RQ2&3` artifact. RQ4 phishing
model reproduction uses the RQ4 artifact. An X-MoE three-class extension built
from RQ1-derived data must not be presented as the paper's RQ1 model
reproduction, because paper RQ1 contains no classifier.
