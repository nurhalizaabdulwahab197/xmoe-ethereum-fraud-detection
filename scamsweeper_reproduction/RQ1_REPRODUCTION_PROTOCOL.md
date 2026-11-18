# ScamSweeper RQ1 artifact-reproduction protocol

## Scope

RQ1 in the ScamSweeper paper is descriptive dataset analysis. It asks what
distinguishes Normal, Phishing and Web3 Scam account networks. It is not the
paper's model-comparison experiment.

This protocol reproduces RQ1 from the released processed NetworkX pickles. It
does not claim to recreate the authors' blockchain crawl independently.

## Frozen success criteria

RQ1 artifact reproduction is complete only when all applicable checks below
are resolved and recorded:

1. The three processed pickle identities and sizes are pinned.
2. Full node-table scans complete without executing pickle constructors.
3. Full adjacency scans complete without executing pickle constructors.
4. Node and edge counts match Table I exactly.
5. The meaning and count of source nodes (`#L`) are reconciled with the paper.
6. In-degree and out-degree frequency distributions are generated for all three
   graphs and qualitatively reproduce Figure 6.
7. The paper's `#SD Degree` definition is fixed and the three values are
   reproduced within a declared numerical tolerance.
8. Commands, code hashes, input hashes, runtime and output hashes are retained.

The independent blockchain collection is a separate tier and is not required
for artifact-level success. It cannot currently be claimed because the complete
original transaction collection and construction code were not released.

## Final verified status

Completed static scans reproduce all Table I structural counts:

| Dataset | Paper/observed nodes | Paper/observed edges |
|---|---:|---:|
| Normal | 12,042,066 | 142,750,370 |
| Phishing | 10,159,847 | 62,011,219 |
| Web3 Scam | 8,736,430 | 64,265,586 |

The full node tables contain 636, 4,701 and 3,125 records with `isp=1` for the
Normal, Phishing and Web3 Scam graphs. Normal and Web3 Scam match the paper's
`#L`; Phishing differs from the reported 4,905 by 204. The released artifact
does not contain enough provenance to recover those 204 records, so no labels
are synthesized.

Full static degree scans establish that `#SD Degree` is the population standard
deviation of total degree (`in-degree + out-degree`) over serialized adjacency
source records. This exactly reproduces the paper after rounding:

| Dataset | Adjacency sources | Computed population SD | Paper SD |
|---|---:|---:|---:|
| Normal | 12,042,066 | 3555.346071649701 | 3555.35 |
| Phishing | 10,159,176 | 1285.251995593048 | 1285.25 |
| Web3 Scam | 8,736,430 | 541.1005668306682 | 541.10 |

The phishing pickle has 10,159,847 node records but only 10,159,176 adjacency
source records. Treating the absent 671 sources as degree-zero nodes gives
1285.2096 and does not reproduce the paper; using the serialized adjacency
source scope gives 1285.2520 and does.

The final status is
**artifact reproduction complete with a documented source-count discrepancy**:

- structural node/edge counts: pass;
- full in/out-degree distributions and Figure 6-style plot: pass;
- all three `#SD Degree` values: pass;
- phishing `#L`: documented paper/artifact discrepancy (4,905 versus 4,701);
- independent blockchain recrawl: outside this artifact-reproduction scope.

## Machine-verifiable Table I check

From the workspace root:

```powershell
.\.venv\Scripts\python.exe .\scamsweeper_reproduction\rq1_verify_table1.py `
  --node-report .\scamsweeper_reproduction\results\rq1_node_scan.json `
  --edge-report .\scamsweeper_reproduction\results\rq1_degree_run01\summary.json `
  --output .\scamsweeper_reproduction\results\rq1_table1_verification.json
```

The structural verifier deliberately reports the unresolved `#L` discrepancy.
The final classification is performed only after the full degree scan.

## Full degree scan, figure and final report

```powershell
node --max-old-space-size=8192 `
  .\scamsweeper_reproduction\rq1_degree_scan.mjs `
  --boundaries .\scamsweeper_reproduction\results\rq1_node_scan.json `
  --output-dir .\scamsweeper_reproduction\results\rq1_degree_run01

.\.venv\Scripts\python.exe .\scamsweeper_reproduction\rq1_plot_figure6.py `
  --input-dir .\scamsweeper_reproduction\results\rq1_degree_run01 `
  --output .\scamsweeper_reproduction\results\rq1_figure6_reproduced.png

.\.venv\Scripts\python.exe .\scamsweeper_reproduction\rq1_finalize_reproduction.py `
  --table-verification .\scamsweeper_reproduction\results\rq1_table1_verification.json `
  --degree-dir .\scamsweeper_reproduction\results\rq1_degree_run01 `
  --figure .\scamsweeper_reproduction\results\rq1_figure6_reproduced.png `
  --output .\scamsweeper_reproduction\results\rq1_reproduction_final.json
```

## Relationship to later model experiments

- The existing derived-RQ1 X-MoE benchmark is a separate three-class extension.
- ScamSweeper versus X-MoE model comparison belongs to RQ2, using the same RQ2
  samples, partitions and metrics.
- BERT4ETH versus an X-MoE sequence variant belongs to a common phishing
  benchmark, not to RQ1.

