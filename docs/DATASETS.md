# Dataset provenance, features and verification

## BLTE processed multiclass artifact

Obtain the released graph and CSV from the
[multiclass upstream repository](https://github.com/salam-ammari/Graph-Neural-Network-Based-Fraudulent-Address-Classification-in-Ethereum-Network).
The study uses those processed inputs rather than independently reconstructing
the raw Ethereum collection. Hashes are in the root README. The graph has 73,033
nodes and 6,588 directed edges, including 67,427 isolated Normal addresses.

The clean adapter uses CSV values, not the graph's already transformed features,
and fits training-only scaling. CSV columns excluded from predictors are
`Address`, `Scam`, `attack` and `attack_encoded`. Feature order is:

```text
all_degree, in_degree, out_degree, unique in_degree, unique out_degree,
Avg amount incoming, Avg amount outgoing, Total amount incoming,
Total amount outgoing, Max amount incoming, Max amount outgoing,
Min amount incoming, Min amount outgoing, Avg time incoming,
Avg time outgoing, Total transaction time, Active Duration,
Clustering coefficient, Mean time interval, Max time interval,
Min time interval, Avg gas price, Avg gas limit
```

Counts in CSV-derived attributes need not numerically equal counts recomputed
from the released sparse edge array. We do not equate those representations.
Source topology is label-correlated; the clean transductive study is separate
from the repository's strict visibility experiment.

## ScamSweeper-derived RQ1

Source: [Online Artifact of ScamSweeper](https://figshare.com/s/65d00a4c50c9d5188c06),
with the [ScamSweeper paper](https://csxqli.github.io/files/ScamSweeper.pdf).
The source describes a processed NetworkX Ethereum graph crawled by second-order
BFS in the first 18 million block heights. Its Normal/Phishing sources and Web3
scam sources differ; a source-list Scams label is not automatically equivalent
to every BLTE Scamming annotation.

RQ1 graph pickle SHA-256 values:

| File | SHA-256 |
|---|---|
| `normal/xlabel_normal.pickle` | `616edbf01a61f6db84158a279f471f7d98743ee8e0f1779189c77e2c8a516465` |
| `phish/phish.pickle` | `49b16315c15392551e02f928d5adb6ca309985b46da0df81a116356ad74f2506` |
| `scams/scams.pickle` | `1c4a9de26e10d4b87570e3a92f62a388b3b5b5d97bd3c933cc39cdd9a65c9839` |

Explicit account lists in RQ4 contain 636 Normal, 1,165 Phishing and 3,125 Scams.
Thirteen Scams entries were flagged in the pre-training source-marker/case review
and excluded, giving the frozen 4,913-account cohort. Their canonical-address
SHA-256 values and the list-member hashes are in
`provenance/rq1_label_policy.json`; raw account lists are not redistributed here.
The public preparation applies this frozen review decision; it does not claim
to independently re-establish ground truth or rerun the original case review.

The retained graph includes all indexed available incident observations of this
cohort across the three source graphs. It contains 1,322,252 canonical nodes,
3,856,855 unique directed pairs (435 observed self-loops), and 1,317,339 unknown
context nodes. Context labels are **-1**, and never become supervised Normal.
Labeled classes are Normal=0, Phishing=1, Scams=2.

Feature definitions, in order:

| Feature | Definition in the retained observation graph |
|---|---|
| `distinct_in_counterparties` | Count of distinct incoming peers, excluding self |
| `distinct_out_counterparties` | Count of distinct outgoing peers, excluding self |
| `reciprocal_counterparties` | Count of distinct non-self peers with both directions observed |
| `active_days` | Count of observed integer Unix-day bins over incident records |
| `activity_span_seconds` | Maximum minus minimum valid incident timestamp |

All five features are computed for context nodes as well as labeled nodes. They
describe this cohort-incident observed graph, not every address's complete
blockchain history. Source observations can overlap across archives; event
frequency and monetary sum features are excluded rather than treating all
observations as unique blockchain transactions. Transaction hashes/units were
not sufficiently established to invent amount/gas features or claim exact
cross-archive transaction deduplication.

Timestamp checks require integral values, consistency when a numeric key and
timestamp attribute coexist, and the declared 2015-01-01 to 2026-10-30 Unix-second
plausibility window. Independent source-unit confirmation remains incomplete.
Nineteen malformed non-address observation occurrences were quarantined; the
paired time/endpoint examples do not represent 38 distinct bad transactions.

Prepared graph and per-seed normalized-array hashes are in
`provenance/rq1_expected_arrays.json`. Successful input verification requires
exact array hashes and independently recomputed scaler/mask equality. The
node-boundary metadata is usable only with the pinned raw pickle hashes.

## Relationship between datasets

The same class IDs make the output task comparable but do not erase annotation
source differences. The prior overlap audit found 231 shared labeled addresses,
including 90 annotation disagreements, and 11,670 BLTE addresses in RQ1 graph
nodes. These counts are disclosed rather than silently relabeling or deleting
test accounts. This is not an address-disjoint external validation claim.

Features are dataset-native (23 versus five). No BLTE model weights or scalers
are reused for RQ1. All models share the same inputs/splits within each dataset.
Input weights, class-weight values and normalized values are naturally refitted
from each dataset's training population using the same formulas.

## Rights and availability

Code is MIT-licensed by the repository author. That license does not grant rights
to upstream datasets. Download artifacts from the source, retain provider terms,
and cite the dataset origins and relevant multiclass/ScamSweeper publications.
No raw graph, account-list archive or prepared graph array is distributed in this
branch. Provider access or availability may change; record artifact hashes and
do not silently substitute a different source version.
